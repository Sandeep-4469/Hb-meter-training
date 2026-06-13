"""
Classifier Training — with gender/age + WHO sex-specific thresholds
====================================================================
1. Merges gender & age from AIIMS hb_values.csv into feature set
2. Applies WHO anaemia thresholds per sex/age:
     Men   (age ≥ 15)  : HB < 13 g/dL
     Women (age ≥ 15)  : HB < 12 g/dL
     Children 5–14 yr  : HB < 11.5 g/dL
     Children < 5 yr   : HB < 11 g/dL
3. Adds gender (0=F,1=M) and age as features
4. Trains 6 classifiers with 5-fold GroupKFold + sample weights
5. Saves best model + all per-fold models

Classifiers
-----------
  Logistic Regression, SVM (RBF), Random Forest,
  Gradient Boosting, LightGBM, XGBoost

Outputs
-------
  saved_models/clf_<name>.pkl       ← full-train final model
  results/clf_report.txt
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                              balanced_accuracy_score, classification_report)
import lightgbm as lgb
import xgboost as xgb

AIIMS_CSV  = Path("/data2/sandeep/AIIMS_DATA/hb_values.csv")
RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

SEED    = 42
N_FOLDS = 5

# ── WHO thresholds ────────────────────────────────────────────────────────────
def who_threshold(gender, age):
    """Return anaemia threshold in g/dL per WHO 2011."""
    if age < 5:   return 11.0
    if age < 15:  return 11.5
    if gender == "M": return 13.0
    return 12.0   # female adult

# ── load demographics ─────────────────────────────────────────────────────────
demo = pd.read_csv(AIIMS_CSV)
demo.columns = demo.columns.str.strip().str.lower().str.replace(" ", "_")
demo["gender"] = demo["gender"].str.strip().str.upper()
# fix obvious age errors (e.g. 331 → 33)
demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
demo.loc[demo["age"] > 120, "age"] = demo.loc[demo["age"] > 120, "age"] / 10
demo["vid_norm"] = (demo["video_name"].str.strip().str.lower()
                    .str.replace(" ", "_", regex=False)
                    .str.replace(r"\.(mkv|mp4|avi)$", "", regex=True))
demo = demo.dropna(subset=["vid_norm", "gender", "age"])
demo = demo[demo["gender"].isin(["M", "F"])].copy()
demo["gender_bin"] = (demo["gender"] == "M").astype(int)
demo["who_thr"]    = demo.apply(lambda r: who_threshold(r["gender"], r["age"]), axis=1)
demo = demo[["vid_norm", "gender_bin", "age", "who_thr"]].drop_duplicates("vid_norm")

print(f"Demographics loaded: {len(demo)} records")
print(f"  Gender: M={int((demo['gender_bin']==1).sum())}  F={int((demo['gender_bin']==0).sum())}")
print(f"  Age range: {demo['age'].min():.0f}–{demo['age'].max():.0f} yrs")
print(f"  WHO thresholds: {demo['who_thr'].value_counts().to_dict()}")

# ── load features ─────────────────────────────────────────────────────────────
tr_feat = pd.read_csv(RESULTS / "features_combined_train.csv")
te_feat = pd.read_csv(RESULTS / "features_combined_test.csv")

def merge_demo(feat_df, demo_df):
    merged = feat_df.merge(demo_df, left_on="video_id", right_on="vid_norm", how="left")
    # fallback: patients with no demo get gender_bin=0.5 (unknown), age=median, threshold=12
    n_missing = merged["gender_bin"].isna().sum()
    if n_missing:
        print(f"  {n_missing} videos missing demographics → using defaults (F, age=30, thr=12)")
    merged["gender_bin"] = merged["gender_bin"].fillna(0)
    merged["age"]        = merged["age"].fillna(merged["age"].median())
    merged["who_thr"]    = merged["who_thr"].fillna(12.0)
    return merged

tr = merge_demo(tr_feat, demo)
te = merge_demo(te_feat, demo)

meta_cols = {"video_id","hb_value","split","protocol","vid_norm","who_thr"}
feat_cols  = [c for c in tr_feat.columns if c not in {"video_id","hb_value","split","protocol"}]
# append gender + age as features
feat_cols_ext = feat_cols + ["gender_bin", "age"]

X_tr_raw = tr[feat_cols_ext].astype(float).fillna(0).values
y_tr_hb  = tr["hb_value"].astype(float).values
y_tr_thr = tr["who_thr"].astype(float).values
y_tr     = (y_tr_hb < y_tr_thr).astype(int)   # 1 = anaemic (WHO sex-specific)
ids_tr   = tr["video_id"].values

X_te_raw = te[feat_cols_ext].astype(float).fillna(0).values
y_te_hb  = te["hb_value"].astype(float).values
y_te_thr = te["who_thr"].astype(float).values
y_te     = (y_te_hb < y_te_thr).astype(int)

# winsorise using training percentiles (1st–99th)
win_lo = np.percentile(X_tr_raw, 1, axis=0)
win_hi = np.percentile(X_tr_raw, 99, axis=0)
X_tr   = np.clip(X_tr_raw, win_lo, win_hi)
X_te   = np.clip(X_te_raw, win_lo, win_hi)

n_pos_tr = y_tr.sum(); n_neg_tr = (1-y_tr).sum()
n_pos_te = y_te.sum(); n_neg_te = (1-y_te).sum()
print(f"\nTrain  anaemic={n_pos_tr}  normal={n_neg_tr}  "
      f"({n_pos_tr/len(y_tr)*100:.0f}% anaemic, WHO thresholds)")
print(f"Test   anaemic={n_pos_te}  normal={n_neg_te}  "
      f"({n_pos_te/len(y_te)*100:.0f}% anaemic, WHO thresholds)")
print(f"Features: {X_tr.shape[1]}  (1060 spectral + gender + age)\n")

# ── sample weights ────────────────────────────────────────────────────────────
def clf_weights(y, cap=5.0):
    pos_rate = y.mean()
    w = np.where(y == 1, 1.0 / (pos_rate + 1e-8),
                         1.0 / (1 - pos_rate + 1e-8))
    w /= w.min()
    return np.clip(w, None, cap)

w_tr = clf_weights(y_tr)

# ── metrics ───────────────────────────────────────────────────────────────────
def clf_metrics(yt, yp_prob, yp_label=None):
    if yp_label is None:
        yp_label = (yp_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp_label, labels=[0,1]).ravel()
    sens    = tp / (tp + fn) if (tp+fn) > 0 else 0.
    spec    = tn / (tn + fp) if (tn+fp) > 0 else 0.
    ppv     = tp / (tp + fp) if (tp+fp) > 0 else 0.
    npv     = tn / (tn + fn) if (tn+fn) > 0 else 0.
    f1      = f1_score(yt, yp_label, zero_division=0)
    bal_acc = balanced_accuracy_score(yt, yp_label)
    try:    auc = roc_auc_score(yt, yp_prob)
    except: auc = float("nan")
    return dict(tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),
                sens=sens,spec=spec,ppv=ppv,npv=npv,
                f1=f1,bal_acc=bal_acc,auc=auc)

# ── classifier configs ────────────────────────────────────────────────────────
def make_classifiers():
    return [
        ("LogReg", Pipeline([
            ("sel", SelectKBest(f_classif, k=80)),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                       class_weight="balanced", random_state=SEED)),
        ]), False),

        ("SVM", Pipeline([
            ("sel", SelectKBest(f_classif, k=80)),
            ("sc",  StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]), False),

        ("RF", RandomForestClassifier(
            n_estimators=300, max_features=0.3, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        ), True),

        ("GBM", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=SEED,
        ), True),

        ("LGB", lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            is_unbalance=True, random_state=SEED, verbose=-1, n_jobs=-1,
        ), True),

        ("XGB", xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            scale_pos_weight=n_neg_tr / (n_pos_tr + 1e-8),
            random_state=SEED, verbosity=0, n_jobs=-1, eval_metric="logloss",
        ), True),
    ]

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
kf = GroupKFold(n_splits=N_FOLDS)
results = {}

print("=" * 65)
print("  5-FOLD CV (GroupKFold by video_id)")
print("=" * 65)

for cname, model, use_sw in make_classifiers():
    oof_prob  = np.zeros(len(X_tr))
    oof_label = np.zeros(len(X_tr), dtype=int)

    for fi, (tri, vai) in enumerate(kf.split(X_tr, y_tr, ids_tr)):
        Xf, yf, wf = X_tr[tri], y_tr[tri], w_tr[tri]
        Xv         = X_tr[vai]

        if use_sw:
            model.fit(Xf, yf, sample_weight=wf)
        else:
            model.fit(Xf, yf)

        prob = model.predict_proba(Xv)[:, 1]
        oof_prob[vai]  = prob
        oof_label[vai] = model.predict(Xv)

    m = clf_metrics(y_tr, oof_prob, oof_label)
    results[cname] = {"oof": m}
    print(f"  {cname:<8}  AUC={m['auc']:.3f}  Sens={m['sens']:.3f}  "
          f"Spec={m['spec']:.3f}  F1={m['f1']:.3f}  BalAcc={m['bal_acc']:.3f}")

# ── retrain on full train, evaluate test, save ────────────────────────────────
print(f"\n{'='*65}")
print("  FINAL MODEL — full train → test evaluation + save")
print(f"{'='*65}")

best_auc   = -1
best_cname = None
report_blocks = []

for cname, model, use_sw in make_classifiers():
    if use_sw:
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        model.fit(X_tr, y_tr)

    prob  = model.predict_proba(X_te)[:, 1]
    label = model.predict(X_te)
    m     = clf_metrics(y_te, prob, label)
    results[cname]["test"] = m

    # save
    save_path = MODELS_DIR / f"clf_{cname.lower()}.pkl"
    joblib.dump({"model": model, "winsor_lo": win_lo, "winsor_hi": win_hi,
                 "feat_cols": feat_cols_ext, "who_thresholds": "applied_at_label_time"},
                save_path)

    if m["auc"] > best_auc:
        best_auc = m["auc"]; best_cname = cname

    block = [
        f"\n  ── {cname}",
        f"  OOF : AUC={results[cname]['oof']['auc']:.3f}  Sens={results[cname]['oof']['sens']:.3f}  "
        f"Spec={results[cname]['oof']['spec']:.3f}  F1={results[cname]['oof']['f1']:.3f}",
        f"  Test: AUC={m['auc']:.3f}  Sens={m['sens']:.3f}  Spec={m['spec']:.3f}  "
        f"F1={m['f1']:.3f}  BalAcc={m['bal_acc']:.3f}",
        f"  Confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  "
        f"PPV={m['ppv']:.3f}  NPV={m['npv']:.3f}",
        f"  Saved → {save_path.name}",
    ]
    print("\n".join(block))
    report_blocks.append("\n".join(block))

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  SUMMARY — Test set (n=95, WHO sex-specific thresholds)")
print(f"  Anaemic: n={int(y_te.sum())}  Normal: n={int((1-y_te).sum())}")
print(f"{'='*65}")
print(f"  {'Model':<8}  {'AUC':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}  {'BalAcc':>7}  {'TP':>3}  {'FP':>3}  {'TN':>3}  {'FN':>3}")
print(f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*3}  {'─'*3}  {'─'*3}  {'─'*3}")
for cname, _, _ in make_classifiers():
    m = results[cname]["test"]
    star = " ←" if cname == best_cname else ""
    print(f"  {cname:<8}  {m['auc']:>6.3f}  {m['sens']:>6.3f}  {m['spec']:>6.3f}  "
          f"{m['f1']:>6.3f}  {m['bal_acc']:>7.3f}  {m['tp']:>3}  {m['fp']:>3}  "
          f"{m['tn']:>3}  {m['fn']:>3}{star}")

# also show gender breakdown on test
print(f"\n  Gender breakdown (test):")
te_demo = te[["video_id","hb_value","who_thr"]].copy()
te_demo["gender_bin"] = te["gender_bin"].astype(int)
te_demo["anaemic_who"] = y_te
male_mask   = te_demo["gender_bin"] == 1
female_mask = te_demo["gender_bin"] == 0
print(f"  Male   n={male_mask.sum():3d}  anaemic={te_demo[male_mask]['anaemic_who'].sum():3d}  "
      f"(threshold 13 g/dL)")
print(f"  Female n={female_mask.sum():3d}  anaemic={te_demo[female_mask]['anaemic_who'].sum():3d}  "
      f"(threshold 12 g/dL)")

print(f"\n  Best classifier by AUC: {best_cname}")
print(f"{'='*65}\n")

# save report
report = "\n".join([
    f"CLASSIFIER REPORT — WHO sex-specific thresholds\n{'='*65}",
    f"Train: {len(X_tr)}  Test: {len(X_te)}  Features: {X_tr.shape[1]}",
    f"Anaemia thresholds: M≥15→13, F≥15→12, 5–14→11.5, <5→11  (g/dL)",
] + report_blocks)
(RESULTS / "clf_report.txt").write_text(report)
print(f"  Report saved → {RESULTS}/clf_report.txt")
print(f"  Models saved → {MODELS_DIR}/clf_*.pkl")
print("\nDone.")
