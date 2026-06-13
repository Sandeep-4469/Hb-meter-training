"""CV per-fold metrics → mean ± std for all 6 classifiers."""
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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score
import lightgbm as lgb
import xgboost as xgb

AIIMS_CSV  = Path("/data2/sandeep/AIIMS_DATA/hb_values.csv")
RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
SEED = 42; N_FOLDS = 5

# ── load (same prep as train_classifiers.py) ──────────────────────────────────
demo = pd.read_csv(AIIMS_CSV)
demo.columns = demo.columns.str.strip().str.lower().str.replace(" ", "_")
demo["gender"] = demo["gender"].str.strip().str.upper()
demo["age"]    = pd.to_numeric(demo["age"], errors="coerce")
demo.loc[demo["age"] > 120, "age"] /= 10
demo["vid_norm"] = (demo["video_name"].str.strip().str.lower()
                    .str.replace(" ", "_", regex=False)
                    .str.replace(r"\.(mkv|mp4|avi)$", "", regex=True))
demo = demo.dropna(subset=["vid_norm","gender","age"])
demo = demo[demo["gender"].isin(["M","F"])].copy()
demo["gender_bin"] = (demo["gender"] == "M").astype(int)

def who_thr(g, a):
    if a < 5:  return 11.0
    if a < 15: return 11.5
    return 13.0 if g == "M" else 12.0

demo["who_thr"] = demo.apply(lambda r: who_thr(r["gender"], r["age"]), axis=1)
demo = demo[["vid_norm","gender_bin","age","who_thr"]].drop_duplicates("vid_norm")

tr_feat = pd.read_csv(RESULTS / "features_combined_train.csv")
feat_cols = [c for c in tr_feat.columns
             if c not in {"video_id","hb_value","split","protocol"}]

def merge_demo(feat_df):
    m = feat_df.merge(demo, left_on="video_id", right_on="vid_norm", how="left")
    m["gender_bin"] = m["gender_bin"].fillna(0)
    m["age"]        = m["age"].fillna(30)
    m["who_thr"]    = m["who_thr"].fillna(12.0)
    return m

tr = merge_demo(tr_feat)
feat_cols_ext = feat_cols + ["gender_bin", "age"]
X_raw = tr[feat_cols_ext].astype(float).fillna(0).values
y_hb  = tr["hb_value"].astype(float).values
y_thr = tr["who_thr"].astype(float).values
y     = (y_hb < y_thr).astype(int)
ids   = tr["video_id"].values

win_lo = np.percentile(X_raw, 1, axis=0)
win_hi = np.percentile(X_raw, 99, axis=0)
X = np.clip(X_raw, win_lo, win_hi)

n_pos = y.sum(); n_neg = (1-y).sum()
def clf_weights(y, cap=5.):
    pr = y.mean()
    w  = np.where(y==1, 1/(pr+1e-8), 1/(1-pr+1e-8))
    w /= w.min(); return np.clip(w, None, cap)
w = clf_weights(y)

# ── per-fold metric ───────────────────────────────────────────────────────────
def fold_metrics(yt, yp_prob):
    yp = (yp_prob >= 0.5).astype(int)
    if len(np.unique(yt)) < 2:
        return dict(auc=np.nan,sens=np.nan,spec=np.nan,
                    f1=np.nan,bal_acc=np.nan,ppv=np.nan,npv=np.nan)
    tn,fp,fn,tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    return dict(
        auc     = roc_auc_score(yt, yp_prob),
        sens    = tp/(tp+fn) if (tp+fn) else 0.,
        spec    = tn/(tn+fp) if (tn+fp) else 0.,
        ppv     = tp/(tp+fp) if (tp+fp) else 0.,
        npv     = tn/(tn+fn) if (tn+fn) else 0.,
        f1      = f1_score(yt, yp, zero_division=0),
        bal_acc = balanced_accuracy_score(yt, yp),
    )

# ── classifiers ───────────────────────────────────────────────────────────────
def classifiers():
    return [
        ("LogReg", Pipeline([
            ("sel", SelectKBest(f_classif, k=80)),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=1., max_iter=1000,
                                       class_weight="balanced", random_state=SEED)),
        ]), False),
        ("SVM", Pipeline([
            ("sel", SelectKBest(f_classif, k=80)),
            ("sc",  StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]), False),
        ("RF",  RandomForestClassifier(n_estimators=300, max_features=.3, min_samples_leaf=4,
                                        class_weight="balanced_subsample",
                                        random_state=SEED, n_jobs=-1), True),
        ("GBM", GradientBoostingClassifier(n_estimators=300, learning_rate=.05, max_depth=4,
                                            subsample=.8, random_state=SEED), True),
        ("LGB", lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=63,
                                    subsample=.8, colsample_bytree=.6, reg_lambda=1.,
                                    is_unbalance=True, random_state=SEED,
                                    verbose=-1, n_jobs=-1), True),
        ("XGB", xgb.XGBClassifier(n_estimators=400, learning_rate=.05, max_depth=5,
                                   subsample=.8, colsample_bytree=.6, reg_lambda=1.,
                                   scale_pos_weight=n_neg/(n_pos+1e-8),
                                   random_state=SEED, verbosity=0,
                                   n_jobs=-1, eval_metric="logloss"), True),
    ]

kf = GroupKFold(n_splits=N_FOLDS)
METRICS = ["auc","sens","spec","f1","bal_acc","ppv","npv"]

print("\n" + "="*70)
print("  5-FOLD CV — Per-fold mean ± std  (WHO sex-specific thresholds)")
print(f"  Train: {len(X)}  Anaemic: {n_pos} ({n_pos/len(y)*100:.0f}%)  Normal: {n_neg}")
print("="*70)

all_results = {}

for cname, model, use_sw in classifiers():
    fold_scores = {m: [] for m in METRICS}

    for fi, (tri, vai) in enumerate(kf.split(X, y, ids)):
        Xf, yf, wf = X[tri], y[tri], w[tri]
        Xv, yv     = X[vai], y[vai]

        model.fit(Xf, yf, sample_weight=wf) if use_sw else model.fit(Xf, yf)
        prob = model.predict_proba(Xv)[:, 1]
        fm   = fold_metrics(yv, prob)
        for m in METRICS:
            fold_scores[m].append(fm[m])

    means = {m: np.nanmean(fold_scores[m]) for m in METRICS}
    stds  = {m: np.nanstd(fold_scores[m])  for m in METRICS}
    all_results[cname] = {"means": means, "stds": stds, "folds": fold_scores}

    print(f"\n  ── {cname}")
    print(f"  {'Metric':<10}  {'Fold1':>7}  {'Fold2':>7}  {'Fold3':>7}  "
          f"{'Fold4':>7}  {'Fold5':>7}  │  {'Mean':>7}  {'±Std':>7}")
    print(f"  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  │  {'─'*7}  {'─'*7}")
    for m in METRICS:
        vals = fold_scores[m]
        row  = f"  {m:<10}  "
        row += "  ".join(f"{v:>7.3f}" for v in vals)
        row += f"  │  {means[m]:>7.3f}  ±{stds[m]:>6.3f}"
        print(row)

# ── summary comparison ────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("  SUMMARY — Mean ± Std across 5 folds")
print("="*70)
print(f"  {'Model':<8}  {'AUC':^13}  {'Sens':^13}  {'Spec':^13}  "
      f"{'F1':^13}  {'BalAcc':^13}")
print(f"  {'─'*7}  {'─'*13}  {'─'*13}  {'─'*13}  {'─'*13}  {'─'*13}")
for cname, _, _ in classifiers():
    r = all_results[cname]
    def ms(m): return f"{r['means'][m]:.3f}±{r['stds'][m]:.3f}"
    print(f"  {cname:<8}  {ms('auc'):^13}  {ms('sens'):^13}  {ms('spec'):^13}  "
          f"{ms('f1'):^13}  {ms('bal_acc'):^13}")

print("\n" + "="*70)
print("  PPV / NPV — Mean ± Std")
print("="*70)
print(f"  {'Model':<8}  {'PPV (Precision)':^17}  {'NPV':^17}")
print(f"  {'─'*7}  {'─'*17}  {'─'*17}")
for cname, _, _ in classifiers():
    r = all_results[cname]
    def ms(m): return f"{r['means'][m]:.3f}±{r['stds'][m]:.3f}"
    print(f"  {cname:<8}  {ms('ppv'):^17}  {ms('npv'):^17}")

# save
lines = []
for cname, r in all_results.items():
    lines.append(f"\n{cname}:")
    for m in METRICS:
        lines.append(f"  {m}: {r['means'][m]:.3f} ± {r['stds'][m]:.3f}  "
                     f"folds={[f'{v:.3f}' for v in r['folds'][m]]}")
(RESULTS / "clf_cv_variance.txt").write_text("\n".join(lines))
print(f"\n  Saved → {RESULTS}/clf_cv_variance.txt\n")
