"""
Final thesis results — comprehensive comparison of all methods.

Loads best predictions from each approach and produces:
  - Full comparison table (all methods × all metrics)
  - Per-HB-bin error breakdown
  - Publication-ready plots
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                               RandomForestRegressor, RandomForestClassifier)
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                              mean_absolute_error, mean_squared_error, r2_score,
                              confusion_matrix, classification_report, roc_curve)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import RANDOM_SEED, ACCURACY_BANDS

OUT_DIR  = Path(__file__).parent / "results"
FEAT_DIR = ROOT / "metric_learning" / "results"
ANEMIA   = 12.0
META     = {"video_id","hb_value","split","protocol"}


# ── data ──────────────────────────────────────────────────────────────────

train_df = pd.read_csv(FEAT_DIR / "features_train.csv")
test_df  = pd.read_csv(FEAT_DIR / "features_test.csv")

feat_cols = [c for c in train_df.columns if c not in META and c != "anemia"]
X_tr_raw  = train_df[feat_cols].values.astype(np.float32)
X_te_raw  = test_df[feat_cols].values.astype(np.float32)
y_tr      = train_df["hb_value"].values.astype(np.float32)
y_te      = test_df["hb_value"].values.astype(np.float32)
groups    = train_df["video_id"].values

imp    = SimpleImputer(strategy="median").fit(X_tr_raw)
X_tr   = imp.transform(X_tr_raw)
X_te   = imp.transform(X_te_raw)

# sample weights: inverse frequency per HB bin
bins   = np.floor(y_tr).astype(int)
cnt    = {b: (bins==b).sum() for b in np.unique(bins)}
sw     = np.array([1.0/cnt[b] for b in bins], dtype=np.float32)
sw     = np.clip(sw / sw.mean(), 0, 5.0)

# extreme-weighted: extremes count 4× more
y_mean, y_std = y_tr.mean(), y_tr.std()
sw_ext = sw * (1 + 3.0 * np.abs(y_tr - y_mean) / y_std)
sw_ext = np.clip(sw_ext / sw_ext.mean(), 0, 8.0)

anemia_tr = (y_tr < ANEMIA).astype(int)
anemia_te = (y_te < ANEMIA).astype(int)

# ── best ML models (retrained with extreme weighting) ────────────────────

print("=" * 68)
print("  Training all ML models...")
print("=" * 68)

# ── Regression ────────────────────────────────────────────────────────────

models_reg = {
    "RF (standard)":   RandomForestRegressor(n_estimators=400, max_depth=7,
                           min_samples_leaf=2, random_state=RANDOM_SEED),
    "GB (standard)":   GradientBoostingRegressor(n_estimators=300, max_depth=4,
                           learning_rate=0.05, subsample=0.8, random_state=RANDOM_SEED),
    "GB (ext-weighted)": GradientBoostingRegressor(n_estimators=300, max_depth=4,
                           learning_rate=0.05, subsample=0.8, random_state=RANDOM_SEED),
    "LightGBM":        lgb.LGBMRegressor(n_estimators=300, objective="mae",
                           learning_rate=0.05, num_leaves=31, reg_lambda=1.0,
                           random_state=RANDOM_SEED, verbose=-1),
}

reg_preds = {}
for name, mdl in models_reg.items():
    w = sw_ext if "ext" in name else sw
    mdl.fit(X_tr, y_tr, sample_weight=w)
    reg_preds[name] = mdl.predict(X_te)
    print(f"  {name:<25} MAE={mean_absolute_error(y_te, reg_preds[name]):.4f}")

# NN predictions from saved file
nn_csv = OUT_DIR / "best_predictions.csv"
if nn_csv.exists():
    nn_df = pd.read_csv(nn_csv)
    reg_preds["BestHbNet (NN)"]    = nn_df["nn_pred"].values
    reg_preds["NN+GB Blend (50%)"] = nn_df["blend_pred"].values
    print(f"  {'BestHbNet (NN)':<25} MAE={mean_absolute_error(y_te, reg_preds['BestHbNet (NN)']):.4f}")
    print(f"  {'NN+GB Blend (50%)':<25} MAE={mean_absolute_error(y_te, reg_preds['NN+GB Blend (50%)']):.4f}")

# SmallHbNet from earlier
small_csv = OUT_DIR / "small_predictions_regression.csv"
if small_csv.exists():
    sm_df = pd.read_csv(small_csv)
    reg_preds["SmallHbNet (NN)"] = sm_df["hb_pred"].values
    print(f"  {'SmallHbNet (NN)':<25} MAE={mean_absolute_error(y_te, reg_preds['SmallHbNet (NN)']):.4f}")

# ── Classification ────────────────────────────────────────────────────────

print()
models_cls = {
    "GB (balanced)":   GradientBoostingClassifier(n_estimators=300, max_depth=3,
                           learning_rate=0.05, subsample=0.8, random_state=RANDOM_SEED),
    "RF (balanced)":   RandomForestClassifier(n_estimators=400, max_depth=6,
                           class_weight="balanced", random_state=RANDOM_SEED),
    "LightGBM (bal)":  lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           is_unbalance=True, num_leaves=31, random_state=RANDOM_SEED,
                           verbose=-1),
}

cls_preds = {}
for name, mdl in models_cls.items():
    w = sw if "GB" in name else None
    pos = anemia_tr.sum(); neg = len(anemia_tr) - pos
    if "GB" in name:
        mdl.fit(X_tr, anemia_tr, sample_weight=sw)
    else:
        mdl.fit(X_tr, anemia_tr)
    prob = mdl.predict_proba(X_te)[:, 1]
    cls_preds[name] = prob
    auc = roc_auc_score(anemia_te, prob)
    print(f"  {name:<25} AUC={auc:.4f}")

# regression-derived anemia: lower HB pred = higher anemia prob
for rname in ["GB (standard)", "GB (ext-weighted)", "LightGBM"]:
    p = reg_preds[rname]
    # convert to probability-like: sigmoid(-(pred - 12))
    prob = 1 / (1 + np.exp((p - ANEMIA)))
    cls_preds[f"{rname} (reg→cls)"] = prob
    auc = roc_auc_score(anemia_te, prob)
    print(f"  {f'{rname} (reg→cls)':<25} AUC={auc:.4f}")


# ── full metrics ───────────────────────────────────────────────────────────

def reg_metrics(y_true, pred):
    mae  = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2   = r2_score(y_true, pred)
    w1   = (np.abs(y_true-pred)<=1.0).mean()
    w15  = (np.abs(y_true-pred)<=1.5).mean()
    w2   = (np.abs(y_true-pred)<=2.0).mean()
    try:   auc = roc_auc_score((y_true<ANEMIA).astype(int), -pred)
    except: auc = float("nan")
    return dict(mae=mae, rmse=rmse, r2=r2, w1=w1, w15=w15, w2=w2, auc=auc)

def cls_metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    auc  = roc_auc_score(y_true, prob)
    f1m  = f1_score(y_true, pred, average="macro",  zero_division=0)
    f1a  = f1_score(y_true, pred, average="binary", zero_division=0)
    cm   = confusion_matrix(y_true, pred, labels=[1,0])
    sens = cm[0,0] / float(y_true.sum())
    spec = cm[1,1] / float((y_true==0).sum())
    bal  = balanced_accuracy_score(y_true, pred)
    acc  = float((pred==y_true).mean())
    return dict(auc=auc, f1m=f1m, f1a=f1a, sens=sens, spec=spec, bal=bal, acc=acc,
                cm=cm, pred=pred)


# ── PRINT TABLES ───────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("  REGRESSION RESULTS — all methods")
print("=" * 90)
print(f"  {'Model':<27}  {'MAE':>6}  {'RMSE':>6}  {'R²':>7}  {'W±1':>6}  {'W±1.5':>7}  {'W±2':>6}  {'AUC':>6}")
print(f"  {'─'*82}")

all_reg = {}
for name, pred in reg_preds.items():
    m = reg_metrics(y_te, pred)
    all_reg[name] = m
    print(f"  {name:<27}  {m['mae']:>6.3f}  {m['rmse']:>6.3f}  {m['r2']:>7.3f}  "
          f"{100*m['w1']:>5.1f}%  {100*m['w15']:>6.1f}%  {100*m['w2']:>5.1f}%  {m['auc']:>6.3f}")

best_reg_name = min(all_reg, key=lambda k: all_reg[k]["mae"])
best_reg_pred = reg_preds[best_reg_name]
print(f"\n  ★ Best regression: {best_reg_name}  MAE={all_reg[best_reg_name]['mae']:.3f}")

print("\n" + "=" * 90)
print("  CLASSIFICATION RESULTS — all methods")
print("=" * 90)
print(f"  {'Model':<27}  {'AUC':>6}  {'Sens':>6}  {'Spec':>6}  {'F1-an':>7}  {'BalAcc':>7}  {'Acc':>6}")
print(f"  {'─'*72}")

all_cls = {}
for name, prob in cls_preds.items():
    m = cls_metrics(anemia_te, prob)
    all_cls[name] = m
    print(f"  {name:<27}  {m['auc']:>6.3f}  {m['sens']:>6.3f}  {m['spec']:>6.3f}  "
          f"{m['f1a']:>7.3f}  {m['bal']:>7.3f}  {100*m['acc']:>5.1f}%")

best_cls_name = max(all_cls, key=lambda k: all_cls[k]["auc"])
best_cls_prob = cls_preds[best_cls_name]
best_cls_m    = all_cls[best_cls_name]
print(f"\n  ★ Best classification: {best_cls_name}  AUC={best_cls_m['auc']:.3f}  "
      f"Sens={best_cls_m['sens']:.3f}  Spec={best_cls_m['spec']:.3f}")

# ── per-bin breakdown for best regression ────────────────────────────────

print(f"\n  Per-bin breakdown — {best_reg_name}:")
print(f"  {'HB bin':>7}  {'n':>4}  {'True mn':>8}  {'Pred mn':>8}  {'Pred SD':>8}  {'Bias':>7}  {'MAE':>6}  {'W±1':>6}  {'W±2':>6}")
print(f"  {'─'*82}")
for lo in range(3, 18):
    hi = lo + 1
    m  = (y_te >= lo) & (y_te < hi)
    if m.sum() == 0: continue
    t  = y_te[m]; p = best_reg_pred[m]; e = p - t
    w1 = (np.abs(e) <= 1.0).mean(); w2 = (np.abs(e) <= 2.0).mean()
    flag = " ◄" if np.abs(e).mean() > 2.5 else ""
    print(f"  {lo:>4}–{hi:<3}  {m.sum():>4}  {t.mean():>8.2f}  {p.mean():>8.2f}  "
          f"{p.std():>8.2f}  {e.mean():>+7.2f}  {np.abs(e).mean():>6.3f}  "
          f"{100*w1:>5.0f}%  {100*w2:>5.0f}%{flag}")

# ── best classification confusion ─────────────────────────────────────────
cm  = best_cls_m["cm"]
print(f"\n  Best classification confusion matrix ({best_cls_name}):")
print(f"                    Pred-Anemic  Pred-Normal")
print(f"    True-Anemic         {cm[0,0]:>4}         {cm[0,1]:>4}   (n={int(anemia_te.sum())})")
print(f"    True-Normal         {cm[1,0]:>4}         {cm[1,1]:>4}   (n={int((anemia_te==0).sum())})")
print()
print(classification_report(anemia_te, best_cls_m["pred"],
                            target_names=["Normal (≥12)","Anemic (<12)"],
                            zero_division=0))

# ── range compression summary ─────────────────────────────────────────────
print(f"  Range compression check ({best_reg_name}):")
print(f"    True  std={y_te.std():.3f}  range=[{y_te.min():.1f},{y_te.max():.1f}]")
print(f"    Pred  std={best_reg_pred.std():.3f}  range=[{best_reg_pred.min():.1f},{best_reg_pred.max():.1f}]")
print(f"    Ratio={best_reg_pred.std()/y_te.std():.3f}  (1.0=no compression)")


# ── PLOTS ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)

# ── 1. Regression comparison bar ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
names = list(all_reg.keys())
maes  = [all_reg[k]["mae"] for k in names]
short = [n.replace(" (standard)","").replace(" (ext-weighted)","★")
          .replace(" (NN)","(NN)").replace(" (50%)","") for n in names]
cols  = ["#2ecc71" if m < 2.1 else "#f39c12" if m < 2.3 else "#e74c3c" for m in maes]
ax.barh(range(len(names)), maes, color=cols, alpha=0.85, edgecolor="white")
ax.axvline(2.0, color="black", lw=1, linestyle="--", alpha=0.5, label="2.0 g/dL")
ax.set_yticks(range(len(names))); ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel("MAE (g/dL)"); ax.set_title("Regression MAE\nAll Methods")
for i, v in enumerate(maes):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
ax.legend(fontsize=7)

# ── 2. Classification AUC comparison ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
cnames = list(all_cls.keys())
aucs   = [all_cls[k]["auc"] for k in cnames]
senss  = [all_cls[k]["sens"] for k in cnames]
cshort = [n.replace(" (balanced)","").replace(" (reg→cls)","→")
           .replace(" (bal)","") for n in cnames]
x = np.arange(len(cnames))
ax.bar(x - 0.2, aucs, 0.35, label="AUC",         color="steelblue", alpha=0.8)
ax.bar(x + 0.2, senss, 0.35, label="Sensitivity", color="coral",     alpha=0.8)
ax.axhline(0.5, color="black", lw=0.8, linestyle=":", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(cshort, rotation=45, fontsize=7, ha="right")
ax.set_title("Classification AUC & Sensitivity")
ax.legend(fontsize=8); ax.set_ylim(0, 1.0)

# ── 3. True vs Pred — best regression ────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
err_c = np.abs(best_reg_pred - y_te)
sc = ax.scatter(y_te, best_reg_pred, c=err_c, cmap="RdYlGn_r",
                vmin=0, vmax=4, s=50, edgecolors="white", lw=0.4, zorder=3)
lim = (min(y_te.min(), best_reg_pred.min())-0.5,
       max(y_te.max(), best_reg_pred.max())+0.5)
ax.plot(lim, lim, "k--", lw=1.2, label="Identity")
ax.plot(lim, [l+2 for l in lim], "orange", lw=0.8, linestyle=":")
ax.plot(lim, [l-2 for l in lim], "orange", lw=0.8, linestyle=":", label="±2 g/dL")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("True HB (g/dL)"); ax.set_ylabel("Predicted HB (g/dL)")
ax.set_title(f"True vs Predicted\n{best_reg_name}\nMAE={all_reg[best_reg_name]['mae']:.3f}")
plt.colorbar(sc, ax=ax, label="|Error|", fraction=0.046, pad=0.04)
ax.legend(fontsize=7)

# ── 4. ROC curves — all classification models ─────────────────────────────
ax = fig.add_subplot(gs[0, 3])
cmap_roc = plt.cm.tab10(np.linspace(0, 0.9, len(cls_preds)))
for i, (cname, prob) in enumerate(cls_preds.items()):
    auc_v = all_cls[cname]["auc"]
    fpr, tpr, _ = roc_curve(anemia_te, prob)
    lw   = 2.5 if cname == best_cls_name else 1.0
    lab  = f"{cshort[i] if i<len(cshort) else cname} ({auc_v:.3f})"
    ax.plot(fpr, tpr, lw=lw, color=cmap_roc[i], label=lab)
ax.plot([0,1],[0,1],"gray",lw=0.8,linestyle=":")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC — Anemia (HB<12)")
ax.legend(fontsize=6, loc="lower right")

# ── 5. MAE per HB bin — best regression ──────────────────────────────────
ax = fig.add_subplot(gs[1, 0:2])
bin_data = []
for lo in range(3, 18):
    hi = lo + 1
    m  = (y_te >= lo) & (y_te < hi)
    if m.sum() == 0: continue
    p  = best_reg_pred[m]; t = y_te[m]
    mae_v = np.abs(p - t).mean()
    w1_v  = (np.abs(p-t)<=1.0).mean()
    bias_v= (p - t).mean()
    bin_data.append({"lo":lo,"hi":hi,"n":m.sum(),"mae":mae_v,"w1":w1_v,"bias":bias_v})
bd  = pd.DataFrame(bin_data)
cols_bar = ["#e74c3c" if v>2.5 else "#f39c12" if v>1.5 else "#2ecc71" for v in bd["mae"]]
x   = np.arange(len(bd))
ax.bar(x, bd["mae"], color=cols_bar, alpha=0.85, edgecolor="white", label="MAE")
ax.plot(x, [abs(b) for b in bd["bias"]], "k--o", ms=4, lw=1.2, label="|Bias|")
ax.axhline(1.0, color="green",  lw=1, linestyle="--", alpha=0.6, label="1 g/dL")
ax.axhline(2.0, color="orange", lw=1, linestyle="--", alpha=0.6, label="2 g/dL")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(r['lo'])}–{int(r['hi'])}\nn={int(r['n'])}\n{100*r['w1']:.0f}%±1"
                    for _, r in bd.iterrows()], fontsize=7)
ax.set_ylabel("Error (g/dL)")
ax.set_title(f"MAE per HB Bin — {best_reg_name}\n(green<1.5, orange<2.5, red≥2.5  |  label shows n, % within ±1)")
ax.legend(fontsize=8)

# ── 6. Compression plot (pred mean±SD per bin) ────────────────────────────
ax = fig.add_subplot(gs[1, 2])
bin_tm, bin_pm, bin_ps = [], [], []
for lo in range(3, 18):
    m = (y_te>=lo)&(y_te<lo+1)
    if m.sum() < 2: continue
    bin_tm.append(y_te[m].mean())
    bin_pm.append(best_reg_pred[m].mean())
    bin_ps.append(best_reg_pred[m].std())
ax.plot(bin_tm, bin_tm, "k--", lw=1.2, label="Ideal (no compression)")
ax.errorbar(bin_tm, bin_pm, yerr=bin_ps, fmt="o-", color="steelblue",
            capsize=4, lw=1.5, ms=5, label="Pred mean ± SD")
ax.fill_between(bin_tm,
                [m-s for m,s in zip(bin_pm,bin_ps)],
                [m+s for m,s in zip(bin_pm,bin_ps)],
                alpha=0.15, color="steelblue")
ax.set_xlabel("True HB bin mean (g/dL)"); ax.set_ylabel("Predicted HB (g/dL)")
ax.set_title("Range Compression Check\n(flat line = regressed to mean)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── 7. Severity band MAE ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
bands = [("<7\nSevere\n(n=7)",0,7), ("7–10\nModerate\n(n=41)",7,10),
         ("10–12\nMild\n(n=22)",10,12), (">12\nNormal\n(n=25)",12,25)]
bmaes, bbias, bw1, bw2 = [], [], [], []
for _, lo, hi in bands:
    m = (y_te>=lo)&(y_te<hi)
    p = best_reg_pred[m]; t = y_te[m]
    bmaes.append(np.abs(p-t).mean())
    bbias.append((p-t).mean())
    bw1.append((np.abs(p-t)<=1.0).mean())
    bw2.append((np.abs(p-t)<=2.0).mean())
x = np.arange(len(bands))
bcs = ["#e74c3c","#e67e22","#2ecc71","#3498db"]
bars = ax.bar(x, bmaes, color=bcs, alpha=0.85, edgecolor="white")
ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
ax.axhline(2.0, color="black", lw=1, linestyle=":", alpha=0.5)
for i, (bar, mae_v, w1_v, w2_v, bias_v) in enumerate(zip(bars,bmaes,bw1,bw2,bbias)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f"MAE={mae_v:.2f}\nBias{bias_v:+.2f}\nW±1:{100*w1_v:.0f}%\nW±2:{100*w2_v:.0f}%",
            ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bands], fontsize=8)
ax.set_ylabel("MAE (g/dL)"); ax.set_title("Error by Severity Band")

# ── 8. Confusion matrix best classifier ──────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
cm_best = best_cls_m["cm"]
im = ax.imshow(cm_best, cmap="Blues", aspect="auto")
ax.set_xticks([0,1]); ax.set_xticklabels(["Pred Anemic","Pred Normal"])
ax.set_yticks([0,1]); ax.set_yticklabels(["True Anemic","True Normal"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_best[i,j]), ha="center", va="center",
                fontsize=16, color="white" if cm_best[i,j]>cm_best.max()/2 else "black",
                fontweight="bold")
ax.set_title(f"Best Classifier\n{best_cls_name}\nSens={best_cls_m['sens']:.3f}  "
             f"Spec={best_cls_m['spec']:.3f}")

# ── 9. Within-band accuracy curves ───────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
thresholds = np.linspace(0, 5, 200)
for rname, pred in list(reg_preds.items())[:5]:
    cum = [(np.abs(y_te - pred) <= t).mean() for t in thresholds]
    ax.plot(thresholds, [100*c for c in cum], lw=1.5, label=rname.replace(" (standard)",""))
ax.axvline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5)
ax.axvline(2.0, color="black", lw=0.8, linestyle=":",  alpha=0.5)
ax.set_xlabel("Error threshold (g/dL)"); ax.set_ylabel("% within threshold")
ax.set_title("Cumulative Accuracy Curve\n(higher & left = better)")
ax.legend(fontsize=7); ax.set_xlim(0,5); ax.grid(True, alpha=0.3)

# ── 10. Summary table (text) ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2:])
ax.axis("off")
rows = [["Method", "MAE", "W±1", "W±2", "AUC-cls", "Sens", "Spec", "F1-an"]]
for rname, rmet in all_reg.items():
    # find best matching classifier
    cname = best_cls_name
    cmet  = best_cls_m
    rows.append([rname[:22],
                 f"{rmet['mae']:.3f}", f"{100*rmet['w1']:.0f}%", f"{100*rmet['w2']:.0f}%",
                 f"{cmet['auc']:.3f}", f"{cmet['sens']:.3f}", f"{cmet['spec']:.3f}",
                 f"{cmet['f1a']:.3f}"])
t = ax.table(cellText=rows[1:], colLabels=rows[0],
             loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8.5)
t.scale(1, 1.4)
for j in range(len(rows[0])):
    t[0, j].set_facecolor("#2c3e50"); t[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(rows)):
    for j in range(len(rows[0])):
        t[i, j].set_facecolor("#ecf0f1" if i%2==0 else "white")
ax.set_title("Full Results Summary", fontsize=10, pad=12)

fig.suptitle(
    f"NISHAD — Complete Results Summary\n"
    f"Best Regression: {best_reg_name}  MAE={all_reg[best_reg_name]['mae']:.3f} g/dL  "
    f"W±2={100*all_reg[best_reg_name]['w2']:.0f}%  |  "
    f"Best Classifier: {best_cls_name}  AUC={best_cls_m['auc']:.3f}  "
    f"Sens={best_cls_m['sens']:.3f}",
    fontsize=11, y=1.005)

plt.savefig(OUT_DIR / "thesis_final_results.png", dpi=145, bbox_inches="tight")
plt.close()
print(f"\n  Saved → {OUT_DIR / 'thesis_final_results.png'}")
print("\nDone.")
