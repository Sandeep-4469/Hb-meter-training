"""Extended metrics from saved models — no re-training."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              roc_auc_score, confusion_matrix,
                              f1_score, precision_score, balanced_accuracy_score)

RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
ANEMIA_THR = 12.0
BANDS      = [(0,7,"0–7   Severe"),(7,10,"7–10  Moderate"),
              (10,14,"10–14 Mild/Normal"),(14,99,"14+   High")]

# ── load data ─────────────────────────────────────────────────────────────────
tr = pd.read_csv(RESULTS/"features_combined_train.csv")
te = pd.read_csv(RESULTS/"features_combined_test.csv")
mc = {"video_id","hb_value","split","protocol"}
fc = [c for c in tr.columns if c not in mc]

X_tr_raw = tr[fc].astype(float).fillna(0).values
y_tr     = tr["hb_value"].astype(float).values
X_te_raw = te[fc].astype(float).fillna(0).values
y_te     = te["hb_value"].astype(float).values

# ── get predictions from each saved model ─────────────────────────────────────
# winsorise using training percentiles (same as training)
lo = np.percentile(X_tr_raw,1,axis=0); hi = np.percentile(X_tr_raw,99,axis=0)
X_tr = np.clip(X_tr_raw,lo,hi);        X_te = np.clip(X_te_raw,lo,hi)

# Model 1 — Ensemble
m1 = joblib.load(MODELS_DIR/"ensemble_best_mae.pkl")
Xte1 = m1["selector"].transform(X_te)
p1   = (m1["rf"].predict(Xte1) + m1["lgb"].predict(Xte1) + m1["xgb"].predict(Xte1)) / 3

# Model 2 — PLS
m2  = joblib.load(MODELS_DIR/"pls_best_auc.pkl")
p2  = m2["pls"].predict(X_te).ravel()

# Model 3 — Meta-stack
m3   = joblib.load(MODELS_DIR/"meta_stack_best_sens.pkl")
base = np.column_stack([
    m3["rf"].predict(X_te),
    m3["lgb"].predict(X_te),
    m3["xgb"].predict(X_te),
])
p3 = m3["meta_ridge"].predict(m3["meta_scaler"].transform(base))

# ── metric helpers ────────────────────────────────────────────────────────────
def clf_metrics(y_true, y_pred, thr=ANEMIA_THR):
    yt  = (y_true < thr).astype(int)   # 1 = anaemic
    yp  = (y_pred < thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    sens    = tp / (tp + fn) if (tp+fn) > 0 else 0.
    spec    = tn / (tn + fp) if (tn+fp) > 0 else 0.
    ppv     = tp / (tp + fp) if (tp+fp) > 0 else 0.   # precision
    npv     = tn / (tn + fn) if (tn+fn) > 0 else 0.   # negative predictive value
    f1      = f1_score(yt, yp, zero_division=0)
    bal_acc = balanced_accuracy_score(yt, yp)
    auc     = roc_auc_score(yt, -y_pred)
    return dict(tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),
                sens=sens,spec=spec,ppv=ppv,npv=npv,
                f1=f1,bal_acc=bal_acc,auc=auc)

def reg_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**.5
    w1   = np.mean(np.abs(y_true-y_pred)<=1)*100
    w2   = np.mean(np.abs(y_true-y_pred)<=2)*100
    bias = (y_pred-y_true).mean()
    ratio= y_pred.std()/(y_true.std()+1e-8)
    return dict(mae=mae,rmse=rmse,w1=w1,w2=w2,bias=bias,ratio=ratio)

def band_table(y_true, y_pred):
    rows=[]
    for lo,hi,label in BANDS:
        m=(y_true>=lo)&(y_true<hi); n=int(m.sum())
        if n==0: rows.append((label,0,{})); continue
        rows.append((label, n, reg_metrics(y_true[m],y_pred[m])))
    return rows

# ── print ─────────────────────────────────────────────────────────────────────
models = [
    ("MODEL 1 — Ensemble RF+LGB+XGB  (SelectKBest k=80)", y_te, p1),
    ("MODEL 2 — PLS Direct (n=5 components)",             y_te, p2),
    ("MODEL 3 — Meta-Stacking Ridge on RF+LGB+XGB OOF",  y_te, p3),
]

W = 84
print("\n"+"="*W)
print("  FULL METRICS — Test set (n=95)  |  Anaemia threshold: HB < 12 g/dL")
print("="*W)
print(f"  Distribution:  0–7: n=7(7%)  7–10: n=41(43%)  10–14: n=38(40%)  14+: n=9(9%)")
print(f"  Anaemic (HB<12): n={int((y_te<12).sum())}   Non-anaemic (HB≥12): n={int((y_te>=12).sum())}\n")

all_lines = []
for name, yt, yp in models:
    r = reg_metrics(yt, yp)
    c = clf_metrics(yt, yp)

    lines = [
        "─"*W,
        f"  {name}",
        "─"*W,
        "",
        "  ── Regression metrics ──────────────────────────────────────────",
        f"  MAE     = {r['mae']:.3f} g/dL",
        f"  RMSE    = {r['rmse']:.3f} g/dL",
        f"  W±1     = {r['w1']:.1f}%   (predictions within ±1 g/dL of truth)",
        f"  W±2     = {r['w2']:.1f}%   (predictions within ±2 g/dL of truth)",
        f"  Bias    = {r['bias']:+.3f} g/dL  (mean over-/under-estimation)",
        f"  PredStd/TrueStd = {r['ratio']:.3f}  (range compression: <1 = compressed)",
        "",
        "  ── Classification metrics  (threshold HB < 12 g/dL = anaemic) ─",
        f"  AUC          = {c['auc']:.3f}",
        f"  Sensitivity  = {c['sens']:.3f}   (recall, TPR)  →  {c['tp']}/{c['tp']+c['fn']} anaemic patients caught",
        f"  Specificity  = {c['spec']:.3f}   (TNR)          →  {c['tn']}/{c['tn']+c['fp']} normal patients correctly cleared",
        f"  Precision    = {c['ppv']:.3f}   (PPV)          →  of positives flagged, {c['ppv']*100:.0f}% truly anaemic",
        f"  NPV          = {c['npv']:.3f}   (neg pred val) →  of negatives cleared, {c['npv']*100:.0f}% truly normal",
        f"  F1 Score     = {c['f1']:.3f}",
        f"  Balanced Acc = {c['bal_acc']:.3f}",
        f"  Confusion matrix:  TP={c['tp']}  FP={c['fp']}  TN={c['tn']}  FN={c['fn']}",
        "",
        "  ── Per-range regression breakdown ──────────────────────────────",
        f"  {'Range':<22}  {'n':>3}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>6}  {'W±1':>6}  {'W±2':>6}",
        f"  {'─'*21}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}",
    ]
    for label, n, rm in band_table(yt, yp):
        if n == 0:
            lines.append(f"  {label:<22}  {n:>3}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}")
        else:
            lines.append(f"  {label:<22}  {n:>3}  {rm['mae']:>6.3f}  {rm['rmse']:>6.3f}"
                         f"  {rm['bias']:>+6.2f}  {rm['w1']:>5.0f}%  {rm['w2']:>5.0f}%")
    lines.append("")
    block = "\n".join(lines)
    print(block)
    all_lines.append(block)

# summary comparison table
print("="*W)
print("  SUMMARY COMPARISON")
print("="*W)
print(f"  {'Model':<42}  {'MAE':>6}  {'AUC':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}  {'W±2':>6}")
print(f"  {'─'*41}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
for name, yt, yp in models:
    r = reg_metrics(yt,yp); c = clf_metrics(yt,yp)
    short = name.split("—")[1].strip()[:40]
    print(f"  {short:<42}  {r['mae']:>6.3f}  {c['auc']:>6.3f}  {c['sens']:>6.3f}"
          f"  {c['spec']:>6.3f}  {c['f1']:>6.3f}  {r['w2']:>5.1f}%")
print("="*W+"\n")

# save
out = RESULTS/"eval_full_metrics.txt"
header = [f"\nFULL METRICS — Test set (n=95)\n{'='*W}"]
out.write_text("\n".join(header + all_lines))
print(f"  Saved → {out}")
