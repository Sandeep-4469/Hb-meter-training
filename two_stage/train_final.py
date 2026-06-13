"""
Train final models with 5-fold CV evaluation + save to disk.
=============================================================
Three models:
  1. Ensemble  RF+LGB+XGB, SelectKBest(k=80)   ← best MAE
  2. PLS direct n=5                              ← best AUC
  3. Meta-stacking Ridge on RF+LGB+XGB OOF      ← best Sensitivity

Procedure for each:
  - 5-fold GroupKFold → OOF predictions (honest CV metrics)
  - Retrain on full training set → test set evaluation
  - Save final model to saved_models/

Outputs
-------
  saved_models/ensemble_best_mae.pkl
  saved_models/pls_best_auc.pkl
  saved_models/meta_stack_best_sens.pkl
  results/final_range_report.txt
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import lightgbm as lgb
import xgboost as xgb

RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

SEED    = 42
N_FOLDS = 5
BANDS   = [(0,7,"0–7   Severe"), (7,10,"7–10  Moderate"),
           (10,14,"10–14 Mild/Normal"), (14,99,"14+   High")]
ANEMIA_THR = 12.0

# ── load & preprocess ─────────────────────────────────────────────────────────
tr = pd.read_csv(RESULTS / "features_combined_train.csv")
te = pd.read_csv(RESULTS / "features_combined_test.csv")
mc = {"video_id","hb_value","split","protocol"}
fc = [c for c in tr.columns if c not in mc]

X_tr_raw = tr[fc].astype(float).fillna(0).values
y_tr     = tr["hb_value"].astype(float).values
X_te_raw = te[fc].astype(float).fillna(0).values
y_te     = te["hb_value"].astype(float).values
ids_tr   = tr["video_id"].values

# winsorise using training percentiles
winsor_lo = np.percentile(X_tr_raw, 1, axis=0)
winsor_hi = np.percentile(X_tr_raw, 99, axis=0)
X_tr = np.clip(X_tr_raw, winsor_lo, winsor_hi)
X_te = np.clip(X_te_raw, winsor_lo, winsor_hi)

print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Features: {X_tr.shape[1]}")
print(f"Winsorised → max abs: {np.abs(X_tr).max():.1f}\n")

# ── helpers ───────────────────────────────────────────────────────────────────
def make_weights(y, cap=5.0):
    bins = [0, 7, 10, 12, 100]
    lbl  = np.digitize(y, bins) - 1
    cnt  = np.bincount(lbl, minlength=4)
    freq = cnt[lbl] / len(y)
    w    = 1.0 / (freq + 1e-8)
    w   /= w.min()
    return np.clip(w, None, cap)

def oversample(X, y, thr=7.0, frac=0.15, seed=0):
    rng  = np.random.default_rng(seed)
    idx  = np.where(y < thr)[0]
    want = int(len(y) * frac)
    if len(idx) >= want or len(idx) == 0:
        return X, y
    extra  = want - len(idx)
    chosen = rng.choice(idx, extra, replace=True)
    noise  = rng.normal(0, X[idx].std(0) * 0.03 + 1e-8, (extra, X.shape[1]))
    return (np.vstack([X, X[chosen] + noise]),
            np.concatenate([y, np.clip(y[chosen] + rng.normal(0, .15, extra), 3, 7)]))

def within(yt, yp, k): return np.mean(np.abs(yt - yp) <= k) * 100

def metrics(yt, yp, tag=""):
    mae  = mean_absolute_error(yt, yp)
    rmse = mean_squared_error(yt, yp) ** 0.5
    w2   = within(yt, yp, 2.0)
    auc  = roc_auc_score((yt < ANEMIA_THR).astype(int), -yp)
    sens = np.mean(yp[yt < ANEMIA_THR] < ANEMIA_THR)
    return {"tag": tag, "mae": mae, "rmse": rmse, "w2": w2, "auc": auc, "sens": sens}

def band_rows(yt, yp):
    rows = []
    for lo, hi, label in BANDS:
        m = (yt >= lo) & (yt < hi)
        n = int(m.sum())
        if n == 0:
            rows.append((label, n, None, None, None, None, None))
            continue
        mae  = mean_absolute_error(yt[m], yp[m])
        rmse = mean_squared_error(yt[m], yp[m]) ** 0.5
        bias = float((yp[m] - yt[m]).mean())
        w1   = within(yt[m], yp[m], 1.0)
        w2   = within(yt[m], yp[m], 2.0)
        rows.append((label, n, mae, rmse, bias, w1, w2))
    return rows

def print_block(name, oof_m, test_m, test_bands):
    lines = []
    lines.append(f"\n{'─'*82}")
    lines.append(f"  {name}")
    lines.append(f"  OOF  → MAE={oof_m['mae']:.3f}  RMSE={oof_m['rmse']:.3f}  "
                 f"W±2={oof_m['w2']:.1f}%  AUC={oof_m['auc']:.3f}  Sens={oof_m['sens']:.3f}")
    lines.append(f"  Test → MAE={test_m['mae']:.3f}  RMSE={test_m['rmse']:.3f}  "
                 f"W±2={test_m['w2']:.1f}%  AUC={test_m['auc']:.3f}  Sens={test_m['sens']:.3f}")
    lines.append(f"{'─'*82}")
    lines.append(f"  {'Range':<22}  {'n':>3}  {'MAE':>6}  {'RMSE':>6}  "
                 f"{'Bias':>6}  {'W±1':>6}  {'W±2':>6}")
    lines.append(f"  {'─'*21}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for label, n, mae, rmse, bias, w1, w2 in test_bands:
        if mae is None:
            lines.append(f"  {label:<22}  {n:>3}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}")
        else:
            lines.append(f"  {label:<22}  {n:>3}  {mae:>6.3f}  {rmse:>6.3f}  "
                         f"{bias:>+6.2f}  {w1:>5.0f}%  {w2:>5.0f}%")
    return "\n".join(lines)

kf = GroupKFold(n_splits=N_FOLDS)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Ensemble RF+LGB+XGB with SelectKBest(k=80)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MODEL 1: Ensemble RF+LGB+XGB  (SelectKBest k=80)")
print("=" * 60)

oof1 = np.zeros(len(X_tr))
for fi, (tri, vai) in enumerate(kf.split(X_tr, y_tr, ids_tr)):
    Xf, yf = oversample(X_tr[tri], y_tr[tri], seed=SEED + fi)
    wf     = make_weights(yf)
    sel_f  = SelectKBest(f_regression, k=80)
    Xfs    = sel_f.fit_transform(Xf, yf)
    Xval   = sel_f.transform(X_tr[vai])
    rf_f   = RandomForestRegressor(300, max_features=.3, min_samples_leaf=4,
                                    random_state=SEED, n_jobs=-1)
    lg_f   = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=63,
                                 subsample=.8, colsample_bytree=.6, reg_lambda=1,
                                 random_state=SEED, verbose=-1, n_jobs=-1)
    xg_f   = xgb.XGBRegressor(n_estimators=400, learning_rate=.05, max_depth=5,
                                subsample=.8, colsample_bytree=.6, reg_lambda=1,
                                random_state=SEED, verbosity=0, n_jobs=-1)
    rf_f.fit(Xfs, yf, sample_weight=wf)
    lg_f.fit(Xfs, yf, sample_weight=wf)
    xg_f.fit(Xfs, yf, sample_weight=wf)
    oof1[vai] = (rf_f.predict(Xval) + lg_f.predict(Xval) + xg_f.predict(Xval)) / 3
    print(f"  Fold {fi+1}/5  MAE={mean_absolute_error(y_tr[vai], oof1[vai]):.3f}")

# final model on full train
w_full = make_weights(y_tr)
X_aug1, y_aug1 = oversample(X_tr, y_tr, seed=SEED)
w_aug1 = make_weights(y_aug1)
sel1   = SelectKBest(f_regression, k=80)
Xtr1   = sel1.fit_transform(X_aug1, y_aug1)
Xte1   = sel1.transform(X_te)

rf1  = RandomForestRegressor(300, max_features=.3, min_samples_leaf=4, random_state=SEED, n_jobs=-1)
lg1  = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=63, subsample=.8,
                           colsample_bytree=.6, reg_lambda=1, random_state=SEED, verbose=-1, n_jobs=-1)
xg1  = xgb.XGBRegressor(n_estimators=400, learning_rate=.05, max_depth=5, subsample=.8,
                          colsample_bytree=.6, reg_lambda=1, random_state=SEED, verbosity=0, n_jobs=-1)
rf1.fit(Xtr1, y_aug1, sample_weight=w_aug1)
lg1.fit(Xtr1, y_aug1, sample_weight=w_aug1)
xg1.fit(Xtr1, y_aug1, sample_weight=w_aug1)
p_ens = (rf1.predict(Xte1) + lg1.predict(Xte1) + xg1.predict(Xte1)) / 3

joblib.dump({"rf": rf1, "lgb": lg1, "xgb": xg1, "selector": sel1,
             "winsor_lo": winsor_lo, "winsor_hi": winsor_hi},
            MODELS_DIR / "ensemble_best_mae.pkl")
print(f"  Saved → {MODELS_DIR}/ensemble_best_mae.pkl\n")

m1_oof  = metrics(y_tr, oof1, "OOF")
m1_test = metrics(y_te, p_ens, "Test")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — PLS direct (n=5)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MODEL 2: PLS Direct (n_components=5)")
print("=" * 60)

oof2 = np.zeros(len(X_tr))
for fi, (tri, vai) in enumerate(kf.split(X_tr, y_tr, ids_tr)):
    pls_f = PLSRegression(n_components=5, scale=True)
    pls_f.fit(X_tr[tri], y_tr[tri])
    oof2[vai] = pls_f.predict(X_tr[vai]).ravel()
    print(f"  Fold {fi+1}/5  MAE={mean_absolute_error(y_tr[vai], oof2[vai]):.3f}")

pls1 = PLSRegression(n_components=5, scale=True)
pls1.fit(X_tr, y_tr)
p_pls = pls1.predict(X_te).ravel()

joblib.dump({"pls": pls1, "winsor_lo": winsor_lo, "winsor_hi": winsor_hi},
            MODELS_DIR / "pls_best_auc.pkl")
print(f"  Saved → {MODELS_DIR}/pls_best_auc.pkl\n")

m2_oof  = metrics(y_tr, oof2, "OOF")
m2_test = metrics(y_te, p_pls, "Test")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Meta-stacking: RF+LGB+XGB OOF → Ridge meta-learner
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MODEL 3: Meta-Stacking Ridge  (base: RF+LGB+XGB)")
print("=" * 60)

oof_base  = np.zeros((len(X_tr), 3))
test_base = np.zeros((len(X_te), 3))

for fi, (tri, vai) in enumerate(kf.split(X_tr, y_tr, ids_tr)):
    Xf, yf = oversample(X_tr[tri], y_tr[tri], seed=SEED + fi)
    wf = make_weights(yf)
    rf_f  = RandomForestRegressor(300, max_features=.3, min_samples_leaf=4, random_state=SEED, n_jobs=-1)
    lg_f  = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=63, subsample=.8,
                                colsample_bytree=.6, reg_lambda=1, random_state=SEED, verbose=-1, n_jobs=-1)
    xg_f  = xgb.XGBRegressor(n_estimators=400, learning_rate=.05, max_depth=5, subsample=.8,
                               colsample_bytree=.6, reg_lambda=1, random_state=SEED, verbosity=0, n_jobs=-1)
    rf_f.fit(Xf, yf, sample_weight=wf)
    lg_f.fit(Xf, yf, sample_weight=wf)
    xg_f.fit(Xf, yf, sample_weight=wf)
    oof_base[vai, 0] = rf_f.predict(X_tr[vai])
    oof_base[vai, 1] = lg_f.predict(X_tr[vai])
    oof_base[vai, 2] = xg_f.predict(X_tr[vai])
    test_base[:, 0] += rf_f.predict(X_te) / N_FOLDS
    test_base[:, 1] += lg_f.predict(X_te) / N_FOLDS
    test_base[:, 2] += xg_f.predict(X_te) / N_FOLDS
    fold_meta = oof_base[vai].mean(1)
    print(f"  Fold {fi+1}/5  base OOF MAE={mean_absolute_error(y_tr[vai], fold_meta):.3f}")

# meta-learner
sc3    = StandardScaler()
meta3  = Ridge(alpha=0.5)
meta3.fit(sc3.fit_transform(oof_base), y_tr, sample_weight=w_full)
oof3   = meta3.predict(sc3.transform(oof_base))
p_meta = meta3.predict(sc3.transform(test_base))

# also save the full-data base models for inference
rf_full  = RandomForestRegressor(300, max_features=.3, min_samples_leaf=4, random_state=SEED, n_jobs=-1)
lg_full  = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=63, subsample=.8,
                               colsample_bytree=.6, reg_lambda=1, random_state=SEED, verbose=-1, n_jobs=-1)
xg_full  = xgb.XGBRegressor(n_estimators=400, learning_rate=.05, max_depth=5, subsample=.8,
                              colsample_bytree=.6, reg_lambda=1, random_state=SEED, verbosity=0, n_jobs=-1)
X_aug3, y_aug3 = oversample(X_tr, y_tr, seed=SEED)
w_aug3 = make_weights(y_aug3)
rf_full.fit(X_aug3, y_aug3, sample_weight=w_aug3)
lg_full.fit(X_aug3, y_aug3, sample_weight=w_aug3)
xg_full.fit(X_aug3, y_aug3, sample_weight=w_aug3)

joblib.dump({"rf": rf_full, "lgb": lg_full, "xgb": xg_full,
             "meta_ridge": meta3, "meta_scaler": sc3,
             "winsor_lo": winsor_lo, "winsor_hi": winsor_hi},
            MODELS_DIR / "meta_stack_best_sens.pkl")
print(f"  Saved → {MODELS_DIR}/meta_stack_best_sens.pkl\n")

m3_oof  = metrics(y_tr, oof3, "OOF")
m3_test = metrics(y_te, p_meta, "Test")

# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════
header = [
    "\n" + "=" * 82,
    "  FINAL RESULTS — 5-Fold CV (OOF) + Test set (n=95)",
    "=" * 82,
    f"\n  Test set: 0–7: n=7(7%)  7–10: n=41(43%)  10–14: n=38(40%)  14+: n=9(9%)",
    f"  Bias = mean(pred−truth). +ve = over-estimate, -ve = under-estimate.",
]

report_lines = header + [
    print_block("MODEL 1 — Ensemble RF+LGB+XGB  (SelectKBest k=80)", m1_oof, m1_test, band_rows(y_te, p_ens)),
    print_block("MODEL 2 — PLS Direct (n=5 components)",              m2_oof, m2_test, band_rows(y_te, p_pls)),
    print_block("MODEL 3 — Meta-Stacking Ridge on RF+LGB+XGB OOF",   m3_oof, m3_test, band_rows(y_te, p_meta)),
    "\n" + "=" * 82,
]

report = "\n".join(report_lines)
print(report)

out = RESULTS / "final_range_report.txt"
out.write_text(report)
print(f"\n  Saved report → {out}")
print(f"  Saved models → {MODELS_DIR}/")
print("\nDone.")
