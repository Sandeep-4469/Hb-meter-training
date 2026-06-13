"""
Meta-Learning Trainer — 1060 combined features
===============================================
Stacking pipeline with imbalance handling on the full spatial+temporal feature set.

Dimensionality reduction used:
  - Linear models: SelectKBest(f_regression, k=80) → StandardScaler
  - PLS base model: PLSRegression(n_components=5) [best AUC from dimred study]
  - Tree models: all 1060 features (implicit feature selection via splits)

Meta-learner:
  Ridge / WeightedMean / LGB trained on 8-dim OOF stack

Imbalance handling:
  - Sample weights: inverse HB-bin frequency, cap=5×
  - Oversampling: jittered duplication of severe anaemia (<7 g/dL) per fold → 15%
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_CSV  = RESULTS / "features_combined_train.csv"
TEST_CSV   = RESULTS / "features_combined_test.csv"
ANEMIA_THR = 12.0
SEED       = 42
N_FOLDS    = 5
K_SELECT   = 80    # SelectKBest k for linear models
N_PLS      = 5     # PLS components (optimal from dimred study)
WINSOR_LO  = 1.0   # winsorise features at percentile 1/99 on training data
WINSOR_HI  = 99.0


# ── data ─────────────────────────────────────────────────────────────────────

def load_data():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    meta_cols = {"video_id", "hb_value", "split", "protocol"}
    feat_cols = [c for c in tr.columns if c not in meta_cols]
    X_tr = tr[feat_cols].astype(float).fillna(0).values
    y_tr = tr["hb_value"].astype(float).values
    X_te = te[feat_cols].astype(float).fillna(0).values
    y_te = te["hb_value"].astype(float).values

    # Winsorise extreme features using training percentiles.
    # Kurtosis features (e.g. seg2_V_kurt) reach 165 883 due to 30s R-channel
    # saturation; StandardScaler then produces ~15σ values that blow up Ridge/PLS.
    lo = np.percentile(X_tr, WINSOR_LO, axis=0)
    hi = np.percentile(X_tr, WINSOR_HI, axis=0)
    X_tr = np.clip(X_tr, lo, hi)
    X_te = np.clip(X_te, lo, hi)   # apply train percentiles to test (no leakage)
    print(f"  Winsorised at [{WINSOR_LO}th, {WINSOR_HI}th] pct.  "
          f"New max abs: {np.abs(X_tr).max():.1f}")

    return X_tr, y_tr, X_te, y_te, tr["video_id"].values, te["video_id"].values


# ── imbalance ─────────────────────────────────────────────────────────────────

def make_weights(y: np.ndarray, cap: float = 5.0) -> np.ndarray:
    bins   = [0, 7, 10, 12, 100]
    labels = np.digitize(y, bins) - 1
    counts = np.bincount(labels, minlength=4)
    freq   = counts[labels] / len(y)
    w      = 1.0 / (freq + 1e-8)
    w      = w / w.min()
    return np.clip(w, None, cap)


def oversample_fold(X, y, severe_thr=7.0, target_frac=0.15, seed=SEED):
    rng     = np.random.default_rng(seed)
    idx_sev = np.where(y < severe_thr)[0]
    n_sev   = len(idx_sev)
    n_want  = int(len(y) * target_frac)
    if n_sev >= n_want or n_sev == 0:
        return X, y
    extra   = n_want - n_sev
    chosen  = rng.choice(idx_sev, size=extra, replace=True)
    noise   = rng.normal(0, X[idx_sev].std(axis=0) * 0.03 + 1e-8, (extra, X.shape[1]))
    y_noise = np.clip(y[chosen] + rng.normal(0, 0.15, extra), 3.0, 7.0)
    return np.vstack([X, X[chosen] + noise]), np.concatenate([y, y_noise])


# ── preprocessors (fit on fold train, applied to val/test) ───────────────────

class LinearPreproc:
    """SelectKBest(f_regression, k) → StandardScaler — fitted per fold."""
    def __init__(self, k=K_SELECT):
        self.sel = SelectKBest(f_regression, k=k)
        self.sc  = StandardScaler()

    def fit_transform(self, X, y):
        return self.sc.fit_transform(self.sel.fit_transform(X, y))

    def transform(self, X):
        return self.sc.transform(self.sel.transform(X))


class PLSPreproc:
    """PLSRegression(n_components) used as a supervised feature extractor."""
    def __init__(self, n=N_PLS):
        self.pls = PLSRegression(n_components=n, scale=True)

    def fit_transform(self, X, y):
        self.pls.fit(X, y)
        return self.pls.transform(X)

    def transform(self, X):
        return self.pls.transform(X)


# ── model configs ─────────────────────────────────────────────────────────────
# Each entry: (name, preproc_factory, model, supports_sample_weight)

def base_model_configs():
    return [
        ("ridge",  LinearPreproc,  Ridge(alpha=1.0),                                   True),
        ("enet",   LinearPreproc,  ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=5000), True),
        ("svr",    LinearPreproc,  SVR(kernel="rbf", C=10, epsilon=0.2),               False),
        ("pls",    PLSPreproc,     Ridge(alpha=0.1),                                   True),
        ("rf",     None,           RandomForestRegressor(n_estimators=300, max_features=0.3,
                                       min_samples_leaf=4, random_state=SEED, n_jobs=-1), True),
        ("et",     None,           ExtraTreesRegressor(n_estimators=300, max_features=0.3,
                                       min_samples_leaf=4, random_state=SEED, n_jobs=-1),  True),
        ("lgb",    None,           lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04,
                                       num_leaves=63, subsample=0.8, colsample_bytree=0.6,
                                       reg_lambda=1.0, min_child_samples=10,
                                       random_state=SEED, verbose=-1, n_jobs=-1),          True),
        ("xgb",    None,           xgb.XGBRegressor(n_estimators=400, learning_rate=0.04,
                                       max_depth=5, subsample=0.8, colsample_bytree=0.6,
                                       reg_lambda=1.0, min_child_weight=5,
                                       random_state=SEED, verbosity=0, n_jobs=-1),         True),
    ]


# ── evaluation ────────────────────────────────────────────────────────────────

def within(yt, yp, k):    return float(np.mean(np.abs(yt - yp) <= k)) * 100
def auc(yt, yp):          return roc_auc_score((yt < ANEMIA_THR).astype(int), -yp)
def sens(yt, yp):         m = yt < ANEMIA_THR; return float(np.mean(yp[m] < ANEMIA_THR)) if m.any() else float("nan")
def f1(yt, yp):           return f1_score((yt < ANEMIA_THR).astype(int), (yp < ANEMIA_THR).astype(int))

def block(tag, yt, yp):
    mae  = mean_absolute_error(yt, yp)
    rmse = mean_squared_error(yt, yp) ** 0.5
    r2   = r2_score(yt, yp)
    w1   = within(yt, yp, 1.0)
    w2   = within(yt, yp, 2.0)
    try:    a, s, f = auc(yt, yp), sens(yt, yp), f1(yt, yp)
    except: a = s = f = float("nan")
    ratio = yp.std() / (yt.std() + 1e-8)
    bands = []
    for lo, hi, name in [(0,7,"Severe"),(7,10,"Moderate"),(10,12,"Mild"),(12,25,"Normal")]:
        m = (yt >= lo) & (yt < hi)
        if not m.any(): continue
        bands.append(f"    {name:10s} n={m.sum():3d}  MAE={mean_absolute_error(yt[m],yp[m]):.3f}"
                     f"  W±1={within(yt[m],yp[m],1):.0f}%  W±2={within(yt[m],yp[m],2):.0f}%")
    return "\n".join([
        f"  {tag}",
        f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  W±1={w1:.1f}%  W±2={w2:.1f}%",
        f"    AUC={a:.3f}  Sens={s:.3f}  F1={f:.3f}  PredStd/TrueStd={ratio:.3f}",
    ] + bands)


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 70)
    print("  Meta-Learning — 1060 features  (SelectKBest/PLS/full per model)")
    print("=" * 70)

    X_tr, y_tr, X_te, y_te, ids_tr, ids_te = load_data()
    print(f"  Train: {len(X_tr)}   Test: {len(X_te)}   Features: {X_tr.shape[1]}")

    w_orig  = make_weights(y_tr)
    configs = base_model_configs()
    n_base  = len(configs)
    kf      = GroupKFold(n_splits=N_FOLDS)

    oof_preds  = np.zeros((len(X_tr), n_base))
    test_preds = np.zeros((len(X_te), n_base))

    print(f"\n── Level-0: OOF base predictions …")
    for m_idx, (mname, preproc_cls, model, use_sw) in enumerate(configs):
        fold_oof  = np.zeros(len(X_tr))
        fold_test = np.zeros((len(X_te), N_FOLDS))

        for f_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_tr, ids_tr)):
            Xf, yf = oversample_fold(X_tr[tr_idx], y_tr[tr_idx], seed=SEED + f_idx)
            wf     = make_weights(yf)

            # apply dimensionality reduction / scaling per fold
            if preproc_cls is not None:
                pp = preproc_cls()
                Xf_t  = pp.fit_transform(Xf, yf)
                Xva_t = pp.transform(X_tr[va_idx])
                Xte_t = pp.transform(X_te)
            else:
                Xf_t  = Xf
                Xva_t = X_tr[va_idx]
                Xte_t = X_te

            if use_sw:
                model.fit(Xf_t, yf, sample_weight=wf)
            else:
                model.fit(Xf_t, yf)

            fold_oof[va_idx]    = model.predict(Xva_t)
            fold_test[:, f_idx] = model.predict(Xte_t)

        oof_preds[:, m_idx]  = fold_oof
        test_preds[:, m_idx] = fold_test.mean(axis=1)

        oof_mae = mean_absolute_error(y_tr, fold_oof)
        oof_auc = auc(y_tr, fold_oof)
        print(f"  {mname:8s}  OOF MAE={oof_mae:.3f}  AUC={oof_auc:.3f}")

    # ── Level-1: meta-learners ────────────────────────────────────────────
    print("\n── Level-1: meta-learners …")

    # A: Ridge on OOF stack
    sc_a = StandardScaler()
    Xm_tr = sc_a.fit_transform(oof_preds)
    Xm_te = sc_a.transform(test_preds)
    meta_a = Ridge(alpha=0.5)
    meta_a.fit(Xm_tr, y_tr, sample_weight=w_orig)
    oof_a  = meta_a.predict(Xm_tr)
    test_a = meta_a.predict(Xm_te)

    # B: Ridge on OOF stack + SelectKBest(80) original features
    sel_b = SelectKBest(f_regression, k=K_SELECT)
    sc_b  = StandardScaler()
    Xorig_tr = sc_b.fit_transform(sel_b.fit_transform(X_tr, y_tr))
    Xorig_te = sc_b.transform(sel_b.transform(X_te))
    meta_b = Ridge(alpha=0.5)
    meta_b.fit(np.hstack([Xm_tr, Xorig_tr]), y_tr, sample_weight=w_orig)
    oof_b  = meta_b.predict(np.hstack([Xm_tr, Xorig_tr]))
    test_b = meta_b.predict(np.hstack([Xm_te, Xorig_te]))

    # C: simple mean
    oof_c, test_c = oof_preds.mean(1), test_preds.mean(1)

    # D: inverse-MAE weighted mean
    inv_w  = 1.0 / (np.array([mean_absolute_error(y_tr, oof_preds[:, i]) for i in range(n_base)]) + 1e-8)
    inv_w /= inv_w.sum()
    oof_d  = (oof_preds  * inv_w).sum(1)
    test_d = (test_preds * inv_w).sum(1)

    # E: LGB meta-learner (non-linear blending)
    meta_e = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                                 num_leaves=15, reg_lambda=5.0,
                                 random_state=SEED, verbose=-1, n_jobs=-1)
    meta_e.fit(Xm_tr, y_tr, sample_weight=w_orig)
    oof_e, test_e = meta_e.predict(Xm_tr), meta_e.predict(Xm_te)

    options = {
        "Ridge(OOF only)":    (oof_a, test_a),
        "Ridge(OOF+feats)":   (oof_b, test_b),
        "Mean(base)":         (oof_c, test_c),
        "WeightedMean(base)": (oof_d, test_d),
        "LGB(OOF only)":      (oof_e, test_e),
    }

    best_name = min(options, key=lambda k: mean_absolute_error(y_tr, options[k][0]))
    best_oof, best_test = options[best_name]
    print(f"\n  Best meta: {best_name}  "
          f"OOF MAE={mean_absolute_error(y_tr, best_oof):.3f}  "
          f"AUC={auc(y_tr, best_oof):.3f}")
    print(f"  MAE weights: " +
          " ".join(f"{n}={w:.3f}" for (n,*_), w in zip(configs, inv_w)))

    # isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(best_oof, y_tr)
    cal_test = iso.predict(best_test)

    # ── report ────────────────────────────────────────────────────────────
    lines  = ["\n" + "=" * 70, "  RESULTS — Test set (n=95)", "=" * 70]
    lines += ["\n── Individual base models:"]
    for i, (mname, *_) in enumerate(configs):
        lines.append(block(mname, y_te, test_preds[:, i]))
    lines += ["\n── Meta-learner options:"]
    for opt_name, (_, opt_test) in options.items():
        lines.append(block(opt_name, y_te, opt_test))
    lines += [f"\n── {best_name} + Isotonic calibration:"]
    lines.append(block("Calibrated", y_te, cal_test))
    lines += ["\n── Reference baselines:"]
    lines.append(block("Predict-mean (naive)", y_te, np.full(len(y_te), y_tr.mean())))
    lines.append("  Prior best (no meta):  MAE=2.093  AUC=0.624  Sens=0.914")
    lines.append("  PLS direct (best AUC): MAE=2.332  AUC=0.687  Sens=0.900")

    report = "\n".join(lines)
    print(report)

    (RESULTS / "meta1060_report.txt").write_text(report)
    pd.DataFrame({"video_id": ids_tr, "hb_true": y_tr,
                  "meta_pred": best_oof, "meta_pred_cal": iso.predict(best_oof)}
                ).to_csv(RESULTS / "meta1060_oof_predictions.csv", index=False)
    pd.DataFrame({"video_id": ids_te, "hb_true": y_te,
                  "meta_pred": best_test, "meta_pred_cal": cal_test}
                ).to_csv(RESULTS / "meta1060_test_predictions.csv", index=False)

    print(f"\n  Saved → {RESULTS}/meta1060_*.  Done.")


if __name__ == "__main__":
    run()
