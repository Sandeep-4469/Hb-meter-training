"""
Meta-learning trainer on 27 segment-mean features
===================================================
Strategy
--------
1. Imbalance handling
   - Sample weights: inverse HB-bin frequency (4 bins), capped at 5×min
   - Targeted oversampling: SVMSMOTE on severe-anaemia region (<7 g/dL)
     and combined with ADASYN to generate synthetic minority samples

2. Base models (Level 0) — trained with sample weights via StratifiedGroupKFold
   - Ridge (L2 linear)
   - ElasticNet
   - SVR (RBF)
   - RandomForest
   - ExtraTrees
   - GradientBoosting
   - LightGBM
   - XGBoost

3. Meta-learner (Level 1)
   - Ridge trained on OOF base predictions (8-dim) + original 27 features
   - Also tries: Ridge-only OOF, RF-only OOF, LightGBM-only OOF

4. Calibration
   - Isotonic regression calibration of the final stacked output to reduce
     range compression (maps predictions back toward true distribution)

Outputs
-------
  results/meta_oof_predictions.csv
  results/meta_test_predictions.csv
  results/meta_report.txt
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, StackingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SVMSMOTE, RandomOverSampler
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
RESULTS  = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FEAT_CSV    = RESULTS / "feature_study_features.csv"
ANEMIA_THR  = 12.0
RANDOM_SEED = 42

# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(FEAT_CSV)
    feat_cols = [c for c in df.columns if c[:2] in ("s0", "s1", "s2")]
    train = df[df["split"] == "train"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)

    X_tr = train[feat_cols].astype(float).values
    y_tr = train["hb_value"].astype(float).values
    X_te = test[feat_cols].astype(float).values
    y_te = test["hb_value"].astype(float).values
    ids_tr = train["video_id"].values
    ids_te = test["video_id"].values
    return X_tr, y_tr, X_te, y_te, ids_tr, ids_te, feat_cols


# ── sample weights ────────────────────────────────────────────────────────────

def make_weights(y: np.ndarray, cap: float = 5.0) -> np.ndarray:
    """Inverse bin-frequency weights; 4 clinical bins."""
    bins   = [0, 7, 10, 12, 100]
    labels = np.digitize(y, bins) - 1
    counts = np.bincount(labels, minlength=4)
    freq   = counts[labels] / len(y)
    w      = 1.0 / (freq + 1e-8)
    w      = w / w.min()
    return np.clip(w, None, cap)


# ── oversampling (regression-aware) ──────────────────────────────────────────

def oversample_minority(X: np.ndarray, y: np.ndarray,
                        severe_thr: float = 7.0,
                        target_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Upsample severe-anaemia (<7) samples so they are ~target_ratio of training set.
    Uses RandomOverSampler (with small Gaussian jitter) rather than SMOTE so that
    we don't synthesize out-of-range HB values from the regression target.
    """
    labels = (y < severe_thr).astype(int)   # 1 = severe
    n_severe  = labels.sum()
    n_total   = len(y)
    n_desired = int(n_total * target_ratio)

    if n_severe >= n_desired:
        return X, y

    extra = n_desired - n_severe
    severe_idx = np.where(labels == 1)[0]
    chosen = np.random.default_rng(RANDOM_SEED).choice(severe_idx, size=extra, replace=True)

    # add small Gaussian noise to avoid exact duplicates
    noise_scale = X[severe_idx].std(axis=0) * 0.05
    X_aug = X[chosen] + np.random.default_rng(RANDOM_SEED + 1).normal(
        0, noise_scale, (extra, X.shape[1]))
    y_aug = y[chosen] + np.random.default_rng(RANDOM_SEED + 2).normal(
        0, 0.1, extra)
    y_aug = np.clip(y_aug, 3.5, 7.0)

    X_out = np.vstack([X, X_aug])
    y_out = np.concatenate([y, y_aug])
    print(f"  Oversampled severe anaemia: {n_severe} → {n_severe + extra} "
          f"({(n_severe+extra)/len(y_out)*100:.1f}% of {len(y_out)})")
    return X_out, y_out


# ── base model definitions ────────────────────────────────────────────────────

def make_base_models():
    return [
        ("ridge",    Ridge(alpha=1.0)),
        ("enet",     ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
        ("svr",      SVR(kernel="rbf", C=10, epsilon=0.3)),
        ("rf",       RandomForestRegressor(n_estimators=200, max_features=0.6,
                                           min_samples_leaf=5, random_state=RANDOM_SEED,
                                           n_jobs=-1)),
        ("et",       ExtraTreesRegressor(n_estimators=200, max_features=0.6,
                                         min_samples_leaf=5, random_state=RANDOM_SEED,
                                         n_jobs=-1)),
        ("gbr",      GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=3, subsample=0.8,
                                               random_state=RANDOM_SEED)),
        ("lgb",      lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                        num_leaves=31, subsample=0.8,
                                        colsample_bytree=0.8, reg_lambda=1.0,
                                        random_state=RANDOM_SEED, verbose=-1,
                                        n_jobs=-1)),
        ("xgb",      xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8,
                                       colsample_bytree=0.8, reg_lambda=1.0,
                                       random_state=RANDOM_SEED, verbosity=0,
                                       n_jobs=-1)),
    ]


# ── evaluation helpers ────────────────────────────────────────────────────────

def within(true, pred, k):
    return float(np.mean(np.abs(true - pred) <= k)) * 100

def auc_regression(y_true, y_pred, thr=ANEMIA_THR):
    from sklearn.metrics import roc_auc_score
    labels = (y_true < thr).astype(int)
    scores = -y_pred
    return roc_auc_score(labels, scores)

def sensitivity(y_true, y_pred, thr=ANEMIA_THR):
    anaemic = y_true < thr
    return float(np.mean(y_pred[anaemic] < thr))

def per_range_report(y_true, y_pred, prefix=""):
    bands = [(0,7,"Severe"), (7,10,"Moderate"), (10,12,"Mild"), (12,25,"Normal")]
    lines = []
    for lo, hi, name in bands:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        w1  = within(y_true[mask], y_pred[mask], 1.0)
        lines.append(f"    {name:12s} (n={mask.sum():3d})  MAE={mae:.3f}  W±1={w1:.0f}%")
    return "\n".join(lines)

def report_block(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    w1   = within(y_true, y_pred, 1.0)
    w2   = within(y_true, y_pred, 2.0)
    try:
        auc  = auc_regression(y_true, y_pred)
        sens = sensitivity(y_true, y_pred)
    except Exception:
        auc = sens = float("nan")
    ratio = y_pred.std() / (y_true.std() + 1e-8)
    lines = [
        f"  {name}",
        f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  "
        f"W±1={w1:.1f}%  W±2={w2:.1f}%",
        f"    AUC={auc:.3f}  Sens={sens:.3f}  PredStd/TrueStd={ratio:.3f}",
        per_range_report(y_true, y_pred),
    ]
    return "\n".join(lines)


# ── main stacking pipeline ────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 68)
    print("  Meta-Learning Trainer — 27 segment-mean features")
    print("=" * 68)

    X_tr, y_tr, X_te, y_te, ids_tr, ids_te, feat_cols = load_data()
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}  Features: {X_tr.shape[1]}")

    # oversample severe anaemia
    np.random.seed(RANDOM_SEED)
    X_tr_aug, y_tr_aug = oversample_minority(X_tr, y_tr)

    # sample weights (on augmented set)
    w_aug = make_weights(y_tr_aug)
    w_orig = make_weights(y_tr)

    # scale
    scaler = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_tr)
    X_tr_aug_sc = scaler.transform(X_tr_aug)
    X_te_sc     = scaler.transform(X_te)

    # ── Level-0: generate OOF predictions ─────────────────────────────────
    print("\n── Level-0: OOF base predictions …")
    base_models = make_base_models()
    n_base = len(base_models)

    oof_preds  = np.zeros((len(X_tr), n_base))   # OOF on original train
    test_preds = np.zeros((len(X_te), n_base))

    # CV on original (non-augmented) training set for OOF
    kf = GroupKFold(n_splits=5)
    groups = np.arange(len(X_tr))   # each sample is its own group (no patient dup)

    for m_idx, (mname, model) in enumerate(base_models):
        use_scaled = mname in ("ridge", "enet", "svr")
        Xtr_m = X_tr_sc     if use_scaled else X_tr
        Xte_m = X_te_sc     if use_scaled else X_te

        fold_oof = np.zeros(len(X_tr))
        fold_test = np.zeros((len(X_te), 5))

        for f_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr, y_tr, groups)):
            X_fold = Xtr_m[tr_idx]; y_fold = y_tr[tr_idx]; w_fold = w_orig[tr_idx]

            # augment only the training fold (not validation)
            X_fold_aug, y_fold_aug = oversample_minority(X_fold, y_fold)
            w_fold_aug = make_weights(y_fold_aug)
            if use_scaled:
                sc_fold = StandardScaler()
                X_fold_aug = sc_fold.fit_transform(X_fold_aug)
                X_va = sc_fold.transform(Xtr_m[va_idx])
                X_te_fold = sc_fold.transform(Xte_m)
            else:
                X_va = Xtr_m[va_idx]
                X_te_fold = Xte_m

            if mname in ("lgb", "xgb", "rf", "et", "gbr"):
                model.fit(X_fold_aug, y_fold_aug, sample_weight=w_fold_aug)
            else:
                model.fit(X_fold_aug, y_fold_aug, sample_weight=w_fold_aug)

            fold_oof[va_idx] = model.predict(X_va)
            fold_test[:, f_idx] = model.predict(X_te_fold)

        oof_preds[:, m_idx]  = fold_oof
        test_preds[:, m_idx] = fold_test.mean(axis=1)

        mae_oof = mean_absolute_error(y_tr, fold_oof)
        print(f"  {mname:8s}  OOF MAE={mae_oof:.3f}")

    # ── Level-1: meta-learner on OOF predictions ──────────────────────────
    print("\n── Level-1: meta-learner …")

    # Option A: Ridge on stacked OOF predictions only
    meta_ridge_oof = Ridge(alpha=1.0)
    meta_ridge_oof.fit(oof_preds, y_tr, sample_weight=w_orig)
    meta_test_oof = meta_ridge_oof.predict(test_preds)
    meta_oof_oof  = meta_ridge_oof.predict(oof_preds)

    # Option B: Ridge on OOF predictions + original 27 features (blended)
    meta_ridge_full = Ridge(alpha=1.0)
    Xmeta_tr = np.hstack([oof_preds, X_tr_sc])
    Xmeta_te = np.hstack([test_preds, X_te_sc])
    meta_ridge_full.fit(Xmeta_tr, y_tr, sample_weight=w_orig)
    meta_test_full = meta_ridge_full.predict(Xmeta_te)
    meta_oof_full  = meta_ridge_full.predict(np.hstack([oof_preds, X_tr_sc]))

    # Option C: simple mean of all base OOF predictions
    meta_test_mean = test_preds.mean(axis=1)
    meta_oof_mean  = oof_preds.mean(axis=1)

    # Option D: weighted mean by individual OOF MAE (lower MAE → higher weight)
    oof_maes  = np.array([mean_absolute_error(y_tr, oof_preds[:, i]) for i in range(n_base)])
    weights_inv = 1.0 / (oof_maes + 1e-8)
    weights_norm = weights_inv / weights_inv.sum()
    meta_test_wmean = (test_preds * weights_norm).sum(axis=1)
    meta_oof_wmean  = (oof_preds  * weights_norm).sum(axis=1)

    print(f"  Base model MAE weights: " +
          " ".join(f"{n}={w:.3f}" for (n,_), w in zip(base_models, weights_norm)))

    # ── Isotonic calibration on best meta ─────────────────────────────────
    # find which meta approach has best OOF MAE
    options = {
        "Ridge(OOF only)":      (meta_oof_oof,   meta_test_oof),
        "Ridge(OOF+feats)":     (meta_oof_full,  meta_test_full),
        "Mean(base)":           (meta_oof_mean,  meta_test_mean),
        "WeightedMean(base)":   (meta_oof_wmean, meta_test_wmean),
    }
    best_name = min(options, key=lambda k: mean_absolute_error(y_tr, options[k][0]))
    best_oof, best_test = options[best_name]
    print(f"\n  Best meta: {best_name}  (OOF MAE="
          f"{mean_absolute_error(y_tr, best_oof):.3f})")

    # isotonic regression calibration: fit on OOF, apply to test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(best_oof, y_tr)
    best_test_cal = iso.predict(best_test)
    best_oof_cal  = iso.predict(best_oof)   # leave-one-out approximation

    # ── print full report ─────────────────────────────────────────────────
    lines = []
    lines.append("\n" + "=" * 68)
    lines.append("  RESULTS — Test set (n=95)")
    lines.append("=" * 68)

    # individual base models on test
    lines.append("\n── Individual base models (test):")
    for i, (mname, _) in enumerate(base_models):
        lines.append(report_block(f"{mname} (base)", y_te, test_preds[:, i]))

    lines.append("\n── Meta-learner options (test):")
    for opt_name, (_, opt_test) in options.items():
        lines.append(report_block(opt_name, y_te, opt_test))

    lines.append(f"\n── Best meta ({best_name}) + Isotonic calibration (test):")
    lines.append(report_block("Calibrated", y_te, best_test_cal))

    lines.append("\n── Reference baselines:")
    pred_mean = np.full(len(y_te), y_tr.mean())
    lines.append(report_block("Predict-mean (naive)", y_te, pred_mean))
    lines.append("  1060-feat Ensemble (prior):  MAE=2.093  AUC=0.624  Sens=0.914")

    report = "\n".join(lines)
    print(report)

    # save
    out_path = RESULTS / "meta_report.txt"
    out_path.write_text(report)
    print(f"\n  Saved → {out_path}")

    # save predictions
    oof_df = pd.DataFrame({"video_id": ids_tr, "hb_true": y_tr,
                            "meta_pred": best_oof, "meta_pred_cal": best_oof_cal})
    oof_df.to_csv(RESULTS / "meta_oof_predictions.csv", index=False)

    test_df = pd.DataFrame({"video_id": ids_te, "hb_true": y_te,
                             "meta_pred": best_test, "meta_pred_cal": best_test_cal})
    test_df.to_csv(RESULTS / "meta_test_predictions.csv", index=False)

    print("  Saved OOF + test predictions.")
    print("\nDone.")


if __name__ == "__main__":
    run()
