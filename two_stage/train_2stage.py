"""
Two-Stage Hemoglobin Prediction
================================
Stage 1: 4-class severity band classifier
  Bands: <7  (severe anemia)
         7-10 (moderate anemia)
         10-12 (mild anemia)
         >12  (normal)

Stage 2: per-band GradientBoosting regressor
  Each regressor trained on a soft window ±1.5 g/dL around band edges
  to avoid sharp boundary effects.

Features: 1060 combined (820 histogram + 240 temporal)
CV: 5-fold StratifiedGroupKFold (no patient leakage)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RES_DIR = Path(__file__).parent / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── severity band definitions ──────────────────────────────────────────────
BANDS = [
    ("severe",   0.0,  7.0),
    ("moderate", 7.0, 10.0),
    ("mild",    10.0, 12.0),
    ("normal",  12.0, 30.0),
]
SOFT_MARGIN = 1.5   # g/dL expansion per edge for Stage-2 training data
K_CLASS = 60        # features for classifier
K_REG   = 40        # features per band regressor

HB_MIN, HB_MAX = 3.0, 20.0

# ── band helpers ───────────────────────────────────────────────────────────

def assign_band(hb: float) -> int:
    for i, (_, lo, hi) in enumerate(BANDS):
        if hb < hi:
            return i
    return len(BANDS) - 1


def band_name(idx: int) -> str:
    return BANDS[idx][0]


# ── metrics ────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    w1   = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    w2   = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
    return mae, rmse, r2, w1, w2


def anemia_metrics(y_true_hb, y_pred_hb, threshold=12.0):
    y_true_bin = (y_true_hb < threshold).astype(int)
    y_pred_bin = (y_pred_hb < threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin,
                                       labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    # soft AUC using inverted predicted HB as score
    try:
        auc = roc_auc_score(y_true_bin, -y_pred_hb)
    except Exception:
        auc = float("nan")
    f1_num = 2 * tp
    f1_den = 2 * tp + fp + fn
    f1  = f1_num / (f1_den + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return dict(AUC=auc, Sens=sens, Spec=spec, F1=f1, Acc=acc)


# ── Stage-1 classifier ─────────────────────────────────────────────────────

def stage1_pipeline(X_tr, y_band_tr, X_te):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    scl = StandardScaler()
    X_tr = scl.fit_transform(X_tr)
    X_te = scl.transform(X_te)

    sel = SelectKBest(mutual_info_classif, k=min(K_CLASS, X_tr.shape[1]))
    X_tr = sel.fit_transform(X_tr, y_band_tr)
    X_te = sel.transform(X_te)

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    clf.fit(X_tr, y_band_tr)
    proba = clf.predict_proba(X_te)          # shape (N_te, 4)
    pred  = clf.predict(X_te)
    return pred, proba


# ── Stage-2 band regressor ─────────────────────────────────────────────────

def stage2_pipeline(X_tr_full, y_hb_tr_full, X_te_full, band_idx):
    lo = BANDS[band_idx][1] - SOFT_MARGIN
    hi = BANDS[band_idx][2] + SOFT_MARGIN
    # train only on samples near this band
    mask = (y_hb_tr_full >= lo) & (y_hb_tr_full < hi)
    if mask.sum() < 5:
        # fallback: use all data
        mask = np.ones(len(y_hb_tr_full), dtype=bool)

    X_b = X_tr_full[mask]
    y_b = y_hb_tr_full[mask]

    imp = SimpleImputer(strategy="median")
    X_b = imp.fit_transform(X_b)
    X_te = imp.transform(X_te_full)

    scl = StandardScaler()
    X_b  = scl.fit_transform(X_b)
    X_te = scl.transform(X_te)

    sel = SelectKBest(mutual_info_regression, k=min(K_REG, X_b.shape[1]))
    X_b  = sel.fit_transform(X_b, y_b)
    X_te = sel.transform(X_te)

    reg = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=4,
        loss="huber", random_state=42)
    reg.fit(X_b, y_b)
    preds = reg.predict(X_te)
    preds = np.clip(preds, HB_MIN, HB_MAX)
    return preds


# ── Soft weighted combination ──────────────────────────────────────────────

def soft_combine(band_preds, proba):
    """
    Weighted average of per-band predictions using classifier probabilities.
    band_preds: (N_te, 4)  each column is Stage-2 output for that band
    proba:      (N_te, 4)  classifier soft probabilities
    """
    return (band_preds * proba).sum(axis=1)


# ── Main CV loop ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Two-Stage HB Prediction  (temporal + histogram features)")
    print("=" * 70)

    train_path = RES_DIR / "features_combined_train.csv"
    test_path  = RES_DIR / "features_combined_test.csv"

    if not train_path.exists():
        print(f"\n  ERROR: {train_path} not found.")
        print("  Run  python two_stage/extract_temporal.py  first.\n")
        return

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    META = {"video_id", "hb_value", "split", "protocol", "anemia"}
    feat_cols = [c for c in train_df.columns if c not in META]
    print(f"\n  Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"  Features: {len(feat_cols)}")

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["hb_value"].values.astype(np.float32)
    groups  = train_df["video_id"].values

    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df["hb_value"].values.astype(np.float32)

    y_band_train = np.array([assign_band(h) for h in y_train])
    y_band_test  = np.array([assign_band(h) for h in y_test])

    # ── 5-fold cross-validation ────────────────────────────────────────────
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_hb_hard  = np.full(len(train_df), np.nan)   # hard band assignment
    oof_hb_soft  = np.full(len(train_df), np.nan)   # soft weighted average
    oof_band     = np.full(len(train_df), -1, dtype=int)

    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_band_train, groups)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        yb_tr      = y_band_train[tr_idx]

        # Stage 1
        band_pred_va, band_proba_va = stage1_pipeline(X_tr, yb_tr, X_va)
        oof_band[va_idx] = band_pred_va

        # Stage 2: train one regressor per band, predict on val
        band_preds_va = np.zeros((len(va_idx), len(BANDS)))
        for b in range(len(BANDS)):
            band_preds_va[:, b] = stage2_pipeline(X_tr, y_tr, X_va, b)

        # Hard prediction: use Stage-1 predicted band
        hb_hard = band_preds_va[np.arange(len(va_idx)), band_pred_va]
        hb_soft = soft_combine(band_preds_va, band_proba_va)

        oof_hb_hard[va_idx] = hb_hard
        oof_hb_soft[va_idx] = hb_soft

        mae_h, rmse_h, r2_h, w1_h, w2_h = regression_metrics(y_va, hb_hard)
        mae_s, rmse_s, r2_s, w1_s, w2_s = regression_metrics(y_va, hb_soft)

        # Stage-1 band accuracy
        yb_va = y_band_train[va_idx]
        band_acc = np.mean(band_pred_va == yb_va) * 100

        fold_results.append(dict(
            fold=fold+1,
            band_acc=band_acc,
            mae_hard=mae_h, rmse_hard=rmse_h, r2_hard=r2_h,
            w1_hard=w1_h,  w2_hard=w2_h,
            mae_soft=mae_s, rmse_soft=rmse_s, r2_soft=r2_s,
            w1_soft=w1_s,  w2_soft=w2_s,
        ))
        print(f"  Fold {fold+1}: BandAcc={band_acc:.1f}%  "
              f"MAE(hard)={mae_h:.3f}  MAE(soft)={mae_s:.3f}")

    # ── OOF summary ────────────────────────────────────────────────────────
    ok = ~np.isnan(oof_hb_hard)
    print(f"\n  OOF Results ({ok.sum()} samples):")

    mae_h, rmse_h, r2_h, w1_h, w2_h = regression_metrics(y_train[ok], oof_hb_hard[ok])
    mae_s, rmse_s, r2_s, w1_s, w2_s = regression_metrics(y_train[ok], oof_hb_soft[ok])

    print(f"  ─── Hard assignment ───────────────────────────────────")
    print(f"    MAE={mae_h:.3f}  RMSE={rmse_h:.3f}  R²={r2_h:.3f}")
    print(f"    Within±1={w1_h:.1f}%  Within±2={w2_h:.1f}%")

    print(f"  ─── Soft weighted  ────────────────────────────────────")
    print(f"    MAE={mae_s:.3f}  RMSE={rmse_s:.3f}  R²={r2_s:.3f}")
    print(f"    Within±1={w1_s:.1f}%  Within±2={w2_s:.1f}%")

    # Anemia classification metrics from regression predictions
    an_hard = anemia_metrics(y_train[ok], oof_hb_hard[ok])
    an_soft = anemia_metrics(y_train[ok], oof_hb_soft[ok])
    print(f"\n  Anemia Classification (AUC/Sens/Spec/F1):")
    print(f"    Hard: AUC={an_hard['AUC']:.3f}  Sens={an_hard['Sens']:.3f}  "
          f"Spec={an_hard['Spec']:.3f}  F1={an_hard['F1']:.3f}")
    print(f"    Soft: AUC={an_soft['AUC']:.3f}  Sens={an_soft['Sens']:.3f}  "
          f"Spec={an_soft['Spec']:.3f}  F1={an_soft['F1']:.3f}")

    # Per-band OOF breakdown
    print(f"\n  Per-Band OOF Regression (true band assignment):")
    print(f"  {'Band':<12} {'N':>5} {'MAE':>7} {'W±1%':>7}")
    for b, (bname, lo, hi) in enumerate(BANDS):
        mask_b = (y_train[ok] >= lo) & (y_train[ok] < hi) if b < len(BANDS)-1 \
                 else y_train[ok] >= lo
        if b == 0:
            mask_b = y_train[ok] < hi
        if b == len(BANDS)-1:
            mask_b = y_train[ok] >= lo

        yt_b = y_train[ok][mask_b]
        yp_b = oof_hb_soft[ok][mask_b]
        if len(yt_b) == 0:
            continue
        m_b = mean_absolute_error(yt_b, yp_b)
        w_b = np.mean(np.abs(yt_b - yp_b) <= 1.0) * 100
        print(f"  {bname:<12} {len(yt_b):>5} {m_b:>7.3f} {w_b:>6.1f}%")

    # ── Retrain on full train, evaluate on test ────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  Hold-out Test Set Evaluation")
    print(f"  {'─'*55}")

    y_band_train_all = np.array([assign_band(h) for h in y_train])

    # Stage 1: train on full train
    _, band_proba_test_all = stage1_pipeline(X_train, y_band_train_all, X_test)
    band_pred_test_all = np.argmax(band_proba_test_all, axis=1)

    # Stage 2: 4 band regressors
    band_preds_test = np.zeros((len(X_test), len(BANDS)))
    for b in range(len(BANDS)):
        band_preds_test[:, b] = stage2_pipeline(X_train, y_train, X_test, b)

    hb_test_hard = band_preds_test[np.arange(len(X_test)), band_pred_test_all]
    hb_test_soft = soft_combine(band_preds_test, band_proba_test_all)

    mae_h, rmse_h, r2_h, w1_h, w2_h = regression_metrics(y_test, hb_test_hard)
    mae_s, rmse_s, r2_s, w1_s, w2_s = regression_metrics(y_test, hb_test_soft)

    print(f"\n  ─── Hard assignment ───────────────────────────────────")
    print(f"    MAE={mae_h:.3f}  RMSE={rmse_h:.3f}  R²={r2_h:.3f}")
    print(f"    Within±1={w1_h:.1f}%  Within±2={w2_h:.1f}%")

    print(f"  ─── Soft weighted  ────────────────────────────────────")
    print(f"    MAE={mae_s:.3f}  RMSE={rmse_s:.3f}  R²={r2_s:.3f}")
    print(f"    Within±1={w1_s:.1f}%  Within±2={w2_s:.1f}%")

    an_h = anemia_metrics(y_test, hb_test_hard)
    an_s = anemia_metrics(y_test, hb_test_soft)
    print(f"\n  Anemia Classification:")
    print(f"    Hard: AUC={an_h['AUC']:.3f}  Sens={an_h['Sens']:.3f}  "
          f"Spec={an_h['Spec']:.3f}  F1={an_h['F1']:.3f}")
    print(f"    Soft: AUC={an_s['AUC']:.3f}  Sens={an_s['Sens']:.3f}  "
          f"Spec={an_s['Spec']:.3f}  F1={an_s['F1']:.3f}")

    # Per-band test breakdown
    print(f"\n  Per-Band Test Regression (true band):")
    print(f"  {'Band':<12} {'N':>5} {'MAE':>7} {'W±1%':>7}")
    for b, (bname, lo, hi) in enumerate(BANDS):
        if b == 0:
            mask_b = y_test < hi
        elif b == len(BANDS) - 1:
            mask_b = y_test >= lo
        else:
            mask_b = (y_test >= lo) & (y_test < hi)
        yt_b = y_test[mask_b]
        yp_b = hb_test_soft[mask_b]
        if len(yt_b) == 0:
            continue
        m_b = mean_absolute_error(yt_b, yp_b)
        w_b = np.mean(np.abs(yt_b - yp_b) <= 1.0) * 100
        print(f"  {bname:<12} {len(yt_b):>5} {m_b:>7.3f} {w_b:>6.1f}%")

    # Save predictions
    out_train = train_df[["video_id", "hb_value"]].copy()
    out_train["pred_hard"] = oof_hb_hard
    out_train["pred_soft"] = oof_hb_soft
    out_train["pred_band"] = oof_band
    out_train.to_csv(RES_DIR / "2stage_oof_predictions.csv", index=False)

    out_test = test_df[["video_id", "hb_value"]].copy()
    out_test["pred_hard"] = hb_test_hard
    out_test["pred_soft"] = hb_test_soft
    out_test["pred_band"] = band_pred_test_all
    out_test.to_csv(RES_DIR / "2stage_test_predictions.csv", index=False)

    print(f"\n  Saved predictions → {RES_DIR}")

    # Compression check
    print(f"\n  Prediction range check (test, soft):")
    print(f"    True HB  : {y_test.min():.2f} – {y_test.max():.2f}  "
          f"std={y_test.std():.3f}")
    print(f"    Predicted: {hb_test_soft.min():.2f} – {hb_test_soft.max():.2f}  "
          f"std={hb_test_soft.std():.3f}")
    print(f"    Compression ratio: {hb_test_soft.std()/y_test.std():.3f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
