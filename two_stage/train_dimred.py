"""
Dimensionality Reduction Comparison for NISHAD HB Prediction
=============================================================
Problem: 1060 features, 386 training samples (ratio 2.75x)
High-dimensional small-sample regime → need compressed representations.

Methods evaluated:
  1. PLS Regression         — spectrophotometry gold standard (supervised DR)
  2. PCA + RF               — unsupervised compression then nonlinear model
  3. PCA + Ensemble         — compressed features for ensemble
  4. Kernel PCA + RF        — nonlinear compression
  5. SelectKBest + Ensemble — current baseline (feature selection, not DR)

PLS theory:
  Finds latent components T = XW that maximise cov(T, y).
  Each component is a linear combination of all features weighted by
  how much they co-vary with HB — directly implements Beer-Lambert
  spectral mixture analysis.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RES_DIR = Path(__file__).parent / "results"

HB_MIN, HB_MAX = 3.0, 20.0

# ── helpers ────────────────────────────────────────────────────────────────

def reg_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    w1   = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    w2   = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
    return mae, rmse, r2, w1, w2


def anemia_auc(y_hb, y_pred_hb, threshold=12.0):
    y_bin = (y_hb < threshold).astype(int)
    try:
        auc = roc_auc_score(y_bin, -y_pred_hb)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    tn, fp, fn, tp = confusion_matrix(
        y_bin, (y_pred_hb < threshold).astype(int), labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-9)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return auc, sens, f1


def preprocess(X_tr, X_te):
    """Impute + standardise. Returns arrays."""
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    scl  = StandardScaler()
    X_tr = scl.fit_transform(X_tr)
    X_te = scl.transform(X_te)
    return X_tr, X_te


# ── cross-validation wrapper ───────────────────────────────────────────────

def cv_eval(name, build_fn, X, y, groups, X_test, y_test,
            n_splits=5, verbose=True):
    """
    Run 5-fold StratifiedGroupKFold, collect OOF predictions,
    then retrain on full data and evaluate on hold-out test.
    build_fn(X_tr, y_tr, X_va) → (pred_va, pipeline_object_for_test)
    """
    y_bin   = (y < 12).astype(int)
    cv      = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof     = np.full(len(y), np.nan)
    fold_maes = []

    for tr_idx, va_idx in cv.split(X, y_bin, groups):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        pred_va, _ = build_fn(X_tr, y_tr, X_va)
        oof[va_idx] = pred_va
        fold_maes.append(mean_absolute_error(y_va, pred_va))

    ok = ~np.isnan(oof)
    mae_oof, rmse_oof, r2_oof, w1_oof, w2_oof = reg_metrics(y[ok], oof[ok])
    auc_oof, sens_oof, f1_oof = anemia_auc(y[ok], oof[ok])

    # retrain on full training set → evaluate on test
    pred_test, _ = build_fn(X, y, X_test)
    mae_te, rmse_te, r2_te, w1_te, w2_te = reg_metrics(y_test, pred_test)
    auc_te, sens_te, f1_te = anemia_auc(y_test, pred_test)

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  {name}")
        print(f"  {'─'*60}")
        print(f"  OOF : MAE={mae_oof:.3f}  RMSE={rmse_oof:.3f}  R²={r2_oof:.3f}  "
              f"W±1={w1_oof:.1f}%  W±2={w2_oof:.1f}%")
        print(f"       AUC={auc_oof:.3f}  Sens={sens_oof:.3f}  F1={f1_oof:.3f}")
        print(f"  Test: MAE={mae_te:.3f}  RMSE={rmse_te:.3f}  R²={r2_te:.3f}  "
              f"W±1={w1_te:.1f}%  W±2={w2_te:.1f}%")
        print(f"       AUC={auc_te:.3f}  Sens={sens_te:.3f}  F1={f1_te:.3f}")
        print(f"       PredStd={pred_test.std():.3f}  TrueStd={y_test.std():.3f}  "
              f"Ratio={pred_test.std()/y_test.std():.3f}")

    return dict(name=name,
                oof_mae=mae_oof, oof_rmse=rmse_oof, oof_w2=w2_oof,
                oof_auc=auc_oof, oof_sens=sens_oof,
                test_mae=mae_te, test_rmse=rmse_te, test_w1=w1_te, test_w2=w2_te,
                test_auc=auc_te, test_sens=sens_te, test_f1=f1_te,
                test_pred_std=pred_test.std())


# ── method builders ────────────────────────────────────────────────────────

def make_pls(n_components):
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)
        pls = PLSRegression(n_components=n_components, max_iter=500)
        pls.fit(Xp_tr, y_tr)
        pred = np.clip(pls.predict(Xp_va).ravel(), HB_MIN, HB_MAX)
        return pred, pls
    return build


def make_pca_rf(n_components, n_trees=500):
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)
        pca = PCA(n_components=n_components, random_state=42)
        Xc_tr = pca.fit_transform(Xp_tr)
        Xc_va = pca.transform(Xp_va)
        rf = RandomForestRegressor(n_estimators=n_trees, max_depth=8,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(Xc_tr, y_tr)
        pred = np.clip(rf.predict(Xc_va), HB_MIN, HB_MAX)
        return pred, (pca, rf)
    return build


def make_pls_rf(n_pls, n_trees=500):
    """PLS for supervised compression, then RF on latent scores."""
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)
        pls = PLSRegression(n_components=n_pls, max_iter=500)
        pls.fit(Xp_tr, y_tr)
        Xc_tr = pls.transform(Xp_tr)
        Xc_va = pls.transform(Xp_va)
        rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(Xc_tr, y_tr)
        pred = np.clip(rf.predict(Xc_va), HB_MIN, HB_MAX)
        return pred, (pls, rf)
    return build


def make_pls_ensemble(n_pls):
    """PLS scores → RF + LightGBM + XGBoost ensemble."""
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)
        pls = PLSRegression(n_components=n_pls, max_iter=500)
        pls.fit(Xp_tr, y_tr)
        Xc_tr = pls.transform(Xp_tr)
        Xc_va = pls.transform(Xp_va)

        rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(Xc_tr, y_tr)

        lb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.5,
                                min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
                                random_state=42, verbose=-1)
        lb.fit(Xc_tr, y_tr)

        xb = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        xb.fit(Xc_tr, y_tr)

        pred = np.clip((rf.predict(Xc_va) + lb.predict(Xc_va) + xb.predict(Xc_va)) / 3,
                       HB_MIN, HB_MAX)
        return pred, (pls, rf, lb, xb)
    return build


def make_selectk_ensemble(k=80):
    """Current best baseline: SelectKBest → Ensemble."""
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)
        sel = SelectKBest(mutual_info_regression, k=min(k, Xp_tr.shape[1]))
        Xc_tr = sel.fit_transform(Xp_tr, y_tr)
        Xc_va = sel.transform(Xp_va)

        rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(Xc_tr, y_tr)

        lb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.5,
                                min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
                                random_state=42, verbose=-1)
        lb.fit(Xc_tr, y_tr)

        xb = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        xb.fit(Xc_tr, y_tr)

        pred = np.clip((rf.predict(Xc_va) + lb.predict(Xc_va) + xb.predict(Xc_va)) / 3,
                       HB_MIN, HB_MAX)
        return pred, (sel, rf, lb, xb)
    return build


def make_pca_ensemble(n_components):
    """PCA → Ensemble."""
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)
        pca = PCA(n_components=n_components, random_state=42)
        Xc_tr = pca.fit_transform(Xp_tr)
        Xc_va = pca.transform(Xp_va)

        rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(Xc_tr, y_tr)

        lb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.5,
                                min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
                                random_state=42, verbose=-1)
        lb.fit(Xc_tr, y_tr)

        xb = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        xb.fit(Xc_tr, y_tr)

        pred = np.clip((rf.predict(Xc_va) + lb.predict(Xc_va) + xb.predict(Xc_va)) / 3,
                       HB_MIN, HB_MAX)
        return pred, (pca, rf, lb, xb)
    return build


def make_pls_pca_concat(n_pls, n_pca):
    """Concatenate PLS scores + PCA components for richer representation."""
    def build(X_tr, y_tr, X_va):
        Xp_tr, Xp_va = preprocess(X_tr, X_va)

        pls = PLSRegression(n_components=n_pls, max_iter=500)
        pls.fit(Xp_tr, y_tr)
        Xpls_tr = pls.transform(Xp_tr)
        Xpls_va = pls.transform(Xp_va)

        pca = PCA(n_components=n_pca, random_state=42)
        Xpca_tr = pca.fit_transform(Xp_tr)
        Xpca_va = pca.transform(Xp_va)

        Xc_tr = np.hstack([Xpls_tr, Xpca_tr])
        Xc_va = np.hstack([Xpls_va, Xpca_va])

        rf = RandomForestRegressor(n_estimators=500, max_depth=8,
                                   min_samples_leaf=5, random_state=42)
        rf.fit(Xc_tr, y_tr)

        lb = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.5,
                                min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
                                random_state=42, verbose=-1)
        lb.fit(Xc_tr, y_tr)

        xb = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        xb.fit(Xc_tr, y_tr)

        pred = np.clip((rf.predict(Xc_va) + lb.predict(Xc_va) + xb.predict(Xc_va)) / 3,
                       HB_MIN, HB_MAX)
        return pred, None
    return build


# ── PLS component sweep helper ─────────────────────────────────────────────

def pls_sweep(X, y, groups, n_splits=5):
    """Find optimal n_components for PLSRegression via CV."""
    y_bin = (y < 12).astype(int)
    cv    = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    components = [5, 10, 15, 20, 25, 30, 40, 50]
    results = []
    print("\n  PLS n_components sweep (OOF MAE):")
    print(f"  {'n_comp':>8}  {'MAE':>8}  {'W±2':>8}  {'AUC':>8}")
    for nc in components:
        oof = np.full(len(y), np.nan)
        for tr, va in cv.split(X, y_bin, groups):
            Xp_tr, Xp_va = preprocess(X[tr], X[va])
            pls = PLSRegression(n_components=nc, max_iter=500)
            pls.fit(Xp_tr, y[tr])
            oof[va] = np.clip(pls.predict(Xp_va).ravel(), HB_MIN, HB_MAX)
        ok  = ~np.isnan(oof)
        mae = mean_absolute_error(y[ok], oof[ok])
        w2  = np.mean(np.abs(y[ok] - oof[ok]) <= 2.0) * 100
        try: auc = roc_auc_score((y[ok]<12).astype(int), -oof[ok])
        except: auc = float("nan")
        print(f"  {nc:>8}  {mae:>8.3f}  {w2:>7.1f}%  {auc:>8.3f}")
        results.append((nc, mae))
    best_nc = min(results, key=lambda x: x[1])[0]
    print(f"\n  Best n_components = {best_nc}")
    return best_nc


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Dimensionality Reduction for NISHAD HB Prediction")
    print("=" * 65)
    print(f"\n  Problem: 1060 features, 386 samples (ratio 2.75×)")

    train_df = pd.read_csv(RES_DIR / "features_combined_train.csv")
    test_df  = pd.read_csv(RES_DIR / "features_combined_test.csv")
    META     = {"video_id", "hb_value", "split", "protocol", "anemia"}
    feat_cols = [c for c in train_df.columns if c not in META]

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["hb_value"].values.astype(np.float32)
    groups  = train_df["video_id"].values
    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df["hb_value"].values.astype(np.float32)

    # ── PLS sweep to find best n_components ───────────────────────────────
    best_nc = pls_sweep(X_train, y_train, groups)

    # ── PCA variance explained ────────────────────────────────────────────
    imp_full = SimpleImputer(strategy="median")
    X_imp    = imp_full.fit_transform(X_train)
    X_scl    = StandardScaler().fit_transform(X_imp)
    pca_full = PCA().fit(X_scl)
    n90 = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.90)) + 1
    n95 = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95)) + 1
    print(f"\n  PCA: {n90} components explain 90% variance, {n95} explain 95%")

    # ── compare all methods ───────────────────────────────────────────────
    all_results = []

    all_results.append(cv_eval(
        f"SelectKBest(k=80) + Ensemble  [baseline]",
        make_selectk_ensemble(k=80), X_train, y_train, groups, X_test, y_test))

    all_results.append(cv_eval(
        f"PLS Regression (n={best_nc})",
        make_pls(best_nc), X_train, y_train, groups, X_test, y_test))

    all_results.append(cv_eval(
        f"PLS(n={best_nc}) → RF",
        make_pls_rf(best_nc), X_train, y_train, groups, X_test, y_test))

    all_results.append(cv_eval(
        f"PLS(n={best_nc}) → Ensemble",
        make_pls_ensemble(best_nc), X_train, y_train, groups, X_test, y_test))

    all_results.append(cv_eval(
        f"PCA(n={n90}) → Ensemble  [90% var]",
        make_pca_ensemble(n90), X_train, y_train, groups, X_test, y_test))

    all_results.append(cv_eval(
        f"PCA(n={n95}) → Ensemble  [95% var]",
        make_pca_ensemble(n95), X_train, y_train, groups, X_test, y_test))

    all_results.append(cv_eval(
        f"PLS(n={best_nc}) + PCA(n={n90}) concat → Ensemble",
        make_pls_pca_concat(best_nc, n90), X_train, y_train, groups, X_test, y_test))

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n\n  {'═'*80}")
    print(f"  SUMMARY TABLE (test set, n=95)")
    print(f"  {'═'*80}")
    print(f"  {'Method':<42} {'MAE':>6} {'RMSE':>6} {'W±2':>6} {'AUC':>6} {'Sens':>6} {'F1':>6}")
    print(f"  {'─'*80}")
    for r in sorted(all_results, key=lambda x: x["test_mae"]):
        marker = " ←" if r == min(all_results, key=lambda x: x["test_mae"]) else ""
        print(f"  {r['name']:<42} {r['test_mae']:>6.3f} {r['test_rmse']:>6.3f} "
              f"{r['test_w2']:>5.1f}% {r['test_auc']:>6.3f} "
              f"{r['test_sens']:>6.3f} {r['test_f1']:>6.3f}{marker}")

    # save results
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(RES_DIR / "dimred_comparison.csv", index=False)
    print(f"\n  Saved → {RES_DIR / 'dimred_comparison.csv'}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
