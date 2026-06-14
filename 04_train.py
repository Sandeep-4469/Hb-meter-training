"""
Stage 4 — Model training.

Uses ALL videos (6s + 30s) since middle-frame extraction is protocol-agnostic.

Two tasks:
  A) Regression   — predict HB (g/dL)
  B) Classification — anemia detection (WHO threshold)

Skewness correction: training samples are weighted inversely proportional to
their HB bin count.  Bins are 1 g/dL wide (e.g. 8–9, 9–10 …).  This prevents
the model from just predicting the mean of the dense 8–12 region.

CV: GroupKFold(5) keyed on video_id — no data leakage across patients.

Run:
    python 04_train.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone as sklearn_clone
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression, QuantileRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
    CV_FOLDS, RANDOM_SEED, DEFAULT_ANEMIA_THRESHOLD,
    ACCURACY_BANDS,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

FEATURES_TRAIN_CSV = FEATURES_DIR / "features_train.csv"
FEATURES_TEST_CSV  = FEATURES_DIR / "features_test.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = {"video_id", "hb_value", "split", "protocol",
             "anemia_label", "sample_weight", "hb_bin",
             # identity / raw demographic columns — excluded from the feature
             # matrix (subject_id is used for grouping; raw sex/gender strings
             # are encoded into numeric demo_* columns below).
             "subject_id", "sex", "gender", "age"}

HB_BIN_WIDTH = 1.0   # g/dL


# ── demographics ──────────────────────────────────────────────────────────────

def add_demographic_features(df: pd.DataFrame) -> list[str]:
    """
    Add numeric demographic features in-place when source columns are present.

    Age and sex are strong, essentially-free predictors of haemoglobin: the
    ablation in Ni et al. (Front. Physiol. 2026, doi:10.3389/fphys.2026.1637455)
    reports that adding age+sex reduced error by ~14% (MRE 16.05% -> 2.20%).
    WHO anaemia thresholds are themselves sex/age-specific, so the model should
    have access to these variables.

    Returns the list of demographic feature names that were actually added
    (empty if the raw columns are unavailable).
    """
    added: list[str] = []
    if "age" in df.columns:
        df["demo_age"] = pd.to_numeric(df["age"], errors="coerce")
        added.append("demo_age")
    sex_col = "sex" if "sex" in df.columns else ("gender" if "gender" in df.columns else None)
    if sex_col is not None:
        # F -> 0, M -> 1 (any other / missing -> NaN, handled by the imputer)
        s = df[sex_col].astype(str).str.upper().str[0]
        df["demo_sex"] = s.map({"F": 0.0, "M": 1.0})
        added.append("demo_sex")
    return added


# ── helpers ───────────────────────────────────────────────────────────────────

def get_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def make_anemia_label(hb: float) -> int:
    return int(hb < DEFAULT_ANEMIA_THRESHOLD)


def bin_sample_weights(hb: np.ndarray, max_weight: float = 5.0) -> np.ndarray:
    """
    Inverse-frequency weights per HB bin (1 g/dL wide).
    Each sample gets weight = 1 / count_of_its_bin, normalised so mean weight = 1.
    Capped at max_weight to prevent extreme outlier bins from dominating.
    """
    bins   = np.floor(hb / HB_BIN_WIDTH).astype(int)
    counts = {b: (bins == b).sum() for b in np.unique(bins)}
    w = np.array([1.0 / counts[b] for b in bins], dtype=np.float64)
    w = w / w.mean()   # normalise: mean weight = 1
    w = np.clip(w, 0.0, max_weight)
    return w.astype(np.float32)


# ── model pipelines ───────────────────────────────────────────────────────────

def reg_pipelines(k: int) -> dict[str, Pipeline]:
    imp = lambda: SimpleImputer(strategy="median")
    sel = lambda: SelectKBest(mutual_info_regression, k=k)
    pipes = {
        "ridge": Pipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()), ("model", Ridge(alpha=50, random_state=RANDOM_SEED)),
        ]),
        "quantile": Pipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()), ("model", QuantileRegressor(quantile=0.5, alpha=0.01,
                                                         solver="highs")),
        ]),
        "svr": Pipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()), ("model", SVR(kernel="rbf", C=5.0,
                                          epsilon=0.5, gamma="scale")),
        ]),
    }
    if HAS_LGBM:
        pipes["lgbm"] = Pipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()), ("model", lgb.LGBMRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, objective="mae",
                random_state=RANDOM_SEED, verbose=-1,
            )),
        ])
    return pipes


def cls_pipelines(k: int) -> dict:
    imp = lambda: SimpleImputer(strategy="median")
    sel = lambda: SelectKBest(mutual_info_classif, k=k)
    pipes = {
        "logreg": ImbPipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()),
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("model", LogisticRegression(C=0.5, max_iter=2000,
                                          random_state=RANDOM_SEED)),
        ]),
        "svc": ImbPipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()),
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("model", SVC(kernel="rbf", C=2.0, probability=True,
                          random_state=RANDOM_SEED)),
        ]),
    }
    if HAS_LGBM:
        pipes["lgbm_clf"] = ImbPipeline([
            ("imp", imp()), ("scaler", StandardScaler()),
            ("sel", sel()),
            ("smote", SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
            ("model", lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=15,
                random_state=RANDOM_SEED, verbose=-1,
            )),
        ])
    return pipes


# ── group-aware CV ────────────────────────────────────────────────────────────

def get_group_values(df: pd.DataFrame) -> np.ndarray:
    """
    Grouping key for GroupKFold.  Prefer subject_id so that all videos from the
    same patient stay in the same fold (no patient-level leakage).  Fall back to
    video_id only if subject_id is unavailable.
    """
    col = "subject_id" if "subject_id" in df.columns else "video_id"
    return df[col].values


def group_cv_mae(pipe, df: pd.DataFrame, feat_cols: list[str],
                 weights: np.ndarray) -> float:
    """GroupKFold CV — train with sample weights, validate MAE on held-out."""
    groups = get_group_values(df)
    cv     = GroupKFold(n_splits=CV_FOLDS)
    maes   = []
    for tr_idx, val_idx in cv.split(df, groups=groups):
        X_tr = df.iloc[tr_idx][feat_cols].values.astype(np.float32)
        y_tr = df.iloc[tr_idx]["hb_value"].values
        w_tr = weights[tr_idx]

        X_val = df.iloc[val_idx][feat_cols].values.astype(np.float32)
        y_val = df.iloc[val_idx]["hb_value"].values

        p = sklearn_clone(pipe)
        # pass sample_weight to the final estimator if supported
        try:
            fit_params = {f"model__sample_weight": w_tr}
            p.fit(X_tr, y_tr, **fit_params)
        except TypeError:
            p.fit(X_tr, y_tr)
        maes.append(mean_absolute_error(y_val, p.predict(X_val)))
    return float(np.mean(maes)) if maes else float("inf")


def group_cv_auc(pipe, df: pd.DataFrame, feat_cols: list[str]) -> float:
    """GroupKFold CV for AUC — SMOTE is inside pipeline so no weights needed."""
    groups = get_group_values(df)
    cv     = GroupKFold(n_splits=CV_FOLDS)
    aucs   = []
    for tr_idx, val_idx in cv.split(df, groups=groups):
        X_tr  = df.iloc[tr_idx][feat_cols].values.astype(np.float32)
        y_tr  = df.iloc[tr_idx]["anemia_label"].values
        X_val = df.iloc[val_idx][feat_cols].values.astype(np.float32)
        y_val = df.iloc[val_idx]["anemia_label"].values
        if len(np.unique(y_val)) < 2:
            continue
        p = sklearn_clone(pipe)
        p.fit(X_tr, y_tr)
        prob = p.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))
    return float(np.mean(aucs)) if aucs else 0.5



def threshold_accuracy(y_true, y_pred, bands) -> dict:
    err = np.abs(y_true - y_pred)
    return {f"within_{b}".replace(".", "_"): float((err <= b).mean())
            for b in bands}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("STAGE 4 — Model Training  (all videos, skew-corrected)")
    print("=" * 60)

    train_df = pd.read_csv(FEATURES_TRAIN_CSV)
    test_df  = pd.read_csv(FEATURES_TEST_CSV)

    # demographics → numeric features (age, sex) when available
    demo_train = add_demographic_features(train_df)
    demo_test  = add_demographic_features(test_df)
    demo_cols  = sorted(set(demo_train) & set(demo_test))

    feat_cols = get_feat_cols(train_df)

    # ── leakage guard: no subject may appear in both train and test ───────────
    if "subject_id" in train_df.columns and "subject_id" in test_df.columns:
        overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
        if overlap:
            raise ValueError(
                f"Patient-level leakage: {len(overlap)} subject(s) appear in BOTH "
                f"train and test splits, e.g. {sorted(overlap)[:5]}. "
                f"Re-split by subject_id before training."
            )
        print(f"  Leakage check OK — {train_df['subject_id'].nunique()} train / "
              f"{test_df['subject_id'].nunique()} test subjects, disjoint.")
        print(f"  CV grouping key: subject_id")
    else:
        print("  WARNING: subject_id column absent — CV groups on video_id only "
              "(cannot guarantee patient-level leakage protection).")

    train_df["anemia_label"] = train_df["hb_value"].apply(make_anemia_label)
    test_df["anemia_label"]  = test_df["hb_value"].apply(make_anemia_label)

    print(f"\nFeatures     : {len(feat_cols)}"
          + (f"  (incl. demographics: {demo_cols})" if demo_cols else "  (no demographics found)"))
    print(f"Train videos : {len(train_df)}  "
          f"(6s={len(train_df[train_df.protocol=='6s'])}  "
          f"30s={len(train_df[train_df.protocol=='30s'])})")
    print(f"Test  videos : {len(test_df)}  "
          f"(6s={len(test_df[test_df.protocol=='6s'])}  "
          f"30s={len(test_df[test_df.protocol=='30s'])})")

    # sample weights for skewness correction
    weights = bin_sample_weights(train_df["hb_value"].values)
    print(f"\nSample weight range: {weights.min():.3f} – {weights.max():.3f}  "
          f"(mean={weights.mean():.3f})")

    # feature selection k: fixed at 40 for better coverage
    k = 40
    print(f"Feature selection k = {k}  "
          f"(CV: GroupKFold({CV_FOLDS}) by video_id)")

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["hb_value"].values
    y_train_cls = train_df["anemia_label"].values

    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df["hb_value"].values
    y_test_cls = test_df["anemia_label"].values

    # ── REGRESSION ────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("TASK A — Regression (predict HB g/dL)")
    print("─" * 50)

    r_pipes = reg_pipelines(k)
    r_cv    = {}
    for name, pipe in r_pipes.items():
        mae = group_cv_mae(pipe, train_df, feat_cols, weights)
        r_cv[name] = mae
        print(f"  {name:<15s}  CV MAE = {mae:.4f} g/dL")

    best_reg_name = min(r_cv, key=r_cv.get)
    print(f"\n  Best: {best_reg_name}  (CV MAE={r_cv[best_reg_name]:.4f})")

    best_reg = reg_pipelines(k)[best_reg_name]
    try:
        best_reg.fit(X_train, y_train,
                     **{f"model__sample_weight": weights})
    except TypeError:
        best_reg.fit(X_train, y_train)

    y_pred = best_reg.predict(X_test)
    print(f"\n  Pred range: {y_pred.min():.2f} – {y_pred.max():.2f}  "
          f"(true: {y_test.min():.2f} – {y_test.max():.2f})")

    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2  = r2_score(y_test, y_pred)
    test_rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    test_mape = float(np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-6, None))) * 100)
    bands    = threshold_accuracy(y_test, y_pred, ACCURACY_BANDS)

    # Mean-predictor baseline: MAE you get by always predicting the TRAIN mean.
    # If the model cannot beat this, it has learned nothing about haemoglobin.
    baseline_pred = float(np.mean(y_train))
    baseline_mae  = float(np.mean(np.abs(y_test - baseline_pred)))
    skill         = baseline_mae - test_mae   # >0 means the model adds value

    print(f"\n  ── Test set ──")
    print(f"  MAE   = {test_mae:.4f} g/dL")
    print(f"  RMSE  = {test_rmse:.4f} g/dL")
    print(f"  MAPE  = {test_mape:.2f} %")
    print(f"  R²    = {test_r2:.4f}")
    print(f"  Baseline MAE (predict train mean {baseline_pred:.2f}) = {baseline_mae:.4f} g/dL")
    print(f"  Skill vs baseline = {skill:+.4f} g/dL  "
          f"({'MODEL ADDS VALUE' if skill > 0 else 'NO BETTER THAN MEAN — model not learning'})")
    for k_name, v in bands.items():
        print(f"  {k_name:<20s} = {100*v:.1f}%")

    joblib.dump(best_reg, MODELS_DIR / "regressor.joblib")
    reg_metrics = {"model": best_reg_name, "cv_mae": r_cv,
                   "test_mae": test_mae, "test_rmse": test_rmse,
                   "test_mape": test_mape, "test_r2": test_r2,
                   "baseline_mae": baseline_mae, "skill_vs_baseline": skill,
                   **bands}
    with open(RESULTS_DIR / "regression_metrics.json", "w") as f:
        json.dump(reg_metrics, f, indent=2)

    preds = test_df[["video_id", "hb_value", "protocol"]].copy()
    preds["hb_pred"] = y_pred
    preds["error"]   = y_pred - y_test
    preds.to_csv(RESULTS_DIR / "regression_predictions.csv", index=False)

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("TASK B — Classification (anemia detection)")
    print("─" * 50)

    c_pipes = cls_pipelines(k)
    c_cv    = {}
    for name, pipe in c_pipes.items():
        auc = group_cv_auc(pipe, train_df, feat_cols)
        c_cv[name] = auc
        print(f"  {name:<15s}  CV AUC = {auc:.4f}")

    best_cls_name = max(c_cv, key=c_cv.get)
    print(f"\n  Best: {best_cls_name}  (CV AUC={c_cv[best_cls_name]:.4f})")

    best_cls = cls_pipelines(k)[best_cls_name]
    best_cls.fit(X_train, y_train_cls)

    y_prob     = best_cls.predict_proba(X_test)[:, 1]
    y_pred_cls = (y_prob >= 0.5).astype(int)
    test_auc   = roc_auc_score(y_test_cls, y_prob)
    test_balacc= balanced_accuracy_score(y_test_cls, y_pred_cls)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_test_cls, y_pred_cls):
        cm[t][p] += 1
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"\n  ── Test set ──")
    print(f"  AUC           = {test_auc:.4f}")
    print(f"  Balanced Acc  = {test_balacc:.4f}")
    print(f"  Sensitivity   = {sensitivity:.4f}  (anemia recall)")
    print(f"  Specificity   = {specificity:.4f}  (non-anemic recall)")
    print(f"\n{classification_report(y_test_cls, y_pred_cls, target_names=['Non-Anemic','Anemic'])}")

    joblib.dump(best_cls, MODELS_DIR / "classifier.joblib")
    cls_metrics = {"model": best_cls_name, "cv_auc": c_cv,
                   "test_auc": test_auc, "test_bal_acc": test_balacc,
                   "test_sensitivity": float(sensitivity),
                   "test_specificity": float(specificity)}
    with open(RESULTS_DIR / "classification_metrics.json", "w") as f:
        json.dump(cls_metrics, f, indent=2)

    cls_preds = test_df[["video_id", "hb_value", "protocol", "anemia_label"]].copy()
    cls_preds["prob_anemic"] = y_prob
    cls_preds["pred_anemic"] = y_pred_cls
    cls_preds.to_csv(RESULTS_DIR / "classification_predictions.csv", index=False)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Regression    best={best_reg_name:<12s}  "
          f"test MAE={test_mae:.3f} g/dL  (prior best: 2.14)")
    print(f"  Classification best={best_cls_name:<12s}  "
          f"test AUC={test_auc:.3f}         (prior best: 0.646)")
    print(f"\n  Models → {MODELS_DIR}")
    print("\nStage 4 complete.\n")


if __name__ == "__main__":
    main()
