"""
Step 3 — Train and evaluate on histogram features.

Regression  (predict HB g/dL):
  Ridge, ElasticNet, SVR, KNN, Random Forest, Extra Trees,
  Gradient Boosting, LightGBM

Classification (anemia: HB < 12 g/dL):
  Logistic Regression, SVC, Random Forest, Extra Trees,
  Gradient Boosting, LightGBM

All models:
  - GroupKFold(5) by video_id  (no data leakage across patients)
  - Inverse-frequency sample weights  (correct HB distribution skew)
  - SelectKBest(k=40) by mutual information  (prepended for linear models)
  - Tree-based models: raw features (handle high-dim natively)

Outputs:
  results/regression_results.csv
  results/classification_results.csv
  results/predictions_best.csv
  results/model_comparison.png

Run:
    python train.py
"""

from __future__ import annotations
import warnings, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor,
                               RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              roc_auc_score, f1_score, balanced_accuracy_score,
                              classification_report)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import RANDOM_SEED, CV_FOLDS, ACCURACY_BANDS

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

OUT_DIR       = Path(__file__).parent / "results"
META          = {"video_id", "hb_value", "split", "protocol"}   # is_30s kept as feature
ANEMIA_THRESH = 12.0
HB_BINS       = [(0,7),(7,10),(10,12),(12,25)]
K_SELECT      = 40     # features selected for linear/distance models


# ── helpers ───────────────────────────────────────────────────────────────────

def feat_cols(df):
    return [c for c in df.columns if c not in META and c != "anemia"]


def bin_weights(hb, max_w=5.0):
    bins   = np.floor(hb).astype(int)
    counts = {b: (bins == b).sum() for b in np.unique(bins)}
    w = np.array([1.0 / counts[b] for b in bins])
    w = w / w.mean()
    return np.clip(w, 0, max_w).astype(np.float32)


def safe_imp(X):
    return SimpleImputer(strategy="median").fit_transform(X)


def within_bands(y_true, y_pred, bands=ACCURACY_BANDS):
    err = np.abs(y_true - y_pred)
    return {f"within_{str(b).replace('.','_')}": float((err <= b).mean())
            for b in bands}


def error_by_range(y_true, y_pred):
    rows = []
    for lo, hi in HB_BINS:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "range": f"{lo}–{hi}",
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "within_1": float((np.abs(y_true[mask]-y_pred[mask]) <= 1.0).mean()),
        })
    return pd.DataFrame(rows)


# ── model definitions ─────────────────────────────────────────────────────────

def reg_models(k):
    """Returns dict of name → sklearn Pipeline (regression)."""
    imp = lambda: SimpleImputer(strategy="median")
    scl = lambda: StandardScaler()
    sel = lambda: SelectKBest(mutual_info_regression, k=min(k, K_SELECT))

    models = {
        "Ridge": Pipeline([
            ("imp", imp()), ("scl", scl()), ("sel", sel()),
            ("mdl", Ridge(alpha=50, random_state=RANDOM_SEED)),
        ]),
        "ElasticNet": Pipeline([
            ("imp", imp()), ("scl", scl()), ("sel", sel()),
            ("mdl", ElasticNet(alpha=0.1, l1_ratio=0.5,
                               max_iter=5000, random_state=RANDOM_SEED)),
        ]),
        "SVR": Pipeline([
            ("imp", imp()), ("scl", scl()), ("sel", sel()),
            ("mdl", SVR(kernel="rbf", C=5.0, epsilon=0.3, gamma="scale")),
        ]),
        "KNN": Pipeline([
            ("imp", imp()), ("scl", scl()), ("sel", sel()),
            ("mdl", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
        "RandomForest": Pipeline([
            ("imp", imp()),
            ("mdl", RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imp", imp()),
            ("mdl", ExtraTreesRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("imp", imp()),
            ("mdl", GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5,
                random_state=RANDOM_SEED)),
        ]),
    }
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("imp", imp()),
            ("mdl", lgb.LGBMRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, num_leaves=31,
                reg_alpha=0.1, reg_lambda=0.1,
                objective="mae", random_state=RANDOM_SEED, verbose=-1)),
        ])
    return models


def cls_models(k):
    """Returns dict of name → sklearn Pipeline (classification)."""
    imp = lambda: SimpleImputer(strategy="median")
    scl = lambda: StandardScaler()
    sel = lambda: SelectKBest(mutual_info_classif, k=min(k, K_SELECT))

    models = {
        "LogisticReg": Pipeline([
            ("imp", imp()), ("scl", scl()), ("sel", sel()),
            ("mdl", LogisticRegression(C=0.5, max_iter=2000,
                                       class_weight="balanced",
                                       random_state=RANDOM_SEED)),
        ]),
        "SVC": Pipeline([
            ("imp", imp()), ("scl", scl()), ("sel", sel()),
            ("mdl", SVC(kernel="rbf", C=2.0, probability=True,
                        class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "RandomForest": Pipeline([
            ("imp", imp()),
            ("mdl", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imp", imp()),
            ("mdl", ExtraTreesClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("imp", imp()),
            ("mdl", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=RANDOM_SEED)),
        ]),
    }
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("imp", imp()),
            ("mdl", lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, num_leaves=31,
                class_weight="balanced",
                random_state=RANDOM_SEED, verbose=-1)),
        ])
    return models


# ── cross-validation ──────────────────────────────────────────────────────────

def cv_regression(models, X, y, groups, weights):
    """GroupKFold CV → returns {model_name: cv_mae}."""
    results = {}
    for name, pipe in models.items():
        maes = []
        for tr, val in GroupKFold(CV_FOLDS).split(X, groups=groups):
            p = clone(pipe)
            try:
                p.fit(X[tr], y[tr], mdl__sample_weight=weights[tr])
            except TypeError:
                p.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[val], p.predict(X[val])))
        results[name] = float(np.mean(maes))
    return results


def cv_classification(models, X, y, groups):
    """GroupKFold CV → returns {model_name: cv_auc}."""
    results = {}
    for name, pipe in models.items():
        aucs = []
        for tr, val in GroupKFold(CV_FOLDS).split(X, groups=groups):
            if len(np.unique(y[val])) < 2:
                continue
            p = clone(pipe)
            p.fit(X[tr], y[tr])
            prob = p.predict_proba(X[val])[:, 1]
            aucs.append(roc_auc_score(y[val], prob))
        results[name] = float(np.mean(aucs)) if aucs else 0.5
    return results


# ── full evaluation ───────────────────────────────────────────────────────────

def evaluate_regression(pipe, X_train, y_train, X_test, y_test, weights):
    try:
        pipe.fit(X_train, y_train, mdl__sample_weight=weights)
    except TypeError:
        pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    bands = within_bands(y_test, y_pred)
    # anemia AUC from regression score
    y_an = (y_test < ANEMIA_THRESH).astype(int)
    auc  = roc_auc_score(y_an, -y_pred) if len(np.unique(y_an)) == 2 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "auc": auc, **bands}, y_pred


def evaluate_classification(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    prob   = pipe.predict_proba(X_test)[:, 1]
    y_pred = (prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, prob) if len(np.unique(y_test)) == 2 else float("nan")
    f1_an  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_non = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    balacc = balanced_accuracy_score(y_test, y_pred)
    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return {
        "auc": auc, "f1_macro": f1_mac,
        "f1_anemic": f1_an, "f1_normal": f1_non,
        "balanced_acc": balacc,
        "sensitivity": float(sens), "specificity": float(spec),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }, prob


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(reg_cv, reg_test, cls_cv, cls_test):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── regression CV MAE ────────────────────────────────────────────────────
    ax = axes[0, 0]
    names = list(reg_cv.keys())
    cv_vals  = [reg_cv[n]   for n in names]
    test_vals = [reg_test[n]["mae"] for n in names]
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.2, cv_vals,  0.35, label="CV MAE",   color="#3498db", alpha=0.8)
    b2 = ax.bar(x + 0.2, test_vals, 0.35, label="Test MAE", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAE (g/dL)"); ax.set_title("Regression — CV vs Test MAE", fontsize=11)
    ax.legend(); ax.axhline(2.0, color="green", lw=0.8, linestyle="--", label="2.0 target")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    # ── regression R² ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    r2_vals = [reg_test[n]["r2"] for n in names]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in r2_vals]
    bars = ax.bar(names, r2_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²"); ax.set_title("Regression — Test R²", fontsize=11)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (0.01 if v >= 0 else -0.03),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ── classification AUC ───────────────────────────────────────────────────
    ax = axes[1, 0]
    cnames = list(cls_cv.keys())
    cv_auc   = [cls_cv[n]          for n in cnames]
    test_auc = [cls_test[n]["auc"] for n in cnames]
    x = np.arange(len(cnames))
    b1 = ax.bar(x - 0.2, cv_auc,  0.35, label="CV AUC",   color="#9b59b6", alpha=0.8)
    b2 = ax.bar(x + 0.2, test_auc, 0.35, label="Test AUC", color="#f39c12", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cnames, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC"); ax.set_title("Classification (Anemia) — CV vs Test AUC", fontsize=11)
    ax.axhline(0.7, color="green", lw=0.8, linestyle="--")
    ax.set_ylim(0, 1.05); ax.legend()
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # ── within-band accuracy ──────────────────────────────────────────────────
    ax = axes[1, 1]
    band_keys = [k for k in list(reg_test.values())[0] if k.startswith("within_")]
    band_labels = [k.replace("within_", "±").replace("_", ".") + " g/dL"
                   for k in band_keys]
    x = np.arange(len(band_labels))
    width = 0.8 / len(names)
    colors_models = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for i, (name, color) in enumerate(zip(names, colors_models)):
        vals = [100 * reg_test[name][k] for k in band_keys]
        ax.bar(x + i * width - 0.4 + width/2, vals, width,
               label=name, color=color, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel("% predictions within threshold")
    ax.set_title("Within-band Accuracy by Model", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 100)

    plt.suptitle("ML Model Comparison — HB Meter (6s protocol, histogram features)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()


def plot_best_predictions(best_name, y_test, y_pred, test_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # scatter
    ax = axes[0]
    lim = (min(y_test.min(), y_pred.min()) - 0.5,
           max(y_test.max(), y_pred.max()) + 0.5)
    ax.scatter(y_test, y_pred, alpha=0.7, s=40, color="steelblue",
               edgecolors="white", lw=0.4)
    ax.plot(lim, lim, "k--", lw=1, label="Identity")
    ax.plot(lim, [l+1 for l in lim], "orange", lw=0.8, linestyle=":")
    ax.plot(lim, [l-1 for l in lim], "orange", lw=0.8, linestyle=":",
            label="±1 g/dL")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True HB (g/dL)"); ax.set_ylabel("Predicted HB (g/dL)")
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    ax.set_title(f"{best_name}\nMAE={mae:.3f}  R²={r2:.3f}", fontsize=10)
    ax.legend(fontsize=8)

    # bland-altman
    ax = axes[1]
    mean_val = (y_test + y_pred) / 2
    diff     = y_pred - y_test
    bias     = diff.mean()
    loa      = 1.96 * diff.std()
    ax.scatter(mean_val, diff, alpha=0.7, s=40, color="teal",
               edgecolors="white", lw=0.4)
    ax.axhline(bias,      color="red",    lw=1.5, label=f"Bias={bias:+.3f}")
    ax.axhline(bias+loa,  color="orange", lw=1.2, linestyle="--",
               label=f"+LoA={bias+loa:+.2f}")
    ax.axhline(bias-loa,  color="orange", lw=1.2, linestyle="--",
               label=f"−LoA={bias-loa:+.2f}")
    ax.axhline(0, color="gray", lw=0.6, linestyle=":")
    ax.set_xlabel("Mean of True & Predicted"); ax.set_ylabel("Pred − True")
    ax.set_title("Bland-Altman Plot", fontsize=10); ax.legend(fontsize=8)

    # error by range
    ax = axes[2]
    err_range = error_by_range(y_test, y_pred)
    x = np.arange(len(err_range))
    bars = ax.bar(err_range["range"], err_range["mae"],
                  color=["#e74c3c","#e67e22","#f1c40f","#2ecc71"], alpha=0.8)
    ax.axhline(1.0, color="gray", lw=1, linestyle="--")
    ax.axhline(2.0, color="gray", lw=1, linestyle=":")
    for bar, (_, row) in zip(bars, err_range.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"n={row['n']}\nw1={row['within_1']:.0%}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("HB range (g/dL)"); ax.set_ylabel("MAE (g/dL)")
    ax.set_title("MAE by HB Range", fontsize=10)

    plt.suptitle(f"Best Model: {best_name}", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "best_model_predictions.png", dpi=140, bbox_inches="tight")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("STEP 3 — Multi-model Training & Evaluation")
    print("=" * 65)

    train_df = pd.read_csv(OUT_DIR / "features_train.csv")
    test_df  = pd.read_csv(OUT_DIR / "features_test.csv")

    fcols = feat_cols(train_df)
    print(f"\nFeatures     : {len(fcols)}")
    print(f"Train videos : {len(train_df)}")
    print(f"Test  videos : {len(test_df)}")

    train_df["anemia"] = (train_df["hb_value"] < ANEMIA_THRESH).astype(int)
    test_df["anemia"]  = (test_df["hb_value"]  < ANEMIA_THRESH).astype(int)

    X_train = train_df[fcols].values.astype(np.float32)
    y_train = train_df["hb_value"].values
    y_train_cls = train_df["anemia"].values
    groups  = train_df["video_id"].values
    weights = bin_weights(y_train)

    X_test  = test_df[fcols].values.astype(np.float32)
    y_test  = test_df["hb_value"].values
    y_test_cls = test_df["anemia"].values

    n_feat = X_train.shape[1]
    print(f"Anemia split : train {y_train_cls.sum()}/{len(y_train_cls)} anemic"
          f"  test {y_test_cls.sum()}/{len(y_test_cls)} anemic")
    print(f"Weight range : {weights.min():.2f} – {weights.max():.2f}")
    print(f"SelectKBest k = {min(n_feat, K_SELECT)}")

    # ── REGRESSION ────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("TASK A — Regression (predict HB g/dL)")
    print("═" * 65)

    r_models = reg_models(n_feat)
    print("\nCross-validation (GroupKFold 5):")
    r_cv = cv_regression(r_models, X_train, y_train, groups, weights)
    for name, mae in sorted(r_cv.items(), key=lambda x: x[1]):
        print(f"  {name:<20s}  CV MAE = {mae:.4f}")

    print("\nTest set evaluation:")
    r_test   = {}
    r_preds  = {}
    header = f"  {'Model':<20s}  {'MAE':>6s}  {'RMSE':>6s}  {'R²':>7s}  " \
             f"{'w±1':>6s}  {'w±2':>6s}  {'AUC':>6s}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for name in sorted(r_cv, key=r_cv.get):
        pipe = clone(r_models[name])
        metrics, y_pred = evaluate_regression(
            pipe, X_train, y_train, X_test, y_test, weights)
        r_test[name]  = metrics
        r_preds[name] = y_pred
        print(f"  {name:<20s}  {metrics['mae']:6.3f}  {metrics['rmse']:6.3f}  "
              f"{metrics['r2']:7.3f}  "
              f"{100*metrics['within_1_0']:5.1f}%  "
              f"{100*metrics['within_2_0']:5.1f}%  "
              f"{metrics['auc']:6.3f}")

    best_reg = min(r_test, key=lambda n: r_test[n]["mae"])
    print(f"\n  ★  Best: {best_reg}  (Test MAE={r_test[best_reg]['mae']:.4f})")

    # per-range breakdown for best model
    print(f"\n  Error by HB range ({best_reg}):")
    err_df = error_by_range(y_test, r_preds[best_reg])
    print(err_df.to_string(index=False))

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("TASK B — Classification (anemia: HB < 12 g/dL)")
    print("═" * 65)

    c_models = cls_models(n_feat)
    print("\nCross-validation (GroupKFold 5):")
    c_cv = cv_classification(c_models, X_train, y_train_cls, groups)
    for name, auc in sorted(c_cv.items(), key=lambda x: -x[1]):
        print(f"  {name:<20s}  CV AUC = {auc:.4f}")

    print("\nTest set evaluation:")
    c_test  = {}
    c_probs = {}
    header2 = f"  {'Model':<20s}  {'AUC':>6s}  {'F1-mac':>7s}  " \
              f"{'F1-anem':>8s}  {'Sens':>6s}  {'Spec':>6s}  {'BalAcc':>7s}"
    print(header2)
    print("  " + "─" * (len(header2) - 2))
    for name in sorted(c_cv, key=lambda n: -c_cv[n]):
        pipe = clone(c_models[name])
        metrics, prob = evaluate_classification(
            pipe, X_train, y_train_cls, X_test, y_test_cls)
        c_test[name]  = metrics
        c_probs[name] = prob
        print(f"  {name:<20s}  {metrics['auc']:6.3f}  {metrics['f1_macro']:7.3f}  "
              f"{metrics['f1_anemic']:8.3f}  "
              f"{metrics['sensitivity']:6.3f}  {metrics['specificity']:6.3f}  "
              f"{metrics['balanced_acc']:7.3f}")

    best_cls = max(c_test, key=lambda n: c_test[n]["auc"])
    print(f"\n  ★  Best: {best_cls}  (Test AUC={c_test[best_cls]['auc']:.4f})")
    print(f"\n  Confusion matrix ({best_cls}):")
    m = c_test[best_cls]
    print(f"    {'':10s}  Pred-Anemic  Pred-Normal")
    print(f"    True-Anemic  {m['tp']:11d}  {m['fn']:11d}")
    print(f"    True-Normal  {m['fp']:11d}  {m['tn']:11d}")
    print(f"\n  Full classification report:")
    y_pred_cls = (c_probs[best_cls] >= 0.5).astype(int)
    print(classification_report(y_test_cls, y_pred_cls,
                                target_names=["Normal (≥12)", "Anemic (<12)"]))

    # ── save results ──────────────────────────────────────────────────────────
    reg_rows = []
    for name in r_test:
        row = {"model": name, "cv_mae": r_cv[name]}
        row.update(r_test[name])
        reg_rows.append(row)
    pd.DataFrame(reg_rows).sort_values("mae").to_csv(
        OUT_DIR / "regression_results.csv", index=False)

    cls_rows = []
    for name in c_test:
        row = {"model": name, "cv_auc": c_cv[name]}
        row.update(c_test[name])
        cls_rows.append(row)
    pd.DataFrame(cls_rows).sort_values("auc", ascending=False).to_csv(
        OUT_DIR / "classification_results.csv", index=False)

    preds_df = test_df[["video_id","hb_value","protocol","anemia"]].copy()
    preds_df[f"pred_{best_reg}"] = r_preds[best_reg]
    preds_df["error"] = r_preds[best_reg] - y_test
    preds_df[f"prob_anemia_{best_cls}"] = c_probs[best_cls]
    preds_df.to_csv(OUT_DIR / "predictions_best.csv", index=False)

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_comparison(r_cv, r_test, c_cv, c_test)
    print(f"\n  saved → {OUT_DIR / 'model_comparison.png'}")

    plot_best_predictions(best_reg, y_test, r_preds[best_reg], test_df)
    print(f"  saved → {OUT_DIR / 'best_model_predictions.png'}")

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("FINAL SUMMARY")
    print("═" * 65)
    print(f"\n  {'Model':<20s}  {'CV-MAE':>7s}  {'MAE':>6s}  {'R²':>7s}  "
          f"{'w±1':>6s}  {'w±2':>6s}  {'AUC-anemia':>11s}")
    print("  " + "─" * 72)
    for name in sorted(r_test, key=lambda n: r_test[n]["mae"]):
        m = r_test[name]
        print(f"  {name:<20s}  {r_cv[name]:7.3f}  {m['mae']:6.3f}  "
              f"{m['r2']:7.3f}  "
              f"{100*m['within_1_0']:5.1f}%  "
              f"{100*m['within_2_0']:5.1f}%  "
              f"{m['auc']:11.3f}")
    print(f"\n  Best regression    : {best_reg} — MAE {r_test[best_reg]['mae']:.3f} g/dL"
          f"  R² {r_test[best_reg]['r2']:.3f}")
    print(f"  Best classification: {best_cls} — AUC {c_test[best_cls]['auc']:.3f}"
          f"  Sens {c_test[best_cls]['sensitivity']:.3f}"
          f"  Spec {c_test[best_cls]['specificity']:.3f}")
    print("\nStep 3 complete.\n")


if __name__ == "__main__":
    main()
