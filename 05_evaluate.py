"""
Stage 5 — Evaluation & visualisation.

Reads saved predictions from Stage 4 and produces:
  1. Regression: Bland-Altman plot, scatter (pred vs true), error-by-HB-bucket
  2. Classification: ROC curve, confusion matrix, sensitivity/specificity table
  3. Clinical summary table printed to console and saved as JSON

Run:
    python 05_evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, ACCURACY_BANDS, HB_BUCKETS

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Bland-Altman ─────────────────────────────────────────────────────────────

def bland_altman_plot(y_true: np.ndarray, y_pred: np.ndarray,
                       out_path: Path) -> dict:
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    bias      = diff_vals.mean()
    sd        = diff_vals.std()
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.6, s=30, color="steelblue")
    ax.axhline(bias,      color="red",   linestyle="-",  linewidth=1.5,
               label=f"Bias = {bias:+.3f}")
    ax.axhline(loa_upper, color="orange", linestyle="--", linewidth=1.2,
               label=f"+1.96 SD = {loa_upper:+.3f}")
    ax.axhline(loa_lower, color="orange", linestyle="--", linewidth=1.2,
               label=f"−1.96 SD = {loa_lower:+.3f}")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Mean of predicted & true Hb (g/dL)")
    ax.set_ylabel("Predicted − True Hb (g/dL)")
    ax.set_title("Bland-Altman Plot")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return {"bias": float(bias), "sd": float(sd),
            "loa_upper": float(loa_upper), "loa_lower": float(loa_lower)}


# ── scatter ───────────────────────────────────────────────────────────────────

def scatter_plot(y_true: np.ndarray, y_pred: np.ndarray,
                  mae: float, out_path: Path) -> None:
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color="steelblue")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect")
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                    alpha=0.12, color="orange", label="±1 g/dL band")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Hb (g/dL)")
    ax.set_ylabel("Predicted Hb (g/dL)")
    ax.set_title(f"Predicted vs True Hb  (MAE={mae:.3f} g/dL)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── per-bucket error ──────────────────────────────────────────────────────────

def bucket_error(y_true: np.ndarray, y_pred: np.ndarray,
                  buckets: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for lo, hi in buckets:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        errs = np.abs(y_pred[mask] - y_true[mask])
        rows.append({
            "hb_range":     f"{lo}–{hi}",
            "n":            int(mask.sum()),
            "mae":          float(errs.mean()),
            "within_1_0":   float((errs <= 1.0).mean()),
        })
    return pd.DataFrame(rows)


# ── ROC ───────────────────────────────────────────────────────────────────────

def roc_plot(y_true: np.ndarray, y_prob: np.ndarray,
              auc: float, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Anemia Detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── confusion matrix plot ─────────────────────────────────────────────────────

def cm_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Anemic", "Anemic"])
    ax.set_yticklabels(["Non-Anemic", "Anemic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("STAGE 5 — Evaluation")
    print("=" * 60)

    # ── regression ────────────────────────────────────────────────────────────
    reg_pred_path = RESULTS_DIR / "regression_predictions.csv"
    if reg_pred_path.exists():
        reg = pd.read_csv(reg_pred_path)
        y_true = reg["hb_value"].values
        y_pred = reg["hb_pred"].values

        mae = float(np.abs(y_true - y_pred).mean())
        print(f"\n── Regression (n={len(reg)}) ──")
        print(f"  MAE:   {mae:.4f} g/dL")
        print(f"  RMSE:  {float(np.sqrt(np.mean((y_true-y_pred)**2))):.4f} g/dL")

        for band in ACCURACY_BANDS:
            pct = 100 * (np.abs(y_true - y_pred) <= band).mean()
            print(f"  Within {band:.1f} g/dL: {pct:.1f}%")

        ba = bland_altman_plot(y_true, y_pred, RESULTS_DIR / "05_bland_altman.png")
        print(f"\n  Bland-Altman: bias={ba['bias']:+.3f}  "
              f"LoA=[{ba['loa_lower']:+.3f}, {ba['loa_upper']:+.3f}]")
        scatter_plot(y_true, y_pred, mae, RESULTS_DIR / "05_scatter.png")

        bucket_df = bucket_error(y_true, y_pred, HB_BUCKETS)
        print("\n  ── Error by HB range ──")
        print(bucket_df.to_string(index=False))
        bucket_df.to_csv(RESULTS_DIR / "05_bucket_error.csv", index=False)

        with open(RESULTS_DIR / "regression_metrics.json") as f:
            rm = json.load(f)
        rm.update({"test_bland_altman": ba})
        with open(RESULTS_DIR / "regression_metrics.json", "w") as f:
            json.dump(rm, f, indent=2)
    else:
        print("  regression_predictions.csv not found — skipping regression eval")

    # ── classification ────────────────────────────────────────────────────────
    cls_pred_path = RESULTS_DIR / "classification_predictions.csv"
    if cls_pred_path.exists():
        cls = pd.read_csv(cls_pred_path)
        y_true_cls  = cls["anemia_label"].values
        y_prob      = cls["prob_anemic"].values
        y_pred_cls  = cls["pred_anemic"].values

        auc     = roc_auc_score(y_true_cls, y_prob)
        bal_acc = balanced_accuracy_score(y_true_cls, y_pred_cls)
        cm      = confusion_matrix(y_true_cls, y_pred_cls)

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        print(f"\n── Classification (n={len(cls)}) ──")
        print(f"  AUC:             {auc:.4f}")
        print(f"  Balanced Acc:    {bal_acc:.4f}")
        print(f"  Sensitivity:     {sensitivity:.4f}  (anemia recall)")
        print(f"  Specificity:     {specificity:.4f}  (non-anemic recall)")
        print(f"\n{classification_report(y_true_cls, y_pred_cls, target_names=['Non-Anemic','Anemic'])}")

        roc_plot(y_true_cls, y_prob, auc, RESULTS_DIR / "05_roc.png")
        cm_plot(y_true_cls, y_pred_cls, RESULTS_DIR / "05_confusion_matrix.png")

        with open(RESULTS_DIR / "classification_metrics.json") as f:
            cm_j = json.load(f)
        cm_j.update({
            "test_sensitivity": float(sensitivity),
            "test_specificity": float(specificity),
        })
        with open(RESULTS_DIR / "classification_metrics.json", "w") as f:
            json.dump(cm_j, f, indent=2)
    else:
        print("  classification_predictions.csv not found — skipping classification eval")

    # ── overall summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("CLINICAL SUMMARY")
    print("=" * 50)
    if reg_pred_path.exists():
        print(f"  HB regression MAE : {mae:.3f} g/dL  (target < 1.5 g/dL)")
    if cls_pred_path.exists():
        print(f"  Anemia AUC        : {auc:.3f}         (target > 0.70)")
        print(f"  Sensitivity       : {sensitivity:.3f}")
        print(f"  Specificity       : {specificity:.3f}")

    print(f"\n  Plots saved → {RESULTS_DIR}")
    print("\nStage 5 complete.\n")


if __name__ == "__main__":
    main()
