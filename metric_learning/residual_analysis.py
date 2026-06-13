"""
Residual analysis on the 6s-only baseline predictions.

Plots:
  1. True vs Predicted scatter (identity line)
  2. Residuals vs Fitted values
  3. Residuals vs True HB
  4. Residual distribution (histogram + Q-Q)
  5. Bland-Altman (mean vs difference)
  6. Absolute error boxplot by HB range
  7. Cumulative accuracy curve (% within X g/dL)
  8. Residual heatmap / error by patient percentile

Run:
    python residual_analysis.py
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_DIR  = Path(__file__).parent / "results"
PRED_CSV = OUT_DIR / "predictions_baseline.csv"

HB_BINS  = [(0, 7), (7, 10), (10, 12), (12, 25)]
BIN_LABELS = ["<7\n(severe)", "7–10\n(moderate)", "10–12\n(mild)", ">12\n(normal)"]


def load():
    df = pd.read_csv(PRED_CSV)
    # error column is pred - true; residual = same convention
    df["abs_error"] = df["error"].abs()
    df["hb_bin"] = pd.cut(
        df["hb_value"],
        bins=[0, 7, 10, 12, 25],
        labels=BIN_LABELS,
        right=False,
    )
    return df


def summary_stats(df):
    true = df["hb_value"].values
    pred = df["hb_pred"].values
    err  = df["error"].values

    mae   = mean_absolute_error(true, pred)
    rmse  = np.sqrt(mean_squared_error(true, pred))
    r2    = r2_score(true, pred)
    bias  = err.mean()
    loa   = 1.96 * err.std()
    w1    = (df["abs_error"] <= 1.0).mean()
    w2    = (df["abs_error"] <= 2.0).mean()
    corr, pval = sp_stats.pearsonr(true, pred)

    print("=" * 50)
    print("RESIDUAL ANALYSIS — 6s-only baseline")
    print("=" * 50)
    print(f"  N test samples  : {len(df)}")
    print(f"  MAE             : {mae:.4f} g/dL")
    print(f"  RMSE            : {rmse:.4f} g/dL")
    print(f"  R²              : {r2:.4f}")
    print(f"  Pearson r       : {corr:.4f}  (p={pval:.4f})")
    print(f"  Bias (mean err) : {bias:+.4f} g/dL")
    print(f"  95% LoA         : {bias-loa:.3f} to {bias+loa:.3f}")
    print(f"  Within ±1 g/dL  : {100*w1:.1f}%")
    print(f"  Within ±2 g/dL  : {100*w2:.1f}%")
    print(f"  Pred range      : {pred.min():.2f} – {pred.max():.2f}")
    print(f"  True range      : {true.min():.2f} – {true.max():.2f}")

    print("\n  Error by HB range:")
    for (lo, hi), lbl in zip(HB_BINS, BIN_LABELS):
        sub = df[(df["hb_value"] >= lo) & (df["hb_value"] < hi)]
        if len(sub) == 0:
            continue
        lbl_clean = lbl.replace("\n", " ")
        print(f"    {lbl_clean:<22s}  n={len(sub):3d}  "
              f"MAE={sub['abs_error'].mean():.3f}  "
              f"bias={sub['error'].mean():+.3f}  "
              f"within1={100*(sub['abs_error']<=1).mean():.0f}%")

    _, p_sw = sp_stats.shapiro(err)
    print(f"\n  Shapiro-Wilk normality (residuals): p={p_sw:.4f} "
          f"({'normal' if p_sw > 0.05 else 'non-normal'})")

    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "loa": loa,
            "corr": corr, "w1": w1, "w2": w2}


def plot_all(df, stats):
    true = df["hb_value"].values
    pred = df["hb_pred"].values
    err  = df["error"].values

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── 1. True vs Predicted ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    lim = (min(true.min(), pred.min()) - 0.5,
           max(true.max(), pred.max()) + 0.5)
    ax1.scatter(true, pred, alpha=0.7, s=30, color="steelblue", edgecolors="white", lw=0.4)
    ax1.plot(lim, lim, "k--", lw=1, label="Identity")
    ax1.plot(lim, [l + 1 for l in lim], color="orange", lw=0.8, linestyle=":", label="±1 g/dL")
    ax1.plot(lim, [l - 1 for l in lim], color="orange", lw=0.8, linestyle=":")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("True HB (g/dL)"); ax1.set_ylabel("Predicted HB (g/dL)")
    ax1.set_title(f"True vs Predicted\nr={stats['corr']:.3f}  MAE={stats['mae']:.3f}", fontsize=10)
    ax1.legend(fontsize=8)

    # ── 2. Residuals vs Fitted ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(pred, err, alpha=0.7, s=30, color="tomato", edgecolors="white", lw=0.4)
    ax2.axhline(0, color="black", lw=1)
    ax2.axhline(stats["bias"] + stats["loa"], color="orange", lw=1, linestyle="--",
                label=f"+LoA={stats['bias']+stats['loa']:+.2f}")
    ax2.axhline(stats["bias"] - stats["loa"], color="orange", lw=1, linestyle="--",
                label=f"−LoA={stats['bias']-stats['loa']:+.2f}")
    ax2.axhline(stats["bias"], color="red", lw=1.5, linestyle="-",
                label=f"Bias={stats['bias']:+.3f}")
    # lowess smoother
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(err, pred, frac=0.4)
        ax2.plot(sm[:, 0], sm[:, 1], "darkgreen", lw=1.5, label="LOWESS")
    except ImportError:
        pass
    ax2.set_xlabel("Fitted (Predicted HB)"); ax2.set_ylabel("Residual (pred − true)")
    ax2.set_title("Residuals vs Fitted\n(heteroscedasticity check)", fontsize=10)
    ax2.legend(fontsize=7)

    # ── 3. Residuals vs True HB ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(true, err, alpha=0.7, s=30, color="mediumpurple",
                edgecolors="white", lw=0.4)
    ax3.axhline(0, color="black", lw=1)
    ax3.axhline(stats["bias"], color="red", lw=1.5, linestyle="-",
                label=f"Bias={stats['bias']:+.3f}")
    for (lo, hi) in HB_BINS[:-1]:
        ax3.axvline(hi, color="gray", lw=0.6, linestyle=":")
    ax3.set_xlabel("True HB (g/dL)"); ax3.set_ylabel("Residual (pred − true)")
    ax3.set_title("Residuals vs True HB\n(range-dependent bias check)", fontsize=10)
    ax3.legend(fontsize=8)

    # ── 4. Residual histogram ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(err, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    x_norm = np.linspace(err.min(), err.max(), 200)
    ax4.plot(x_norm,
             sp_stats.norm.pdf(x_norm, err.mean(), err.std()) * len(err) * (err.max()-err.min()) / 15,
             "red", lw=1.5, label="Normal fit")
    ax4.axvline(0, color="black", lw=1, linestyle="--")
    ax4.axvline(stats["bias"], color="red", lw=1.2, label=f"Bias={stats['bias']:+.3f}")
    ax4.set_xlabel("Residual (g/dL)"); ax4.set_ylabel("Count")
    ax4.set_title("Residual Distribution", fontsize=10)
    ax4.legend(fontsize=8)

    # ── 5. Q-Q plot ───────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sp_stats.probplot(err, dist="norm", plot=ax5)
    ax5.set_title("Q-Q Plot (normality of residuals)", fontsize=10)
    ax5.get_lines()[0].set(markersize=4, alpha=0.7, color="steelblue")
    ax5.get_lines()[1].set(color="red", lw=1.2)

    # ── 6. Bland-Altman ───────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    mean_val = (true + pred) / 2
    ax6.scatter(mean_val, err, alpha=0.7, s=30, color="teal",
                edgecolors="white", lw=0.4)
    ax6.axhline(stats["bias"], color="red", lw=1.5,
                label=f"Bias={stats['bias']:+.3f}")
    ax6.axhline(stats["bias"] + stats["loa"], color="orange", lw=1.2,
                linestyle="--", label=f"+LoA={stats['bias']+stats['loa']:+.2f}")
    ax6.axhline(stats["bias"] - stats["loa"], color="orange", lw=1.2,
                linestyle="--", label=f"−LoA={stats['bias']-stats['loa']:+.2f}")
    ax6.axhline(0, color="gray", lw=0.6, linestyle=":")
    ax6.set_xlabel("Mean of True & Predicted (g/dL)")
    ax6.set_ylabel("Pred − True (g/dL)")
    ax6.set_title("Bland-Altman Plot", fontsize=10)
    ax6.legend(fontsize=7)

    # ── 7. Abs error boxplot by HB range ─────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    groups = [df[df["hb_bin"] == lbl]["abs_error"].dropna().values
              for lbl in BIN_LABELS]
    bp = ax7.boxplot(groups, labels=BIN_LABELS, patch_artist=True,
                     medianprops={"color": "red", "lw": 1.5})
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax7.axhline(1.0, color="gray", lw=0.8, linestyle="--", label="1 g/dL")
    ax7.axhline(2.0, color="gray", lw=0.8, linestyle=":",  label="2 g/dL")
    ax7.set_xlabel("HB Range"); ax7.set_ylabel("|Error| (g/dL)")
    ax7.set_title("Abs Error by HB Range", fontsize=10)
    ax7.legend(fontsize=8)
    for i, g in enumerate(groups):
        ax7.annotate(f"n={len(g)}", (i+1, -0.3), ha="center", fontsize=8, color="gray")

    # ── 8. Cumulative accuracy ────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    thresholds = np.linspace(0, 5, 200)
    cum = [(df["abs_error"] <= t).mean() for t in thresholds]
    ax8.plot(thresholds, [100*c for c in cum], color="steelblue", lw=2)
    for t, color in [(0.5, "red"), (1.0, "orange"), (2.0, "green")]:
        pct = 100 * (df["abs_error"] <= t).mean()
        ax8.axvline(t, color=color, lw=0.8, linestyle="--")
        ax8.annotate(f"{pct:.0f}%\n@±{t}",
                     (t + 0.05, pct - 8), fontsize=8, color=color)
    ax8.set_xlabel("Error Threshold (g/dL)"); ax8.set_ylabel("% Predictions within threshold")
    ax8.set_title("Cumulative Accuracy Curve", fontsize=10)
    ax8.set_xlim(0, 5); ax8.set_ylim(0, 105)
    ax8.grid(True, alpha=0.3)

    # ── 9. Error sorted by true HB ────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    df_sorted = df.sort_values("hb_value").reset_index(drop=True)
    ax9.bar(range(len(df_sorted)), df_sorted["error"],
            color=["tomato" if e > 0 else "steelblue" for e in df_sorted["error"]],
            alpha=0.7, width=0.8)
    ax9.axhline(0, color="black", lw=0.8)
    ax9.axhline(1, color="orange", lw=0.6, linestyle="--")
    ax9.axhline(-1, color="orange", lw=0.6, linestyle="--")
    ax9.set_xlabel("Samples sorted by true HB (low→high)")
    ax9.set_ylabel("Residual (pred − true)")
    ax9.set_title("Signed Errors (sorted by true HB)\nred=over-predict  blue=under-predict",
                  fontsize=9)

    fig.suptitle("Residual Analysis — 6s-only LGBM Baseline\n"
                 f"MAE={stats['mae']:.3f}  RMSE={stats['rmse']:.3f}  "
                 f"R²={stats['r2']:.3f}  Bias={stats['bias']:+.3f}  "
                 f"within±1={100*stats['w1']:.0f}%  within±2={100*stats['w2']:.0f}%",
                 fontsize=11, y=1.01)

    out = OUT_DIR / "residual_analysis.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  saved → {out}")


def main():
    df    = load()
    stats = summary_stats(df)
    plot_all(df, stats)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
