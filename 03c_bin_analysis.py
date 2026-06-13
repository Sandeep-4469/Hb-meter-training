"""
Stage 3c — Bin-level discriminability analysis (time-series features).

Groups training patients into 1 g/dL HB bins and shows how each channel's
DC mean and AC amplitude shift across bins. Computes per-bin sample counts
and identifies the best individual features by one-way ANOVA F-score.

Run:
    python 03c_bin_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_DIR, RESULTS_DIR

FEATURES_CSV = FEATURES_DIR / "features_train.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS  = ["R", "G", "B", "H", "S", "V", "Gray"]
BIN_EDGES = np.arange(4, 18, 1)


def main() -> None:
    print("=" * 60)
    print("STAGE 3c — Bin Discriminability Analysis")
    print("=" * 60)

    df = pd.read_csv(FEATURES_CSV)
    df["hb_bin"] = pd.cut(df["hb_value"], bins=BIN_EDGES,
                           labels=[f"{int(e)}–{int(e)+1}" for e in BIN_EDGES[:-1]],
                           right=False)
    df = df.dropna(subset=["hb_bin"])

    counts = df["hb_bin"].value_counts().sort_index()
    print("\n── Training samples per HB bin ──")
    for bin_lbl, cnt in counts.items():
        bar = "█" * cnt
        print(f"  {bin_lbl:6s}: {bar} ({cnt})")

    bin_labels = counts.index.tolist()

    # ── DC mean per bin per channel ────────────────────────────────────────────
    print("\n── DC mean vs HB bin ──")
    fig, axes = plt.subplots(3, 7, figsize=(22, 10), sharey=False)
    fig.suptitle("DC mean per HB bin across LED segments", fontsize=12)

    for seg in range(3):
        for ci, ch in enumerate(CHANNELS):
            ax = axes[seg][ci]
            col = f"seg{seg}_{ch}_dc"
            if col not in df.columns:
                ax.axis("off"); continue
            means, stds = [], []
            for bin_lbl in bin_labels:
                sub = df[df["hb_bin"] == bin_lbl]
                means.append(sub[col].mean() if not sub.empty else np.nan)
                stds.append(sub[col].std()   if not sub.empty else 0)
            x = np.arange(len(bin_labels))
            ax.errorbar(x, means, yerr=stds, marker="o",
                        markersize=4, linewidth=1.5, capsize=3, color="steelblue")
            ax.set_title(f"seg{seg} {ch}", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=90, fontsize=6)
            if ci == 0:
                ax.set_ylabel(["RED", "ORANGE", "YELLOW"][seg], fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "03c_dc_mean_per_bin.png"
    plt.savefig(out, dpi=130); plt.close()
    print(f"  saved → {out}")

    # ── AC/DC ratio per bin ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 7, figsize=(22, 10), sharey=False)
    fig.suptitle("AC/DC ratio per HB bin across LED segments", fontsize=12)

    for seg in range(3):
        for ci, ch in enumerate(CHANNELS):
            ax = axes[seg][ci]
            col = f"seg{seg}_{ch}_ac_dc"
            if col not in df.columns:
                ax.axis("off"); continue
            means, stds = [], []
            for bin_lbl in bin_labels:
                sub = df[df["hb_bin"] == bin_lbl]
                means.append(sub[col].mean() if not sub.empty else np.nan)
                stds.append(sub[col].std()   if not sub.empty else 0)
            x = np.arange(len(bin_labels))
            ax.errorbar(x, means, yerr=stds, marker="o",
                        markersize=4, linewidth=1.5, capsize=3, color="darkorange")
            ax.set_title(f"seg{seg} {ch}", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=90, fontsize=6)
            if ci == 0:
                ax.set_ylabel(["RED", "ORANGE", "YELLOW"][seg], fontsize=8)

    plt.tight_layout()
    out2 = RESULTS_DIR / "03c_acdc_per_bin.png"
    plt.savefig(out2, dpi=130); plt.close()
    print(f"  saved → {out2}")

    # ── ANOVA F-score ──────────────────────────────────────────────────────────
    meta = {"video_id", "hb_value", "split", "protocol", "hb_bin",
            "anemia_label", "sample_weight"}
    feat_cols = [c for c in df.columns if c not in meta]
    groups_by_bin = {lbl: df[df["hb_bin"] == lbl] for lbl in bin_labels}

    f_scores = {}
    for col in feat_cols:
        group_vals = [grp[col].dropna().values for grp in groups_by_bin.values()
                      if len(grp[col].dropna()) >= 2]
        if len(group_vals) < 2:
            continue
        try:
            f, _ = sp_stats.f_oneway(*group_vals)
            if not np.isnan(f):
                f_scores[col] = f
        except Exception:
            pass

    top_feats = sorted(f_scores, key=f_scores.get, reverse=True)[:20]
    print("\n── Top 20 features by ANOVA F-score across HB bins ──")
    for feat in top_feats:
        print(f"  {feat:<55s}  F={f_scores[feat]:.2f}")

    fscore_df = pd.DataFrame({"feature": list(f_scores.keys()),
                               "f_score": list(f_scores.values())})
    fscore_df.sort_values("f_score", ascending=False).to_csv(
        RESULTS_DIR / "03c_anova_fscores.csv", index=False)

    # ── Inverse-frequency weights ──────────────────────────────────────────────
    bin_counts    = df["hb_bin"].value_counts()
    inv_freq      = 1.0 / bin_counts
    inv_freq_norm = inv_freq / inv_freq.sum()

    print("\n── Inverse-frequency sample weights ──")
    for lbl, w in inv_freq_norm.sort_index().items():
        print(f"  {lbl:6s}: weight={w:.4f}  (n={bin_counts[lbl]})")

    weight_map = inv_freq_norm.reset_index()
    weight_map.columns = ["hb_bin", "sample_weight"]
    weight_map.to_csv(RESULTS_DIR / "03c_bin_weights.csv", index=False)
    print(f"\n  bin weights → {RESULTS_DIR / '03c_bin_weights.csv'}")
    print("\nStage 3c complete.\n")


if __name__ == "__main__":
    main()
