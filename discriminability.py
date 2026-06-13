"""
Discriminability analysis — denoised, multi-frame average per segment.

Noise reduction:
  1. Temporal: average WINDOW_FRAMES frames centred on the segment midpoint
     (frames outside brightness bounds are rejected before averaging)
  2. Spatial: Gaussian blur on each frame's ROI before computing channel means

Channels extracted per segment (10):
  R, G, B  (RGB)
  H, S, V  (HSV)
  L, A, B  (Lab — perceptually uniform)
  Gray

Features analysed:
  A) Per-segment means (30 raw features)
  B) Cross-segment ratios seg_i / seg_j (pairs 01, 02, 12) — 30 ratio features
     These cancel patient-specific brightness and isolate wavelength absorption.

Outlier removal:
  Global IQR filter: for each feature, drop rows where value is outside
  [Q1 - 1.5*IQR, Q3 + 1.5*IQR].  A patient dropped by ANY feature is removed.

Outputs (all in data_analysis/):
  discrim_boxplots_raw_seg{n}.png     box plots, raw means
  discrim_boxplots_ratio_seg{n}.png   box plots, cross-segment ratios
  discrim_correlations.png            Pearson r bar chart
  discrim_anova.csv                   F-score + |r| for every feature

Run:
    python discriminability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ROI_Y, ROI_X

OUT_DIR = Path(__file__).parent / "data_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS   = ["R", "G", "B", "H", "S", "V", "L", "A", "B_lab", "Gray"]
SEG_NAMES  = ["RED (660nm)", "ORANGE (610nm)", "YELLOW (590nm)"]

WINDOW_HALF   = 15          # ±15 frames around midpoint → up to 31 frames averaged
MIN_BRIGHT    = 10.0        # reject frames darker than this (Gray mean)
MAX_BRIGHT    = 250.0       # reject frames brighter than this (saturated)
BLUR_KSIZE    = (7, 7)      # Gaussian blur kernel applied to ROI


# ── frame / ROI helpers ───────────────────────────────────────────────────────

def crop_roi(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    return bgr[int(h * ROI_Y[0]):int(h * ROI_Y[1]),
               int(w * ROI_X[0]):int(w * ROI_X[1])]


def frame_means(bgr: np.ndarray) -> np.ndarray:
    """Return 10-channel mean vector from a single (already-blurred) ROI."""
    roi  = cv2.GaussianBlur(crop_roi(bgr), BLUR_KSIZE, 0)
    rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab  = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.array([
        rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean(),
        hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean(),
        lab[:, :, 0].mean(), lab[:, :, 1].mean(), lab[:, :, 2].mean(),
        gray.mean(),
    ], dtype=np.float32)


def segment_mean_denoised(path: str, s_start: int, s_end: int) -> np.ndarray | None:
    """
    Average channel means over a window of frames around the segment midpoint.
    Frames outside brightness bounds are rejected before averaging.
    Returns a (10,) array or None if no usable frames found.
    """
    mid   = (s_start + s_end) // 2
    f_lo  = max(s_start, mid - WINDOW_HALF)
    f_hi  = min(s_end,   mid + WINDOW_HALF + 1)

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)

    stack = []
    for _ in range(f_hi - f_lo):
        ret, frame = cap.read()
        if not ret:
            break
        means = frame_means(frame)
        gray_mean = means[9]              # Gray is index 9
        if MIN_BRIGHT <= gray_mean <= MAX_BRIGHT:
            stack.append(means)
    cap.release()

    return np.stack(stack).mean(axis=0) if stack else None


# ── video extraction ──────────────────────────────────────────────────────────

def extract_all(index: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    df = index[index["file_ok"]].merge(
             segs[["video_id", "s0_start", "s0_end",
                   "s1_start", "s1_end", "s2_start", "s2_end"]],
             on="video_id", how="inner")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Denoised extraction"):
        rec = {"video_id": row["video_id"],
               "hb_value": row["hb_value"],
               "protocol": row["protocol"]}
        ok = True
        seg_means = []
        for seg in range(3):
            m = segment_mean_denoised(
                    row["video_path"],
                    int(row[f"s{seg}_start"]),
                    int(row[f"s{seg}_end"]))
            if m is None:
                ok = False; break
            for ci, ch in enumerate(CHANNELS):
                rec[f"seg{seg}_{ch}"] = float(m[ci])
            seg_means.append(m)

        if ok:
            # cross-segment ratios (gain-invariant)
            for (i, j) in [(0, 1), (0, 2), (1, 2)]:
                for ci, ch in enumerate(CHANNELS):
                    mi = seg_means[i][ci] + 1e-6
                    mj = seg_means[j][ci] + 1e-6
                    rec[f"r{i}{j}_{ch}"] = float(mi / mj)
            records.append(rec)

    return pd.DataFrame(records)


# ── outlier removal ───────────────────────────────────────────────────────────

def remove_outliers_iqr(df: pd.DataFrame, feat_cols: list[str],
                         k: float = 3.0, vote_thresh: float = 0.20) -> pd.DataFrame:
    """
    Per-bin IQR outlier removal.
    Within each HB bin, count how many features each patient falls outside
    [Q1 - k*IQR, Q3 + k*IQR] (computed on that bin alone).
    A patient is removed only if they are outliers in more than
    vote_thresh fraction of features — a single noisy feature doesn't discard them.
    """
    outlier_count = pd.Series(0, index=df.index, dtype=int)
    n_checked     = pd.Series(0, index=df.index, dtype=int)

    for bin_lbl, grp in df.groupby("hb_bin", observed=True):
        if len(grp) < 4:
            continue
        for col in feat_cols:
            q1, q3 = grp[col].quantile(0.25), grp[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            is_out = ~grp[col].between(q1 - k * iqr, q3 + k * iqr)
            outlier_count[grp.index] += is_out.astype(int)
            n_checked[grp.index]     += 1

    frac_out = outlier_count / n_checked.replace(0, 1)
    keep     = frac_out <= vote_thresh

    n_before = len(df)
    df_clean = df[keep].copy()
    print(f"  Per-bin outlier removal (k={k}, vote>{vote_thresh:.0%}): "
          f"{n_before} → {len(df_clean)} ({n_before - len(df_clean)} removed)")
    return df_clean


# ── discriminability stats ────────────────────────────────────────────────────

def discriminability(df: pd.DataFrame, feat_cols: list[str],
                     valid_bins: list[str]) -> pd.DataFrame:
    results = []
    for col in feat_cols:
        if col not in df.columns:
            continue
        groups = [df[df["hb_bin"] == b][col].dropna().values
                  for b in valid_bins]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            f, p = sp_stats.f_oneway(*groups)
        except Exception:
            f, p = np.nan, np.nan
        valid = df[col].notna()
        r, _ = sp_stats.pearsonr(df.loc[valid, col],
                                  df.loc[valid, "hb_value"])
        results.append({"feature": col, "f_score": f, "p_anova": p,
                         "pearson_r": r, "abs_r": abs(r)})
    return pd.DataFrame(results).sort_values("f_score", ascending=False)


# ── box plot helper ───────────────────────────────────────────────────────────

def boxplot_grid(df: pd.DataFrame, feat_cols: list[str], channels: list[str],
                 valid_bins: list[str], res_df: pd.DataFrame,
                 title: str, out_path: Path) -> None:
    cmap = matplotlib.colormaps["RdYlGn"].resampled(len(valid_bins))
    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    fig.suptitle(title, fontsize=11)

    for ax, col, ch in zip(axes.flat, feat_cols, channels):
        if col not in df.columns:
            ax.axis("off"); continue
        bin_data = [df[df["hb_bin"] == b][col].dropna().values
                    for b in valid_bins]
        bp = ax.boxplot(bin_data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=2, alpha=0.4),
                        whiskerprops=dict(linewidth=0.7),
                        capprops=dict(linewidth=0.7))
        for patch, i in zip(bp["boxes"], range(len(valid_bins))):
            patch.set_facecolor(cmap(i / max(len(valid_bins) - 1, 1)))
            patch.set_alpha(0.75)

        row_r = res_df[res_df["feature"] == col]
        if not row_r.empty:
            f = row_r["f_score"].values[0]
            r = row_r["pearson_r"].values[0]
            ax.set_title(f"{ch}   F={f:.2f}  r={r:+.3f}", fontsize=8)
        else:
            ax.set_title(ch, fontsize=8)

        ax.set_xticks(range(1, len(valid_bins) + 1))
        ax.set_xticklabels(valid_bins, rotation=90, fontsize=5)
        ax.set_xlabel("HB bin (g/dL)", fontsize=6)
        ax.set_ylabel("Value", fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"  saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("DISCRIMINABILITY — denoised frames + cross-segment ratios")
    print("=" * 60)

    index = pd.read_csv(DATA_DIR / "index.csv")
    segs  = pd.read_csv(DATA_DIR / "segments.csv")
    print(f"\nTotal videos: {len(index[index['file_ok']])}")

    df_raw = extract_all(index, segs)
    print(f"Extracted: {len(df_raw)} videos\n")

    # HB bins
    hb_min = int(df_raw["hb_value"].min())
    hb_max = int(df_raw["hb_value"].max()) + 1
    bin_edges  = np.arange(hb_min, hb_max + 1, 1)
    bin_labels = [f"{int(e)}–{int(e)+1}" for e in bin_edges[:-1]]
    df_raw["hb_bin"] = pd.cut(df_raw["hb_value"], bins=bin_edges,
                               labels=bin_labels, right=False)
    df_raw = df_raw.dropna(subset=["hb_bin"])

    counts = df_raw["hb_bin"].value_counts().sort_index()
    print("── Sample counts per HB bin ──")
    for b, n in counts.items():
        print(f"  {b:6s}: {'█' * n} ({n})")

    valid_bins = [b for b in bin_labels if b in counts.index and counts[b] >= 2]

    raw_cols   = [f"seg{s}_{ch}" for s in range(3) for ch in CHANNELS]
    ratio_cols = [f"r{i}{j}_{ch}"
                  for (i, j) in [(0,1),(0,2),(1,2)]
                  for ch in CHANNELS]
    all_cols   = raw_cols + ratio_cols

    # ── outlier removal ────────────────────────────────────────────────────────
    print("\n── Outlier removal (IQR ×1.5) ──")
    df = remove_outliers_iqr(df_raw, all_cols, k=3.0)

    # ── discriminability stats ────────────────────────────────────────────────
    res_df = discriminability(df, all_cols, valid_bins)
    res_df.to_csv(OUT_DIR / "discrim_anova.csv", index=False)

    print("\n── Top 30 features by ANOVA F-score ──")
    print(f"  {'Feature':<28s}  {'F':>7s}  {'|r|':>6s}  direction")
    for _, row in res_df.head(30).iterrows():
        sign = "↑ HB" if row["pearson_r"] > 0 else "↓ HB"
        print(f"  {row['feature']:<28s}  {row['f_score']:7.2f}"
              f"  {row['abs_r']:6.3f}  {sign}")

    # verdict
    good = res_df[res_df["abs_r"] > 0.20]
    ok   = res_df[(res_df["abs_r"] > 0.10) & (res_df["abs_r"] <= 0.20)]
    weak = res_df[res_df["abs_r"] <= 0.10]
    print(f"\n  |r| > 0.20 (useful)   : {len(good)}")
    print(f"  |r| 0.10–0.20 (marginal): {len(ok)}")
    print(f"  |r| ≤ 0.10 (noise)    : {len(weak)}")

    # ── box plots: raw means ───────────────────────────────────────────────────
    print("\n── Generating box plots ──")
    for seg in range(3):
        cols = [f"seg{seg}_{ch}" for ch in CHANNELS]
        boxplot_grid(
            df, cols, CHANNELS, valid_bins, res_df,
            title=(f"Raw channel means — {SEG_NAMES[seg]}\n"
                   f"(denoised: {2*WINDOW_HALF+1}-frame avg + Gaussian blur, "
                   f"outliers removed)"),
            out_path=OUT_DIR / f"discrim_boxplots_raw_seg{seg}.png")

    # ── box plots: cross-segment ratios ───────────────────────────────────────
    pairs = [(0,1), (0,2), (1,2)]
    for (i, j) in pairs:
        cols = [f"r{i}{j}_{ch}" for ch in CHANNELS]
        boxplot_grid(
            df, cols, CHANNELS, valid_bins, res_df,
            title=(f"Cross-segment ratio seg{i}/seg{j} — "
                   f"{SEG_NAMES[i]} / {SEG_NAMES[j]}\n"
                   f"(gain-invariant, outliers removed)"),
            out_path=OUT_DIR / f"discrim_boxplots_ratio_{i}{j}.png")

    # ── correlation bar chart ─────────────────────────────────────────────────
    top40 = res_df.head(40).sort_values("pearson_r")
    colors = ["#d73027" if r < 0 else "#1a9850" for r in top40["pearson_r"]]
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(top40["feature"], top40["pearson_r"], color=colors, alpha=0.8)
    ax.axvline(0,     color="black", linewidth=0.8)
    ax.axvline( 0.10, color="gray",  linewidth=0.6, linestyle="--", label="±0.10")
    ax.axvline(-0.10, color="gray",  linewidth=0.6, linestyle="--")
    ax.axvline( 0.20, color="steelblue", linewidth=0.8, linestyle="--", label="±0.20")
    ax.axvline(-0.20, color="steelblue", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson r with HB value", fontsize=10)
    ax.set_title("Top 40 features — Pearson r with HB\n"
                 "(denoised + IQR outlier removal)", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "discrim_correlations.png", dpi=140)
    plt.close()
    print(f"  saved → {OUT_DIR / 'discrim_correlations.png'}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
