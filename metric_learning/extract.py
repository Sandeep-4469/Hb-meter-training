"""
Step 1 — Histogram-based feature extraction (middle frame per segment).

For each segment:
  1. Extract middle frame, apply Gaussian blur to ROI
  2. Convert to RGB, HSV, Lab, Gray (10 channels)
  3. Per channel:
       - 16-bin normalised histogram (pixel value distribution)
       - 8 statistics: mean, std, median, skew, kurt, entropy, p10, p90
  4. Cross-segment ratios (Beer-Lambert differential transmittance):
       - Mean ratio   seg_i / seg_j  for 3 pairs × 10 channels = 30
       - Std ratio    seg_i / seg_j                             = 30
       - χ² histogram distance between segments                 = 30
  5. Within-segment gain-invariant ratios:
       - R/G, R/B, G/B per segment                             =  9

Feature count:
  Histogram bins : 3 × 10 × 16 = 480
  Statistics     : 3 × 10 ×  8 = 240
  Mean ratios    : 3 ×  10     =  30
  Std ratios     : 3 ×  10     =  30
  χ² dist        : 3 ×  10     =  30
  RG/RB/GB       : 3 ×   3     =   9
  ─────────────────────────────────────
  Total                          819

Outputs:
  results/features.csv
  results/features_train.csv
  results/features_test.csv
  results/feature_correlations.csv
  results/feature_importance.png

Run:
    python extract.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, ROI_Y, ROI_X

OUT_DIR    = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS   = ["R", "G", "B", "H", "S", "V", "L", "A", "B_lab", "Gray"]
N_BINS     = 16
BLUR_KSIZE = (7, 7)
MIN_BRIGHT = 10.0
MAX_BRIGHT = 250.0

# channel histogram ranges (OpenCV conventions)
CH_RANGES  = {
    "R": (0, 256), "G": (0, 256), "B": (0, 256),
    "H": (0, 180), "S": (0, 256), "V": (0, 256),
    "L": (0, 256), "A": (0, 256), "B_lab": (0, 256),
    "Gray": (0, 256),
}


# ── image utils ───────────────────────────────────────────────────────────────

def crop_roi(bgr):
    h, w = bgr.shape[:2]
    return bgr[int(h*ROI_Y[0]):int(h*ROI_Y[1]),
               int(w*ROI_X[0]):int(w*ROI_X[1])]


def extract_channels(bgr):
    """Return dict of 2D arrays for each channel."""
    roi  = cv2.GaussianBlur(crop_roi(bgr), BLUR_KSIZE, 0)
    rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab  = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "R": rgb[:,:,0], "G": rgb[:,:,1], "B": rgb[:,:,2],
        "H": hsv[:,:,0], "S": hsv[:,:,1], "V": hsv[:,:,2],
        "L": lab[:,:,0], "A": lab[:,:,1], "B_lab": lab[:,:,2],
        "Gray": gray,
    }


def get_middle_frame(path, s_start, s_end):
    """Read middle frame of segment; skip ±3 frames if brightness out of range."""
    mid = (s_start + s_end) // 2
    cap = cv2.VideoCapture(path)
    # try mid ± increasing offsets; fall back to best-brightness frame in segment
    best_frame, best_dist = None, float("inf")
    for offset in [0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8]:
        fid = mid + offset
        if fid < s_start or fid >= s_end:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        gray_mean = cv2.cvtColor(
            cv2.GaussianBlur(crop_roi(frame), BLUR_KSIZE, 0),
            cv2.COLOR_BGR2GRAY
        ).mean()
        if MIN_BRIGHT <= gray_mean <= MAX_BRIGHT:
            cap.release()
            return frame
        # track the frame closest to acceptable range
        dist = min(abs(gray_mean - MIN_BRIGHT), abs(gray_mean - MAX_BRIGHT))
        if dist < best_dist:
            best_dist, best_frame = dist, frame.copy()
    cap.release()
    # return best available frame even if slightly outside brightness bounds
    return best_frame


# ── feature computation ───────────────────────────────────────────────────────

def channel_histogram(pixels, ch_name):
    lo, hi = CH_RANGES[ch_name]
    hist, _ = np.histogram(pixels.flatten(), bins=N_BINS, range=(lo, hi))
    total = hist.sum()
    return (hist / (total + 1e-8)).astype(np.float32)


def channel_stats(pixels):
    """[mean, std, median, skew, kurt, entropy, p10, p90]"""
    flat = pixels.flatten().astype(np.float64)
    hist, _ = np.histogram(flat, bins=64, range=(0, 256))
    p = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))
    return np.array([
        flat.mean(),
        flat.std(),
        float(np.median(flat)),
        float(sp_stats.skew(flat)),
        float(sp_stats.kurtosis(flat)),
        entropy,
        float(np.percentile(flat, 10)),
        float(np.percentile(flat, 90)),
    ], dtype=np.float32)


def chi2_hist_distance(h1, h2):
    """Normalised χ² distance between two normalised histograms."""
    denom = h1 + h2 + 1e-8
    return float(np.sum((h1 - h2) ** 2 / denom))


def safe_ratio(a, b, lo=0.001, hi=1000.0):
    return float(np.clip((a + 1e-6) / (b + 1e-6), lo, hi))


# ── feature name builder ──────────────────────────────────────────────────────

def build_feature_names():
    names = []
    # histogram bins
    for s in range(3):
        for ch in CHANNELS:
            for b in range(N_BINS):
                names.append(f"seg{s}_{ch}_h{b:02d}")
    # statistics
    STAT_NAMES = ["mean", "std", "median", "skew", "kurt", "entropy", "p10", "p90"]
    for s in range(3):
        for ch in CHANNELS:
            for st in STAT_NAMES:
                names.append(f"seg{s}_{ch}_{st}")
    # cross-segment mean ratios
    for (i, j) in [(0,1),(0,2),(1,2)]:
        for ch in CHANNELS:
            names.append(f"r{i}{j}_{ch}_mean")
    # cross-segment std ratios
    for (i, j) in [(0,1),(0,2),(1,2)]:
        for ch in CHANNELS:
            names.append(f"r{i}{j}_{ch}_std")
    # cross-segment χ² histogram distance
    for (i, j) in [(0,1),(0,2),(1,2)]:
        for ch in CHANNELS:
            names.append(f"chi2_{i}{j}_{ch}")
    # within-segment colour ratios
    for s in range(3):
        names.append(f"seg{s}_RG")
        names.append(f"seg{s}_RB")
        names.append(f"seg{s}_GB")
    return names


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("STEP 1 — Histogram Feature Extraction (middle frame)")
    print("=" * 65)

    index = pd.read_csv(DATA_DIR / "index.csv")
    segs  = pd.read_csv(DATA_DIR / "segments.csv")
    df    = index[index["file_ok"]].merge(
                segs[["video_id","s0_start","s0_end",
                      "s1_start","s1_end","s2_start","s2_end"]],
                on="video_id", how="inner")

    df = df.reset_index(drop=True)
    print(f"Videos: {len(df)}  (6s: {(df['protocol']=='6s').sum()}  30s: {(df['protocol']=='30s').sum()})")
    print("  Note: 30s R-channel may saturate at 255 — histogram encodes this naturally")

    feat_names = build_feature_names()
    records = []
    errors  = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        frames = []
        ok = True
        for s in range(3):
            frame = get_middle_frame(row["video_path"],
                                     int(row[f"s{s}_start"]),
                                     int(row[f"s{s}_end"]))
            if frame is None:
                errors.append(row["video_id"]); ok = False; break
            frames.append(frame)
        if not ok:
            continue

        # extract channels for all 3 segments
        seg_channels = [extract_channels(f) for f in frames]

        rec = {"video_id": row["video_id"],
               "hb_value": row["hb_value"],
               "split":    row["split"],
               "protocol": row["protocol"],
               "is_30s":   float(row["protocol"] == "30s")}

        # precompute histograms and stats for all segments and channels
        seg_hists = []   # seg_hists[s][ch] = np.array of N_BINS
        seg_stats = []   # seg_stats[s][ch] = np.array of 8
        for s in range(3):
            hists, stats = {}, {}
            for ch in CHANNELS:
                px = seg_channels[s][ch]
                hists[ch] = channel_histogram(px, ch)
                stats[ch] = channel_stats(px)
            seg_hists.append(hists)
            seg_stats.append(stats)

        # ── histogram bins ────────────────────────────────────────────────────
        for s in range(3):
            for ch in CHANNELS:
                for b, v in enumerate(seg_hists[s][ch]):
                    rec[f"seg{s}_{ch}_h{b:02d}"] = float(v)

        # ── statistics ───────────────────────────────────────────────────────
        STAT_NAMES = ["mean", "std", "median", "skew", "kurt",
                      "entropy", "p10", "p90"]
        for s in range(3):
            for ch in CHANNELS:
                for st, v in zip(STAT_NAMES, seg_stats[s][ch]):
                    rec[f"seg{s}_{ch}_{st}"] = float(v)

        # ── cross-segment mean ratios ─────────────────────────────────────────
        for (i, j) in [(0,1),(0,2),(1,2)]:
            for ch in CHANNELS:
                mi = float(seg_stats[i][ch][0])   # mean
                mj = float(seg_stats[j][ch][0])
                rec[f"r{i}{j}_{ch}_mean"] = safe_ratio(mi, mj)

        # ── cross-segment std ratios ──────────────────────────────────────────
        for (i, j) in [(0,1),(0,2),(1,2)]:
            for ch in CHANNELS:
                si = float(seg_stats[i][ch][1])   # std
                sj = float(seg_stats[j][ch][1])
                rec[f"r{i}{j}_{ch}_std"] = safe_ratio(si, sj)

        # ── cross-segment χ² histogram distance ──────────────────────────────
        for (i, j) in [(0,1),(0,2),(1,2)]:
            for ch in CHANNELS:
                rec[f"chi2_{i}{j}_{ch}"] = chi2_hist_distance(
                    seg_hists[i][ch], seg_hists[j][ch])

        # ── within-segment colour ratios ──────────────────────────────────────
        for s in range(3):
            mr = float(seg_stats[s]["R"][0])
            mg = float(seg_stats[s]["G"][0])
            mb = float(seg_stats[s]["B"][0])
            rec[f"seg{s}_RG"] = safe_ratio(mr, mg)
            rec[f"seg{s}_RB"] = safe_ratio(mr, mb)
            rec[f"seg{s}_GB"] = safe_ratio(mg, mb)

        records.append(rec)

    if errors:
        print(f"  Errors (no valid frame): {len(errors)}")

    out      = pd.DataFrame(records)
    train_df = out[out["split"] == "train"]
    test_df  = out[out["split"] == "test"]

    out.to_csv(OUT_DIR / "features.csv",           index=False)
    train_df.to_csv(OUT_DIR / "features_train.csv", index=False)
    test_df.to_csv(OUT_DIR  / "features_test.csv",  index=False)

    print(f"\nTotal: {len(out)}  Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"Features: {len(feat_names)}")

    # ── correlation analysis ──────────────────────────────────────────────────
    feat_names_full = feat_names + ["is_30s"]
    corrs = train_df[feat_names_full].corrwith(train_df["hb_value"]).abs()
    corrs_signed = train_df[feat_names_full].corrwith(train_df["hb_value"])

    corr_df = pd.DataFrame({
        "feature":     corrs.index,
        "abs_r":       corrs.values,
        "signed_r":    corrs_signed.values,
    }).sort_values("abs_r", ascending=False).reset_index(drop=True)

    # assign group label
    def group_label(n):
        if "_h" in n and n.split("_h")[-1].isdigit():
            return "Histogram bin"
        if any(n.endswith(f"_{s}") for s in
               ["mean","std","median","skew","kurt","entropy","p10","p90"]):
            return "Statistic"
        if n.startswith("chi2"):
            return "χ² hist dist"
        if "_mean" in n or "_std" in n:
            return "Cross-seg ratio"
        if "_RG" in n or "_RB" in n or "_GB" in n:
            return "Colour ratio"
        return "Other"

    corr_df["group"] = corr_df["feature"].apply(group_label)
    corr_df.to_csv(OUT_DIR / "feature_correlations.csv", index=False)

    print("\nTop 25 features by |r| with HB (train):")
    print(f"  {'Feature':<35s}  {'|r|':>6s}  {'r':>7s}  Group")
    print("  " + "─" * 70)
    for _, row2 in corr_df.head(25).iterrows():
        print(f"  {row2['feature']:<35s}  {row2['abs_r']:6.4f}  "
              f"{row2['signed_r']:+7.4f}  {row2['group']}")

    # ── group summary ─────────────────────────────────────────────────────────
    print("\nCorrelation summary by feature group:")
    print(f"  {'Group':<20s}  {'Count':>6s}  {'Mean|r|':>8s}  "
          f"{'Max|r|':>7s}  {'|r|>0.20':>9s}  {'|r|>0.25':>9s}")
    print("  " + "─" * 70)
    for g, sub in corr_df.groupby("group"):
        print(f"  {g:<20s}  {len(sub):6d}  {sub['abs_r'].mean():8.4f}  "
              f"{sub['abs_r'].max():7.4f}  "
              f"{(sub['abs_r']>0.20).sum():9d}  "
              f"{(sub['abs_r']>0.25).sum():9d}")

    # ── visualisation ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # top 30 features bar chart
    ax = axes[0, 0]
    top30 = corr_df.head(30)
    colors = {"Histogram bin": "#3498db", "Statistic": "#e74c3c",
              "χ² hist dist": "#2ecc71", "Cross-seg ratio": "#f39c12",
              "Colour ratio": "#9b59b6", "Other": "#95a5a6"}
    bar_colors = [colors.get(g, "#95a5a6") for g in top30["group"]]
    ax.barh(range(len(top30)), top30["abs_r"].values[::-1],
            color=list(reversed(bar_colors)), alpha=0.8)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"].values[::-1], fontsize=7)
    ax.axvline(0.20, color="red", lw=1, linestyle="--", label="|r|=0.20")
    ax.axvline(0.25, color="orange", lw=1, linestyle="--", label="|r|=0.25")
    ax.set_xlabel("|Pearson r| with HB")
    ax.set_title("Top 30 Features by |r|", fontsize=11)
    ax.legend(fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in colors.items()]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    # distribution of |r| by group
    ax = axes[0, 1]
    for g, sub in corr_df.groupby("group"):
        ax.hist(sub["abs_r"], bins=20, alpha=0.5, label=g,
                color=colors.get(g, "#95a5a6"))
    ax.axvline(0.20, color="red", lw=1.5, linestyle="--", label="|r|=0.20")
    ax.set_xlabel("|Pearson r| with HB")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of |r| by Feature Group", fontsize=11)
    ax.legend(fontsize=7)

    # channel-level breakdown: mean |r| per channel across all segments & stats
    ax = axes[1, 0]
    ch_means = {}
    for ch in CHANNELS:
        mask = corr_df["feature"].str.contains(f"_{ch}") | \
               corr_df["feature"].str.contains(f"_{ch}_")
        ch_means[ch] = corr_df[mask]["abs_r"].mean()
    ch_df = pd.Series(ch_means).sort_values(ascending=False)
    ax.bar(ch_df.index, ch_df.values, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axhline(0.15, color="red", lw=1, linestyle="--")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean |r| across all features")
    ax.set_title("Mean |r| by Channel", fontsize=11)
    ax.tick_params(axis="x", rotation=30)

    # top histogram bins per segment
    ax = axes[1, 1]
    hist_corrs = corr_df[corr_df["group"] == "Histogram bin"]
    for s, color in [(0, "#e74c3c"), (1, "#3498db"), (2, "#2ecc71")]:
        seg_hist = hist_corrs[hist_corrs["feature"].str.startswith(f"seg{s}_")]
        by_ch = seg_hist.groupby(
            seg_hist["feature"].str.extract(r"seg\d_(\w+)_h")[0]
        )["abs_r"].max()
        ax.bar(np.arange(len(by_ch)) + s * 0.25, by_ch.values,
               width=0.25, label=f"Seg{s}", color=color, alpha=0.7)
    ax.set_xticks(np.arange(len(CHANNELS)) + 0.25)
    ax.set_xticklabels(CHANNELS, rotation=30, fontsize=8)
    ax.set_ylabel("Max |r| of any histogram bin")
    ax.set_title("Best Histogram Bin |r| by Channel × Segment", fontsize=11)
    ax.legend()

    plt.suptitle("Feature Correlation Analysis — HB Meter (6s protocol)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  saved → {OUT_DIR / 'feature_importance.png'}")
    print(f"  saved → {OUT_DIR / 'feature_correlations.csv'}")
    print("\nStep 1 complete.\n")


if __name__ == "__main__":
    main()
