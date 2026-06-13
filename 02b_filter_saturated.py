"""
Stage 2b — Saturation filter for 30s videos.

For each 30s video, check whether the R channel is clipped (mean > threshold)
in any segment. Saturated videos carry zero cross-LED spectral information and
must be excluded from training.

Also flags any 6s videos with partial saturation (>10% of frames clipped).

Writes:
  data/usable.csv   — videos that pass the filter, with a 'use_r' column
                      (True = R channel usable, False = R saturated but
                       G/B channels still available)

Run:
    python 02b_filter_saturated.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, RESULTS_DIR, INDEX_CSV, USABLE_CSV,
    R_SAT_THRESHOLD, ROI_Y, ROI_X,
)

SEGMENTS_CSV = DATA_DIR / "segments.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def _segment_r_stats(video_path: str,
                     s_start: int, s_end: int) -> tuple[float, float]:
    """Return (mean_R, sat_fraction) for a segment defined by frame range."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_start)
    r_vals = []
    for _ in range(s_end - s_start):
        ret, f = cap.read()
        if not ret:
            break
        h, w = f.shape[:2]
        roi = f[int(h * ROI_Y[0]):int(h * ROI_Y[1]),
                int(w * ROI_X[0]):int(w * ROI_X[1])]
        r_mean = roi[:, :, 2].astype(float).mean()   # BGR → index 2 = R
        r_vals.append(r_mean)
    cap.release()
    if not r_vals:
        return 0.0, 0.0
    arr = np.array(r_vals)
    return float(arr.mean()), float((arr > R_SAT_THRESHOLD).mean())


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 2b — Saturation Filter")
    print("=" * 60)

    index    = pd.read_csv(INDEX_CSV)
    segments = pd.read_csv(SEGMENTS_CSV)
    df = index.merge(segments, on=["video_id", "protocol"])

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["video_path"]

        r_means  = []
        sat_frac = []
        for seg_idx in range(3):
            mean_r, frac = _segment_r_stats(
                path,
                int(row[f"s{seg_idx}_start"]),
                int(row[f"s{seg_idx}_end"]),
            )
            r_means.append(mean_r)
            sat_frac.append(frac)

        max_mean_r   = max(r_means)
        max_sat_frac = max(sat_frac)

        # R is usable if no segment has mean R > threshold AND sat fraction < 20%
        use_r = (max_mean_r < R_SAT_THRESHOLD) and (max_sat_frac < 0.20)

        records.append({
            "video_id":       row["video_id"],
            "protocol":       row["protocol"],
            "split":          row["split"],
            "hb_value":       row["hb_value"],
            "video_path":     path,
            "seg0_mean_r":    r_means[0],
            "seg1_mean_r":    r_means[1],
            "seg2_mean_r":    r_means[2],
            "max_sat_frac":   max_sat_frac,
            "use_r":          use_r,
            # always include; downstream code decides whether to use G/B fallback
            "include":        True,
        })

    result = pd.DataFrame(records)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n── R-channel saturation summary ──")
    for proto in ["6s", "30s"]:
        sub = result[result["protocol"] == proto]
        sat = sub[~sub["use_r"]]
        print(f"  {proto:4s}: total={len(sub):3d}  "
              f"R-saturated={len(sat):3d} ({100*len(sat)/max(len(sub),1):.0f}%)  "
              f"R-usable={len(sub)-len(sat):3d}")

    # For 30s videos where R is fully saturated, mark as exclude
    # (G/B channels have almost no cross-segment variation either — confirmed in EDA)
    fully_dead = (result["protocol"] == "30s") & (~result["use_r"]) & \
                 (result["max_sat_frac"] > 0.80)
    result.loc[fully_dead, "include"] = False
    excluded = result[~result["include"]]
    print(f"\n  Excluded (fully saturated R, no fallback): {len(excluded)}")
    print(f"  Remaining usable: {len(result[result['include']])}")

    # ── saturation scatter plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, proto in zip(axes, ["6s", "30s"]):
        sub = result[result["protocol"] == proto]
        colors = sub["use_r"].map({True: "steelblue", False: "tomato"})
        ax.scatter(sub["hb_value"], sub["seg0_mean_r"],
                   c=colors, alpha=0.6, s=20)
        ax.axhline(R_SAT_THRESHOLD, color="black", linestyle="--",
                   linewidth=1, label=f"sat threshold ({R_SAT_THRESHOLD})")
        ax.set_title(f"{proto} — R mean seg0 vs HB")
        ax.set_xlabel("HB (g/dL)")
        ax.set_ylabel("Mean R (seg0)")
        ax.legend(fontsize=8)
    plt.suptitle("R-channel saturation by protocol  (blue=usable, red=saturated)")
    plt.tight_layout()
    sat_plot = RESULTS_DIR / "02b_saturation.png"
    plt.savefig(sat_plot, dpi=120)
    plt.close()
    print(f"\n  saturation plot → {sat_plot}")

    # ── write usable.csv ──────────────────────────────────────────────────────
    usable = result[result["include"]].copy()
    usable_cols = [
        "video_id", "video_path", "hb_value", "split", "protocol",
        "use_r", "seg0_mean_r", "seg1_mean_r", "seg2_mean_r", "max_sat_frac",
    ]
    usable[usable_cols].to_csv(USABLE_CSV, index=False)
    print(f"\n  usable.csv written → {USABLE_CSV}")

    # merge segment boundaries back
    seg_cols = ["video_id", "n_frames",
                "s0_start", "s0_end", "s1_start", "s1_end",
                "s2_start", "s2_end", "method"]
    usable_full = usable[usable_cols].merge(
        segments[seg_cols], on="video_id", how="left"
    )
    usable_full.to_csv(USABLE_CSV, index=False)

    print(f"  final rows: {len(usable_full)}  "
          f"(6s={len(usable_full[usable_full.protocol=='6s'])}  "
          f"30s={len(usable_full[usable_full.protocol=='30s'])})")
    print("\nStage 2b complete.\n")


if __name__ == "__main__":
    main()
