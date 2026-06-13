"""
Stage 2 — LED segment detection.

For each video, detect the frame indices where each LED phase starts and ends:
  Segment 0: RED   (660 nm)
  Segment 1: ORANGE (610 nm)
  Segment 2: YELLOW (590 nm)

Detection strategy:
  1. Read every frame, compute mean brightness of center ROI.
  2. Detect two transition points (abrupt brightness changes) using a sliding-
     window variance detector on the R channel time-series.
  3. If no clean transitions are found, fall back to equal-thirds division.
  4. Validate that each segment has >= MIN_SEGMENT_FRAMES frames.

Outputs:
  data/segments.csv  —  columns: video_id, s0_start, s0_end, s1_start, s1_end,
                                  s2_start, s2_end, method (detected|fallback)

Run:
    python 02_segment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, INDEX_CSV, ROI_Y, ROI_X

SEGMENTS_CSV      = DATA_DIR / "segments.csv"
MIN_SEGMENT_FRAMES = 20


# ── helpers ──────────────────────────────────────────────────────────────────

def _roi_mean_r(frame_bgr: np.ndarray) -> float:
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * ROI_Y[0]):int(h * ROI_Y[1]),
                    int(w * ROI_X[0]):int(w * ROI_X[1])]
    # BGR → R is channel index 2
    return float(roi[:, :, 2].mean())


def _sliding_variance(ts: np.ndarray, half_win: int = 5) -> np.ndarray:
    """Local variance of ts in a window of size 2*half_win+1."""
    n = len(ts)
    var = np.zeros(n)
    for i in range(half_win, n - half_win):
        var[i] = ts[i - half_win: i + half_win + 1].var()
    return var


def _find_two_transitions(r_series: np.ndarray,
                           half_win: int = 5,
                           min_gap: int = 15) -> list[int] | None:
    """
    Find two frame indices where the R-channel signal changes most abruptly.
    Returns [t1, t2] (sorted) if two clear transitions are found, else None.
    """
    n = len(r_series)
    # absolute frame-to-frame difference
    diff = np.abs(np.diff(r_series.astype(float)))
    # smooth slightly
    kernel = np.ones(3) / 3
    diff_smooth = np.convolve(diff, kernel, mode="same")

    threshold = diff_smooth.mean() + 1.5 * diff_smooth.std()
    candidates = np.where(diff_smooth > threshold)[0]

    if len(candidates) < 2:
        return None

    # cluster candidates: merge those within min_gap frames
    groups: list[list[int]] = []
    for c in candidates:
        if groups and c - groups[-1][-1] <= min_gap:
            groups[-1].append(c)
        else:
            groups.append([c])

    if len(groups) < 2:
        return None

    # pick the strongest representative from each group
    peaks = [int(g[np.argmax(diff_smooth[g])]) for g in groups]

    # keep only the two strongest, then sort
    peaks = sorted(peaks, key=lambda p: diff_smooth[p], reverse=True)[:2]
    peaks.sort()

    # sanity: each segment must be >= MIN_SEGMENT_FRAMES frames
    t1, t2 = peaks
    if t1 < MIN_SEGMENT_FRAMES:
        return None
    if (t2 - t1) < MIN_SEGMENT_FRAMES:
        return None
    if (n - t2) < MIN_SEGMENT_FRAMES:
        return None

    return [t1, t2]


def detect_segments(video_path: str) -> dict:
    """
    Returns a dict:
      s0_start, s0_end, s1_start, s1_end, s2_start, s2_end, method
    """
    cap = cv2.VideoCapture(video_path)
    r_series = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        r_series.append(_roi_mean_r(f))
    cap.release()

    r_arr = np.array(r_series)
    n = len(r_arr)

    transitions = _find_two_transitions(r_arr)
    method = "detected"

    if transitions is None:
        # fallback: equal thirds
        t1, t2 = n // 3, 2 * n // 3
        method = "fallback"
    else:
        t1, t2 = transitions

    return {
        "s0_start": 0,       "s0_end": t1,
        "s1_start": t1,      "s1_end": t2,
        "s2_start": t2,      "s2_end": n,
        "n_frames": n,
        "method":   method,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 2 — LED Segment Detection")
    print("=" * 60)

    df = pd.read_csv(INDEX_CSV)
    usable = df[df["file_ok"] & (df["protocol"] != "other")].reset_index(drop=True)
    print(f"\nProcessing {len(usable)} videos …")

    records = []
    method_counts = {"detected": 0, "fallback": 0, "error": 0}

    for _, row in tqdm(usable.iterrows(), total=len(usable)):
        try:
            seg = detect_segments(row["video_path"])
            seg["video_id"] = row["video_id"]
            seg["protocol"] = row["protocol"]
            records.append(seg)
            method_counts[seg["method"]] += 1
        except Exception as e:
            print(f"  ERROR {row['video_id']}: {e}")
            method_counts["error"] += 1

    out = pd.DataFrame(records)
    cols = ["video_id", "protocol", "n_frames",
            "s0_start", "s0_end", "s1_start", "s1_end",
            "s2_start", "s2_end", "method"]
    out = out[cols]
    out.to_csv(SEGMENTS_CSV, index=False)

    print(f"\n── Detection method breakdown ──")
    for k, v in method_counts.items():
        print(f"  {k:10s}: {v}")

    # check segment length distribution
    out["seg0_len"] = out["s0_end"] - out["s0_start"]
    out["seg1_len"] = out["s1_end"] - out["s1_start"]
    out["seg2_len"] = out["s2_end"] - out["s2_start"]

    print(f"\n── Segment length stats (frames) ──")
    for proto in ["6s", "30s"]:
        sub = out[out["protocol"] == proto]
        if sub.empty:
            continue
        print(f"  {proto}: seg0={sub['seg0_len'].mean():.0f}±{sub['seg0_len'].std():.0f}  "
              f"seg1={sub['seg1_len'].mean():.0f}±{sub['seg1_len'].std():.0f}  "
              f"seg2={sub['seg2_len'].mean():.0f}±{sub['seg2_len'].std():.0f}")

    print(f"\n  segments.csv written → {SEGMENTS_CSV}")
    print("\nStage 2 complete.\n")


if __name__ == "__main__":
    main()
