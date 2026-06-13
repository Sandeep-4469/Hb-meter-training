"""
Temporal feature extraction — reads ALL frames in each segment.

Per segment × per channel (10 channels):
  DC     : mean of per-frame spatial means  (average transmittance)
  AC     : std  of per-frame spatial means  (pulsatile / heartbeat amplitude)
  AC/DC  : normalised pulsatile ratio
  FFT1   : magnitude at ~1 Hz (heartbeat fundamental, bin 2 of 30fps signal)
  FFT2   : magnitude at ~2 Hz (first harmonic)
  trend  : linear slope of DC signal over time (motion / breathing drift)

Cross-segment ratios (Beer-Lambert differential):
  DC_ratio seg_i/seg_j   per channel  (30 features)
  AC_ratio seg_i/seg_j   per channel  (30 features)

Total temporal features: 3×10×6 = 180  +  60  = 240
Combined with existing 820 histogram features → 1060 total

Outputs:
  results/features_combined_train.csv
  results/features_combined_test.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, ROI_Y, ROI_X

OUT_DIR  = Path(__file__).parent / "results"
HIST_DIR = ROOT / "metric_learning" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS   = ["R","G","B","H","S","V","L","A","B_lab","Gray"]
BLUR_KSIZE = (7, 7)


def crop_roi(bgr):
    h, w = bgr.shape[:2]
    return bgr[int(h*ROI_Y[0]):int(h*ROI_Y[1]),
               int(w*ROI_X[0]):int(w*ROI_X[1])]


def frame_channel_means(bgr):
    """Return dict ch→scalar mean for one frame."""
    roi  = cv2.GaussianBlur(crop_roi(bgr), BLUR_KSIZE, 0)
    rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab  = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "R":     rgb[:,:,0].mean(), "G": rgb[:,:,1].mean(),
        "B":     rgb[:,:,2].mean(),
        "H":     hsv[:,:,0].mean(), "S": hsv[:,:,1].mean(),
        "V":     hsv[:,:,2].mean(),
        "L":     lab[:,:,0].mean(), "A": lab[:,:,1].mean(),
        "B_lab": lab[:,:,2].mean(),
        "Gray":  gray.mean(),
    }


def safe_ratio(a, b, lo=0.001, hi=1000.0):
    return float(np.clip((a + 1e-6) / (b + 1e-6), lo, hi))


def extract_all_segments(video_path, seg_ranges):
    """
    Read the video in ONE sequential pass and collect per-channel means
    for all requested frame ranges.  Much faster than seeking for each range.

    seg_ranges : list of (s_start, s_end) tuples, sorted by s_start
    Returns list of dicts  ch → np.array  (one per segment)
    """
    # build a flat target set and a map fid→segment indices
    all_fids = set()
    for lo, hi in seg_ranges:
        all_fids.update(range(lo, hi))
    max_fid = max(all_fids) if all_fids else 0

    cap = cv2.VideoCapture(video_path)
    seg_series = [{ch: [] for ch in CHANNELS} for _ in seg_ranges]

    fid = 0
    while fid <= max_fid:
        ret, frame = cap.read()
        if not ret:
            break
        if fid in all_fids:
            means = frame_channel_means(frame)
            for s_idx, (lo, hi) in enumerate(seg_ranges):
                if lo <= fid < hi:
                    for ch in CHANNELS:
                        seg_series[s_idx][ch].append(means[ch])
        fid += 1

    cap.release()
    return [{ch: np.array(v, dtype=np.float32) for ch, v in ss.items()}
            for ss in seg_series]


def temporal_features(series_dict, fps=30.0):
    """
    Given per-channel time series, return feature dict.
    Features: DC, AC, AC/DC, FFT1, FFT2, trend  (per channel)
    """
    feats = {}
    for ch, sig in series_dict.items():
        if len(sig) < 4:
            dc = ac = acdc = f1 = f2 = slope = float("nan")
        else:
            dc    = float(sig.mean())
            ac    = float(sig.std())
            acdc  = safe_ratio(ac, dc)
            # FFT: magnitude at ~1Hz and ~2Hz
            n     = len(sig)
            fft   = np.abs(np.fft.rfft(sig - sig.mean()))
            freqs = np.fft.rfftfreq(n, d=1.0/fps)
            # bin closest to 1 Hz and 2 Hz
            b1    = int(np.argmin(np.abs(freqs - 1.0)))
            b2    = int(np.argmin(np.abs(freqs - 2.0)))
            norm  = fft.sum() + 1e-8
            f1    = float(fft[b1] / norm)
            f2    = float(fft[b2] / norm)
            # linear trend (slope)
            t     = np.arange(len(sig), dtype=np.float32)
            slope = float(np.polyfit(t, sig, 1)[0])
        feats[ch] = dict(DC=dc, AC=ac, ACDC=acdc, FFT1=f1, FFT2=f2, trend=slope)
    return feats


def build_feature_row(seg_feats):
    """
    seg_feats: list of 3 dicts (one per segment) from temporal_features()
    Returns flat dict of temporal features.
    """
    rec = {}
    STAT_NAMES = ["DC","AC","ACDC","FFT1","FFT2","trend"]

    # per-segment per-channel features
    for s, sf in enumerate(seg_feats):
        for ch in CHANNELS:
            for st in STAT_NAMES:
                rec[f"t_seg{s}_{ch}_{st}"] = sf[ch][st]

    # cross-segment DC ratios
    for (i,j) in [(0,1),(0,2),(1,2)]:
        for ch in CHANNELS:
            rec[f"t_r{i}{j}_{ch}_DC"] = safe_ratio(
                seg_feats[i][ch]["DC"], seg_feats[j][ch]["DC"])
            rec[f"t_r{i}{j}_{ch}_AC"] = safe_ratio(
                seg_feats[i][ch]["AC"], seg_feats[j][ch]["AC"])

    return rec


def process_one(args):
    """Worker function for parallel processing."""
    video_id, video_path, seg_ranges = args
    try:
        all_series = extract_all_segments(video_path, seg_ranges)
        if all(all(len(v) == 0 for v in ts.values()) for ts in all_series):
            return None, video_id
        seg_feats = [temporal_features(ts, fps=30.0) for ts in all_series]
        rec = {"video_id": video_id}
        rec.update(build_feature_row(seg_feats))
        return rec, None
    except Exception as e:
        return None, f"{video_id}: {e}"


def main():
    print("=" * 65)
    print("  Temporal Feature Extraction — all frames per segment")
    print("=" * 65)

    index = pd.read_csv(DATA_DIR / "index.csv")
    segs  = pd.read_csv(DATA_DIR / "segments.csv")
    df    = index[index["file_ok"]].merge(
                segs.drop(columns=["protocol"]), on="video_id", how="inner")
    df    = df.reset_index(drop=True)
    print(f"  Videos: {len(df)}  (6s: {(df['protocol']=='6s').sum()}  "
          f"30s: {(df['protocol']=='30s').sum()})")

    # load existing histogram features to merge with
    hist_train = pd.read_csv(HIST_DIR / "features_train.csv")
    hist_test  = pd.read_csv(HIST_DIR / "features_test.csv")
    hist_all   = pd.concat([hist_train, hist_test], ignore_index=True)

    # build task list
    tasks = []
    for _, row in df.iterrows():
        seg_ranges = [(int(row[f"s{s}_start"]), int(row[f"s{s}_end"]))
                      for s in range(3)]
        tasks.append((row["video_id"], row["video_path"], seg_ranges))

    n_workers = min(16, cpu_count())
    print(f"  Workers: {n_workers}")

    temporal_rows = []
    errors        = []

    with Pool(n_workers) as pool:
        for rec, err in tqdm(pool.imap(process_one, tasks), total=len(tasks)):
            if err is not None:
                errors.append(err)
            else:
                temporal_rows.append(rec)

    if errors:
        print(f"  Errors: {len(errors)}")

    temp_df = pd.DataFrame(temporal_rows)
    print(f"  Temporal features per video: {len(temp_df.columns)-1}")

    # merge with histogram features
    combined = hist_all.merge(temp_df, on="video_id", how="inner")
    train_c  = combined[combined["split"] == "train"].reset_index(drop=True)
    test_c   = combined[combined["split"] == "test"].reset_index(drop=True)

    feat_cols = [c for c in combined.columns
                 if c not in {"video_id","hb_value","split","protocol"}
                 and c != "anemia"]
    print(f"  Combined features: {len(feat_cols)}")
    print(f"    Histogram : 820")
    print(f"    Temporal  : {len(feat_cols)-820}")
    print(f"  Train: {len(train_c)}  Test: {len(test_c)}")

    train_c.to_csv(OUT_DIR / "features_combined_train.csv", index=False)
    test_c.to_csv(OUT_DIR  / "features_combined_test.csv",  index=False)
    print(f"\n  Saved → {OUT_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
