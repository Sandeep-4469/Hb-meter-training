"""
Stage 3 — Time-series feature extraction (all videos, both protocols).

For each LED segment we read ALL frames and compute a per-channel signal
(mean ROI pixel value over time).  From that signal we extract:

  Per segment × per channel (7 channels: R, G, B, H, S, V, Gray):
    DC         — mean signal level (avg brightness)
    AC         — std of signal (pulsatile amplitude)
    AC/DC      — normalised pulsatile ratio (like pulse-ox R-ratio)
    min, max, range
    slope      — linear trend across segment
    p10/p25/p50/p75/p90
    skewness, kurtosis
    dom_freq   — dominant FFT frequency index (heartbeat)
    dom_mag    — magnitude at dominant frequency
    fft_top2/3 — 2nd and 3rd strongest FFT components

  Cross-segment on DC means (pairs 01, 02, 12):
    ratio    = DC_seg_i / DC_seg_j      (gain-invariant)
    logratio = log|ratio| × sign

  Cross-segment on AC/DC (pairs 01, 02, 12):
    acdc_ratio = (AC/DC)_seg_i / (AC/DC)_seg_j   (pulse-ox style)

Outputs
───────
  outputs/features/features.csv
  outputs/features/features_train.csv
  outputs/features/features_test.csv

Run:
    python 03_extract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, FEATURES_DIR, ROI_Y, ROI_X

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_CSV    = DATA_DIR / "index.csv"
SEGMENTS_CSV = DATA_DIR / "segments.csv"
FEATURES_CSV = FEATURES_DIR / "features.csv"

CHANNELS = ["R", "G", "B", "H", "S", "V", "Gray"]

STATS_PER_CH = [
    "dc", "ac", "ac_dc",
    "min", "max", "range",
    "slope",
    "p10", "p25", "p50", "p75", "p90",
    "skew", "kurt",
    "dom_freq", "dom_mag", "fft2_mag", "fft3_mag",
]


# ── per-frame helpers ─────────────────────────────────────────────────────────

def crop_roi(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    y0, y1 = int(h * ROI_Y[0]), int(h * ROI_Y[1])
    x0, x1 = int(w * ROI_X[0]), int(w * ROI_X[1])
    return bgr[y0:y1, x0:x1]


def _circular_mean_hue(hue_opencv: np.ndarray) -> float:
    """
    Correct mean of the OpenCV Hue channel.

    Hue is a CIRCULAR quantity: OpenCV encodes it in [0, 180) to represent the
    full [0, 360) degree colour wheel, so 179 and 1 are 2 degrees apart, not 178.
    A plain arithmetic ``.mean()`` (as used previously) is therefore invalid and
    produces a meaningless feature.  We compute the proper circular mean via the
    vector-average of the corresponding unit angles.
    """
    angles = hue_opencv.astype(np.float64) * (np.pi / 90.0)   # [0,180) -> [0,2π)
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    mean_deg = np.degrees(mean_angle) % 360.0
    return float(mean_deg / 2.0)                              # back to OpenCV scale


def frame_channel_means(bgr: np.ndarray) -> np.ndarray:
    """Return mean ROI value for each of 7 channels."""
    roi  = crop_roi(bgr)
    rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.array([
        rgb[:, :, 0].mean(),                # R
        rgb[:, :, 1].mean(),                # G
        rgb[:, :, 2].mean(),                # B
        _circular_mean_hue(hsv[:, :, 0]),   # H  (circular mean — see helper)
        hsv[:, :, 1].mean(),                # S
        hsv[:, :, 2].mean(),                # V
        gray.mean(),                        # Gray
    ], dtype=np.float32)


# ── segment time-series ───────────────────────────────────────────────────────

def read_segment(video_path: str, s_start: int, s_end: int) -> np.ndarray:
    """
    Read all frames in [s_start, s_end) and return mean-per-channel array
    of shape (n_frames, 7).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_start)
    rows = []
    for _ in range(s_end - s_start):
        ret, frame = cap.read()
        if not ret:
            break
        rows.append(frame_channel_means(frame))
    cap.release()
    if not rows:
        raise ValueError(f"No frames read from {video_path} [{s_start},{s_end})")
    return np.stack(rows)   # (n_frames, 7)


# ── feature extraction from one signal vector ─────────────────────────────────

def signal_features(sig: np.ndarray) -> np.ndarray:
    """
    18 features from a 1-D time series (one channel over one segment).
    """
    n  = len(sig)
    dc = float(sig.mean())
    ac = float(sig.std())
    ac_dc = ac / (dc + 1e-8)

    mn, mx = float(sig.min()), float(sig.max())
    rng = mx - mn

    x = np.arange(n, dtype=np.float32)
    slope = float(np.polyfit(x, sig, 1)[0])

    p10, p25, p50, p75, p90 = np.percentile(sig, [10, 25, 50, 75, 90]).tolist()

    skew = float(sp_stats.skew(sig)) if n > 2 else 0.0
    kurt = float(sp_stats.kurtosis(sig)) if n > 2 else 0.0

    # FFT on detrended signal
    detrended = sig - np.polyval(np.polyfit(x, sig, 1), x)
    fft_mag = np.abs(np.fft.rfft(detrended))[1:]   # skip DC bin
    if len(fft_mag) >= 3:
        top3_idx = np.argsort(fft_mag)[-3:][::-1]
        dom_freq = float(top3_idx[0] + 1)
        dom_mag  = float(fft_mag[top3_idx[0]])
        fft2_mag = float(fft_mag[top3_idx[1]])
        fft3_mag = float(fft_mag[top3_idx[2]])
    elif len(fft_mag) >= 1:
        dom_freq = float(np.argmax(fft_mag) + 1)
        dom_mag  = float(fft_mag.max())
        fft2_mag = fft3_mag = 0.0
    else:
        dom_freq = dom_mag = fft2_mag = fft3_mag = 0.0

    return np.array([
        dc, ac, ac_dc,
        mn, mx, rng,
        slope,
        p10, p25, p50, p75, p90,
        skew, kurt,
        dom_freq, dom_mag, fft2_mag, fft3_mag,
    ], dtype=np.float32)


# ── full video feature vector ──────────────────────────────────────────────────

def extract_video(video_path: str,
                  seg_bounds: list[tuple[int, int]]) -> np.ndarray:
    """
    Extract full feature vector for one video.
    seg_bounds: [(s0_start, s0_end), (s1_start, s1_end), (s2_start, s2_end)]
    """
    parts    = []
    dc_means = []   # shape (3, 7)  — for cross-seg ratio features
    acdc     = []   # shape (3, 7)  — for cross-seg AC/DC ratio features

    for s_start, s_end in seg_bounds:
        ts = read_segment(video_path, s_start, s_end)   # (n_frames, 7)

        seg_dc   = []
        seg_acdc = []
        for ci in range(len(CHANNELS)):
            sig  = ts[:, ci].astype(np.float64)
            feat = signal_features(sig.astype(np.float32))
            parts.append(feat)
            seg_dc.append(float(sig.mean()))
            seg_acdc.append(feat[2])   # ac_dc index

        dc_means.append(seg_dc)
        acdc.append(seg_acdc)

    # cross-segment DC ratios
    dm = np.array(dc_means, dtype=np.float64)   # (3, 7)
    ad = np.array(acdc, dtype=np.float64)        # (3, 7)

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        mi = dm[i] + 1e-8
        mj = dm[j] + 1e-8
        ratio    = (mi / mj).astype(np.float32)
        logratio = (np.sign(mi / mj) * np.log(np.abs(mi / mj) + 1e-8)).astype(np.float32)
        parts.append(ratio)
        parts.append(logratio)
        # AC/DC ratio between segments (pulse-ox style)
        ai = ad[i] + 1e-8
        aj = ad[j] + 1e-8
        acdc_ratio = (ai / aj).astype(np.float32)
        parts.append(acdc_ratio)

    return np.concatenate(parts).astype(np.float32)


def build_feature_names() -> list[str]:
    names = []
    for s in range(3):
        for ch in CHANNELS:
            for stat in STATS_PER_CH:
                names.append(f"seg{s}_{ch}_{stat}")
    for pair in ["01", "02", "12"]:
        for ch in CHANNELS:
            names.append(f"r{pair}_{ch}_dc_ratio")
        for ch in CHANNELS:
            names.append(f"r{pair}_{ch}_dc_logratio")
        for ch in CHANNELS:
            names.append(f"r{pair}_{ch}_acdc_ratio")
    return names


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("STAGE 3 — Time-series Feature Extraction")
    print("=" * 60)

    index = pd.read_csv(INDEX_CSV)
    segs  = pd.read_csv(SEGMENTS_CSV)
    df    = index[index["file_ok"]].merge(
                segs[["video_id", "s0_start", "s0_end",
                       "s1_start", "s1_end",
                       "s2_start", "s2_end"]],
                on="video_id", how="inner")

    print(f"\nVideos : {len(df)}  "
          f"(6s={len(df[df.protocol=='6s'])}  "
          f"30s={len(df[df.protocol=='30s'])})")
    print(f"Train  : {len(df[df.split=='train'])}  "
          f"Test   : {len(df[df.split=='test'])}")

    feat_names = build_feature_names()
    n_feat = len(feat_names)
    n_per_seg = len(CHANNELS) * len(STATS_PER_CH)
    n_cross   = 3 * len(CHANNELS) * 3
    print(f"\nFeatures : {n_feat}  "
          f"= {n_per_seg}×3 per-seg + {n_cross} cross-seg")

    records = []
    errors  = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        bounds = [
            (int(row["s0_start"]), int(row["s0_end"])),
            (int(row["s1_start"]), int(row["s1_end"])),
            (int(row["s2_start"]), int(row["s2_end"])),
        ]
        try:
            fv  = extract_video(row["video_path"], bounds)
            rec = {
                "video_id": row["video_id"],
                "hb_value": row["hb_value"],
                "split":    row["split"],
                "protocol": row["protocol"],
            }
            # Carry subject_id (for leakage-safe, subject-level CV) and any
            # available demographics (age / sex) — strong predictors per the
            # literature (Ni et al., Front. Physiol. 2026: +~14% accuracy).
            for opt_col in ("subject_id", "age", "sex", "gender"):
                if opt_col in row.index and pd.notna(row[opt_col]):
                    rec[opt_col] = row[opt_col]
            for name, val in zip(feat_names, fv):
                rec[name] = val
            records.append(rec)
        except Exception as e:
            errors.append((row["video_id"], str(e)))

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for vid, err in errors[:5]:
            print(f"    {vid}: {err}")

    out = pd.DataFrame(records)
    nan_count = out[feat_names].isnull().sum().sum()
    print(f"\n  NaN in features: {nan_count}")

    train_df = out[out["split"] == "train"]
    test_df  = out[out["split"] == "test"]

    out.to_csv(FEATURES_CSV, index=False)
    train_df.to_csv(FEATURES_DIR / "features_train.csv", index=False)
    test_df.to_csv(FEATURES_DIR  / "features_test.csv",  index=False)

    print(f"\n── Complete ──")
    print(f"  Total: {len(out)}  Train: {len(train_df)}  Test: {len(test_df)}")

    corrs = train_df[feat_names].corrwith(train_df["hb_value"]).abs()
    print("\n── Top 10 |corr| with HB (train) ──")
    for name, c in corrs.sort_values(ascending=False).head(10).items():
        print(f"  {name:<50s}  r={c:.4f}")

    print(f"\n  → {FEATURES_CSV}")
    print("\nStage 3 complete.\n")


if __name__ == "__main__":
    main()
