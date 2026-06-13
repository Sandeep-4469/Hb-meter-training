"""
Signal-processing helpers: AC/DC decomposition, cross-segment ratios,
frequency-domain features, intra-video normalization.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


# ── Intra-video normalization ────────────────────────────────────────────────

def normalize_intra_video(segments: list[np.ndarray]) -> list[np.ndarray]:
    """
    Divide every channel by the video-wide mean to cancel multiplicative camera gain.

    Columns layout:  [R, G, B, RC, GC, RGN]  (indices 0-5)
    RGN = (R-G)/(R+G) lives in [-1, +1] — it is already dimensionless and gain-
    invariant by construction.  Normalising it by its own mean is undefined when
    the mean is near zero (different LED phases can push RGN to opposite signs).
    We therefore leave RGN (column 5) untouched; only columns 0-4 are scaled.

    segments : list of 3 arrays each shape (N_i, 6)
    returns  : same structure; columns 0-4 divided by video mean, column 5 unchanged
    """
    combined = np.concatenate(segments, axis=0)          # (total_frames, 6)
    video_mean = combined.mean(axis=0) + 1e-8            # (6,)
    # do NOT normalise RGN (col 5) — already dimensionless, mean can be ~0
    video_mean[5] = 1.0
    return [seg / video_mean for seg in segments]


# ── DC / AC decomposition ────────────────────────────────────────────────────

def dc_features(seg: np.ndarray) -> np.ndarray:
    """
    DC (slow/mean) features per channel.
    seg shape: (N, C)
    Returns vector of length 5*C: mean, std, p10, p50, p90
    """
    feats = []
    for c in range(seg.shape[1]):
        col = seg[:, c]
        feats.extend([
            col.mean(),
            col.std(),
            np.percentile(col, 10),
            np.percentile(col, 50),
            np.percentile(col, 90),
        ])
    return np.array(feats, dtype=np.float32)


def ac_features(seg: np.ndarray) -> np.ndarray:
    """
    AC (pulsatile) features per channel after linear detrending.
    Returns vector of length 3*C: ac_std, ac_dc_ratio, peak_to_peak
    """
    feats = []
    for c in range(seg.shape[1]):
        col = seg[:, c].astype(np.float64)
        # remove slow trend
        detrended = sp_signal.detrend(col, type="linear")
        dc_mean   = col.mean() + 1e-8
        ac_std    = float(detrended.std())
        feats.extend([
            ac_std,
            ac_std / dc_mean,                           # AC/DC (perfusion index)
            float(detrended.max() - detrended.min()),   # peak-to-peak
        ])
    return np.array(feats, dtype=np.float32)


def freq_features(seg: np.ndarray, fps: float,
                  hr_low: float = 0.5, hr_high: float = 3.0) -> np.ndarray:
    """
    Frequency-domain features per channel (after detrend).
    Returns vector of length 2*C: hr_band_power_fraction, dominant_hr_freq
    """
    feats = []
    n = len(seg)
    for c in range(seg.shape[1]):
        col = sp_signal.detrend(seg[:, c].astype(np.float64), type="linear")
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        power = np.abs(np.fft.rfft(col)) ** 2
        total_power = power.sum() + 1e-12

        hr_mask = (freqs >= hr_low) & (freqs <= hr_high)
        hr_power = power[hr_mask].sum()
        band_frac = float(hr_power / total_power)

        dom_freq = float(freqs[hr_mask][power[hr_mask].argmax()]) if hr_mask.any() else 0.0
        feats.extend([band_frac, dom_freq])
    return np.array(feats, dtype=np.float32)


# ── Cross-segment ratio features ─────────────────────────────────────────────

def cross_segment_features(segs: list[np.ndarray]) -> np.ndarray:
    """
    Cross-segment features for all 3 pairs (0-1, 0-2, 1-2).

    Channels 0-4  (R, G, B, RC, GC) — always positive:
      • ratio        = mean_i / mean_j          (gain-invariant)
      • log_ratio    = log(mean_i / mean_j)     (Beer-Lambert; positive → real-valued)
      • acdc_ratio   = (AC/DC)_i / (AC/DC)_j

    Channel 5 (RGN = (R-G)/(R+G)) — lives in [-1, +1], CAN be negative:
      • ratio        = mean_i / mean_j          (kept; can be large but finite)
      • diff         = mean_i - mean_j          (replaces log_ratio; always defined)
      • acdc_ratio   = same as above

    Returns vector of length n_pairs * C * 3  (same length as before).
    """
    n_channels = segs[0].shape[1]
    RGN_IDX = 5          # index of the RGN channel
    pairs = [(0, 1), (0, 2), (1, 2)]
    feats = []
    for i, j in pairs:
        mean_i = segs[i].mean(axis=0)          # (C,)  no epsilon yet
        mean_j = segs[j].mean(axis=0)

        # ratio (all channels): safe divide
        ratio = mean_i / (mean_j + np.where(mean_j >= 0, 1e-8, -1e-8))

        # second feature: log-ratio for cols 0-4, difference for RGN (col 5)
        log_or_diff = np.empty(n_channels, dtype=np.float64)
        for c in range(n_channels):
            if c == RGN_IDX:
                log_or_diff[c] = float(mean_i[c] - mean_j[c])
            else:
                mi = max(float(mean_i[c]), 1e-8)
                mj = max(float(mean_j[c]), 1e-8)
                log_or_diff[c] = np.log(mi / mj)

        # AC/DC ratio
        acdc_i = np.array([segs[i][:, c].std() / (abs(segs[i][:, c].mean()) + 1e-8)
                           for c in range(n_channels)])
        acdc_j = np.array([segs[j][:, c].std() / (abs(segs[j][:, c].mean()) + 1e-8)
                           for c in range(n_channels)])
        acdc_ratio = acdc_i / (acdc_j + 1e-8)

        feats.append(ratio.astype(np.float32))
        feats.append(log_or_diff.astype(np.float32))
        feats.append(acdc_ratio.astype(np.float32))

    return np.concatenate(feats).astype(np.float32)


# ── Feature names ─────────────────────────────────────────────────────────────

def build_feature_names(channels: list[str]) -> list[str]:
    names = []
    for s in range(3):
        for c in channels:
            for stat in ["mean", "std", "p10", "p50", "p90"]:
                names.append(f"seg{s}_{c}_dc_{stat}")
        for c in channels:
            for stat in ["ac_std", "acdc_ratio", "peak2peak"]:
                names.append(f"seg{s}_{c}_{stat}")
        for c in channels:
            for stat in ["hr_band_frac", "hr_dom_freq"]:
                names.append(f"seg{s}_{c}_{stat}")

    pairs = ["01", "02", "12"]
    for p in pairs:
        for c in channels:
            names.append(f"r{p}_{c}_ratio")
        for c in channels:
            # RGN uses difference instead of log-ratio (RGN can be negative)
            suffix = "diff" if c == "RGN" else "logratio"
            names.append(f"r{p}_{c}_{suffix}")
        for c in channels:
            names.append(f"r{p}_{c}_acdc_ratio")

    return names


def extract_all_features(segs_raw: list[np.ndarray],
                         fps: float,
                         hr_low: float = 0.5,
                         hr_high: float = 3.0) -> np.ndarray:
    """
    Full feature vector for one video.
    segs_raw: list of 3 arrays (N_i, 6) raw (un-normalised) signal
    Returns 1-D float32 feature vector.
    """
    segs = normalize_intra_video(segs_raw)

    parts = []
    for seg in segs:
        parts.append(dc_features(seg))
        parts.append(ac_features(seg))
        parts.append(freq_features(seg, fps, hr_low, hr_high))

    parts.append(cross_segment_features(segs))
    return np.concatenate(parts).astype(np.float32)
