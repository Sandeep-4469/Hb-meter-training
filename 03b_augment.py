"""
Stage 3b — Augmentation.

Takes the raw signals saved by Stage 3 and produces an augmented feature matrix
by applying four techniques. All augmentations are applied ONLY to training rows.
The test set is never touched.

Techniques applied to each training video's signal:
  1. Temporal windowing   — sample multiple WINDOW_FRAMES-frame sub-windows
                            from each segment (largest gain, especially for 30s)
  2. Global brightness    — multiply all channels uniformly by factor in
                            [AUG_BRIGHTNESS_RANGE]  (camera gain simulation)
  3. Gaussian noise       — add per-frame noise σ = AUG_NOISE_SIGMA_FRAC × mean
  4. Temporal jitter      — shift segment boundaries by ±AUG_JITTER_FRAMES

For each training video the pipeline produces:
  • all temporal windows                    (×AUG_TEMPORAL_WINDOWS)
  • AUG_COPIES_PER_VIDEO more copies with brightness + noise applied

Outputs:
  outputs/features/features_augmented.csv  — train rows only (augmented + originals)
  outputs/features/features_test.csv       — test rows only  (no augmentation)

Run:
    python 03b_augment.py
"""

from __future__ import annotations

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SIGNALS_DIR, FEATURES_DIR,
    WINDOW_FRAMES,
    AUG_TEMPORAL_WINDOWS_6S, AUG_TEMPORAL_WINDOWS_30S,
    AUG_BRIGHTNESS_RANGE,
    AUG_NOISE_SIGMA_FRAC,
    AUG_JITTER_FRAMES,
    AUG_COPIES_PER_VIDEO,
    HR_BAND_LOW, HR_BAND_HIGH,
    CHANNELS, RANDOM_SEED,
)
from utils.signal import extract_all_features, build_feature_names

FEATURES_CSV       = FEATURES_DIR / "features.csv"
FEATURES_AUG_CSV   = FEATURES_DIR / "features_augmented.csv"
FEATURES_TEST_CSV  = FEATURES_DIR / "features_test.csv"

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

FEAT_NAMES = build_feature_names(CHANNELS)


# ── augmentation helpers ──────────────────────────────────────────────────────

def _window_segments(segs: list[np.ndarray], n_windows: int) -> list[list[np.ndarray]]:
    """
    Sample n_windows sub-windows of WINDOW_FRAMES from each segment.
    Returns a list of length n_windows, each element is a list of 3 sub-segs.
    """
    windows = []
    min_seg_len = min(len(s) for s in segs)
    w = min(WINDOW_FRAMES, min_seg_len)

    max_starts = [max(0, len(s) - w) for s in segs]
    # evenly-spaced starts for deterministic coverage
    for i in range(n_windows):
        windowed = []
        for seg, max_start in zip(segs, max_starts):
            if max_start == 0:
                windowed.append(seg[:w])
            else:
                # spread starts evenly
                start = int(i * max_start / max(n_windows - 1, 1))
                windowed.append(seg[start: start + w])
        windows.append(windowed)
    return windows


def _apply_brightness(segs: list[np.ndarray]) -> list[np.ndarray]:
    """Multiply ALL channels in ALL segments by the same random scale."""
    scale = rng.uniform(*AUG_BRIGHTNESS_RANGE)
    return [s * scale for s in segs]


def _apply_noise(segs: list[np.ndarray]) -> list[np.ndarray]:
    """Add Gaussian noise proportional to each segment's mean."""
    out = []
    for s in segs:
        sigma = AUG_NOISE_SIGMA_FRAC * (s.mean(axis=0) + 1e-8)  # per-channel sigma
        noise = rng.normal(0, sigma, size=s.shape).astype(np.float32)
        out.append(np.clip(s + noise, 0, None))
    return out


def _apply_jitter(segs: list[np.ndarray]) -> list[np.ndarray]:
    """Randomly trim a few frames from the start/end of each segment."""
    out = []
    for s in segs:
        j = rng.integers(-AUG_JITTER_FRAMES, AUG_JITTER_FRAMES + 1)
        start = max(0, j)
        end   = len(s) + min(0, j)
        out.append(s[start:end] if end > start + 5 else s)
    return out


def featurise(segs: list[np.ndarray], fps: float) -> np.ndarray | None:
    try:
        return extract_all_features(segs, fps, HR_BAND_LOW, HR_BAND_HIGH)
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("STAGE 3b — Augmentation")
    print("=" * 60)

    base_df = pd.read_csv(FEATURES_CSV)
    meta_cols = ["video_id", "hb_value", "split", "protocol", "use_r"]

    # separate test — never augmented
    test_df = base_df[base_df["split"] == "test"].copy()
    test_df.to_csv(FEATURES_TEST_CSV, index=False)
    print(f"\n  test rows (no aug): {len(test_df)}")
    print(f"  test.csv → {FEATURES_TEST_CSV}")

    train_df = base_df[base_df["split"] == "train"].reset_index(drop=True)
    print(f"  train rows (before aug): {len(train_df)}")

    aug_rows = []
    skipped  = 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        video_id = row["video_id"]
        protocol = row["protocol"]
        hb       = row["hb_value"]

        signal_path = SIGNALS_DIR / f"{video_id}.npy"
        if not signal_path.exists():
            skipped += 1
            continue

        data = np.load(signal_path, allow_pickle=True).item()
        full_signal = data["signal"]       # (N, 6)
        fps         = float(data["fps"])
        bounds      = data["seg_bounds"]   # [(s,e), (s,e), (s,e)]

        # raw segments (not yet normalised — normalisation is inside extract_all_features)
        segs_raw = [full_signal[b[0]:b[1]] for b in bounds]

        if any(len(s) < 10 for s in segs_raw):
            skipped += 1
            continue

        n_wins = AUG_TEMPORAL_WINDOWS_6S if protocol == "6s" else AUG_TEMPORAL_WINDOWS_30S

        # ── 1. temporal windows ───────────────────────────────────────────────
        for windowed in _window_segments(segs_raw, n_wins):
            fv = featurise(windowed, fps)
            if fv is not None:
                aug_rows.append({
                    "video_id": video_id, "hb_value": hb,
                    "split": "train", "protocol": protocol, "use_r": row["use_r"],
                    "aug_type": "temporal_window",
                    **dict(zip(FEAT_NAMES, fv)),
                })

        # ── 2. brightness + noise + jitter copies ────────────────────────────
        for _ in range(AUG_COPIES_PER_VIDEO):
            segs_aug = _apply_brightness(segs_raw)
            segs_aug = _apply_noise(segs_aug)
            segs_aug = _apply_jitter(segs_aug)
            fv = featurise(segs_aug, fps)
            if fv is not None:
                aug_rows.append({
                    "video_id": video_id, "hb_value": hb,
                    "split": "train", "protocol": protocol, "use_r": row["use_r"],
                    "aug_type": "perturbation",
                    **dict(zip(FEAT_NAMES, fv)),
                })

    # ── assemble ──────────────────────────────────────────────────────────────
    aug_df = pd.DataFrame(aug_rows)

    # add aug_type column to original train rows (they count as original)
    train_with_type = train_df.copy()
    train_with_type["aug_type"] = "original"
    aug_df = pd.concat([train_with_type, aug_df], ignore_index=True)
    aug_df.to_csv(FEATURES_AUG_CSV, index=False)

    print(f"\n── Augmentation summary ──")
    print(aug_df["aug_type"].value_counts().to_string())
    print(f"\n  total train rows after augmentation: {len(aug_df)}")
    print(f"  (×{len(aug_df)/max(len(train_df),1):.1f} expansion)")
    print(f"  skipped (no signal file): {skipped}")
    print(f"\n  features_augmented.csv → {FEATURES_AUG_CSV}")
    print("\nStage 3b complete.\n")


if __name__ == "__main__":
    main()
