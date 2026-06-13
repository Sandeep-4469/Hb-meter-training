"""
Low-level video helpers: frame reading, ROI crop, per-frame channel extraction.
"""

from __future__ import annotations

import cv2
import numpy as np


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    return cap


def video_meta(path: str) -> dict:
    cap = open_video(path)
    meta = {
        "fps":      cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_sec"] = meta["n_frames"] / meta["fps"] if meta["fps"] > 0 else 0.0
    cap.release()
    return meta


def read_all_frames_rgb(path: str) -> np.ndarray:
    """Return all frames as uint8 array of shape (N, H, W, 3) in RGB order."""
    cap = open_video(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def crop_roi(frame_rgb: np.ndarray,
             y_frac: tuple[float, float],
             x_frac: tuple[float, float]) -> np.ndarray:
    """Center-crop a single frame by fractional coordinates."""
    h, w = frame_rgb.shape[:2]
    y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
    x0, x1 = int(w * x_frac[0]), int(w * x_frac[1])
    return frame_rgb[y0:y1, x0:x1]


def frame_channel_means(roi_rgb: np.ndarray) -> dict[str, float]:
    """
    Given an RGB ROI array (H, W, 3), return a dict of channel means:
      R, G, B            — raw 0-255 means
      RC                 — R chromaticity: R / (R+G+B)
      GC                 — G chromaticity: G / (R+G+B)
      RGN                — normalised redness: (R-G) / (R+G)  in [-1, 1]
    """
    r = roi_rgb[:, :, 0].astype(np.float32)
    g = roi_rgb[:, :, 1].astype(np.float32)
    b = roi_rgb[:, :, 2].astype(np.float32)
    total = r + g + b + 1e-6
    rg    = r + g + 1e-6

    return {
        "R":   float(r.mean()),
        "G":   float(g.mean()),
        "B":   float(b.mean()),
        "RC":  float((r / total).mean()),
        "GC":  float((g / total).mean()),
        "RGN": float(((r - g) / rg).mean()),
    }


def extract_signal(frames_rgb: np.ndarray,
                   y_frac: tuple[float, float],
                   x_frac: tuple[float, float],
                   min_brightness: float = 5.0,
                   max_brightness: float = 253.0) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame in frames_rgb (N, H, W, 3):
      1. Crop ROI
      2. Compute per-frame channel means
      3. Drop frames that are too dark or too bright (bad frames)

    Returns:
      signal  — float32 array (M, 6)  M<=N  columns=[R,G,B,RC,GC,RGN]
      mask    — bool array (N,) True = frame kept
    """
    channels = ["R", "G", "B", "RC", "GC", "RGN"]
    rows = []
    mask = []
    for frame in frames_rgb:
        roi = crop_roi(frame, y_frac, x_frac)
        brightness = roi.mean()
        if brightness < min_brightness or brightness > max_brightness:
            mask.append(False)
            continue
        ch = frame_channel_means(roi)
        rows.append([ch[c] for c in channels])
        mask.append(True)

    if not rows:
        raise ValueError("All frames were rejected by quality filter")

    signal = np.array(rows, dtype=np.float32)
    return signal, np.array(mask, dtype=bool)
