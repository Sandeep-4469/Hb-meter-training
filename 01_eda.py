"""
Stage 1 — EDA & data inventory.

Reads the raw feature_index.csv, tags each video with its protocol (6s / 30s),
verifies video files exist, prints summary statistics, and writes data/index.csv.
Also saves a 3×3 grid of sample frames from one 6s and one 30s video.

Run:
    python 01_eda.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, RESULTS_DIR, RAW_INDEX_CSV, VIDEO_ROOT,
    INDEX_CSV, PROTO_6S_MAX_DURATION, PROTO_30S_MIN_DURATION,
    ROI_Y, ROI_X,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def tag_protocol(duration: float) -> str:
    if duration <= PROTO_6S_MAX_DURATION:
        return "6s"
    if duration >= PROTO_30S_MIN_DURATION:
        return "30s"
    return "other"


def file_exists(path: str) -> bool:
    return Path(path).exists()


def read_frames(path: str, max_frames: int = 9) -> list[np.ndarray]:
    """Return up to max_frames evenly-spaced BGR frames."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if ret:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def save_sample_grid(path: str, title: str, out_path: Path) -> None:
    frames = read_frames(path, max_frames=9)
    if not frames:
        return
    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    fig.suptitle(title, fontsize=11)
    for i, ax in enumerate(axes.flat):
        if i < len(frames):
            ax.imshow(frames[i])
            # draw ROI box
            h, w = frames[i].shape[:2]
            y0, y1 = int(h * ROI_Y[0]), int(h * ROI_Y[1])
            x0, x1 = int(w * ROI_X[0]), int(w * ROI_X[1])
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  edgecolor="lime", linewidth=2, fill=False)
            ax.add_patch(rect)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 1 — EDA & Data Inventory")
    print("=" * 60)

    # ── load raw index ────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_INDEX_CSV)
    print(f"\nRaw index: {len(df)} rows")

    # ── tag protocol ─────────────────────────────────────────────────────────
    df["protocol"] = df["duration_sec"].apply(tag_protocol)
    df["file_ok"]  = df["video_path"].apply(file_exists)

    # ── summary stats ─────────────────────────────────────────────────────────
    print("\n── Protocol breakdown ──")
    proto_split = df.groupby(["protocol", "split"]).size().unstack(fill_value=0)
    print(proto_split.to_string())

    print("\n── File availability ──")
    print(df.groupby(["protocol", "file_ok"]).size().unstack(fill_value=0).to_string())

    missing = df[~df["file_ok"]]
    if len(missing):
        print(f"  WARNING: {len(missing)} video files not found on disk")

    print("\n── HB distribution ──")
    for proto in ["6s", "30s", "other"]:
        sub = df[df["protocol"] == proto]["hb_value"]
        print(f"  {proto:5s}: n={len(sub):3d}  mean={sub.mean():.2f}  "
              f"std={sub.std():.2f}  min={sub.min():.1f}  max={sub.max():.1f}")

    print("\n── Train / test split ──")
    print(df.groupby(["split", "protocol"]).size().unstack(fill_value=0).to_string())

    # ── HB histogram ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, proto, color in zip(axes, ["6s", "30s"], ["steelblue", "coral"]):
        sub = df[df["protocol"] == proto]["hb_value"]
        ax.hist(sub, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(12.0, color="red", linestyle="--", linewidth=1.5, label="WHO 12 g/dL")
        ax.set_title(f"{proto} protocol (n={len(sub)})")
        ax.set_xlabel("Hb (g/dL)")
        ax.set_ylabel("Count")
        ax.legend()
    plt.suptitle("Haemoglobin distribution by recording protocol")
    plt.tight_layout()
    hb_plot = RESULTS_DIR / "01_hb_distribution.png"
    plt.savefig(hb_plot, dpi=120)
    plt.close()
    print(f"\n  HB histogram → {hb_plot}")

    # ── per-frame channel means in one 6s and one 30s video ─────────────────
    print("\n── Per-frame channel signal preview ──")
    for proto, label, color in [("6s", "6 s video", "steelblue"),
                                 ("30s", "30 s video", "coral")]:
        row = df[(df["protocol"] == proto) & df["file_ok"]].iloc[0]
        cap = cv2.VideoCapture(row["video_path"])
        means_r, means_g, means_b = [], [], []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            roi = rgb[int(h * ROI_Y[0]):int(h * ROI_Y[1]),
                      int(w * ROI_X[0]):int(w * ROI_X[1])]
            means_r.append(roi[:, :, 0].mean())
            means_g.append(roi[:, :, 1].mean())
            means_b.append(roi[:, :, 2].mean())
        cap.release()

        n = len(means_r)
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(x, means_r, color="red",   label="R", linewidth=1.2)
        ax.plot(x, means_g, color="green", label="G", linewidth=1.2)
        ax.plot(x, means_b, color="blue",  label="B", linewidth=1.2)
        # mark LED segment boundaries (equal thirds)
        for boundary in [n // 3, 2 * n // 3]:
            ax.axvline(boundary, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"{label} — {Path(row['video_path']).name}  HB={row['hb_value']}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean pixel value (ROI)")
        ax.legend()
        ax.set_ylim(0, 270)
        plt.tight_layout()
        sig_plot = RESULTS_DIR / f"01_signal_{proto.replace('s','s')}.png"
        plt.savefig(sig_plot, dpi=120)
        plt.close()
        print(f"  {label} signal plot → {sig_plot}")

    # ── sample frame grids ────────────────────────────────────────────────────
    print("\n── Saving sample frame grids ──")
    for proto in ["6s", "30s"]:
        row = df[(df["protocol"] == proto) & df["file_ok"]].iloc[0]
        out = RESULTS_DIR / f"01_frames_{proto}.png"
        save_sample_grid(
            row["video_path"],
            f"{proto} — {Path(row['video_path']).name}  HB={row['hb_value']} g/dL",
            out,
        )

    # ── write index.csv ───────────────────────────────────────────────────────
    keep_cols = [
        "video_id", "video_path", "subject_id", "subject_name",
        "hb_value", "split", "duration_sec", "protocol", "file_ok",
    ]
    out_df = df[keep_cols].copy()
    out_df.to_csv(INDEX_CSV, index=False)
    print(f"\n  index.csv written → {INDEX_CSV}")
    print(f"  rows: {len(out_df)}  (6s={len(out_df[out_df.protocol=='6s'])}  "
          f"30s={len(out_df[out_df.protocol=='30s'])}  "
          f"other={len(out_df[out_df.protocol=='other'])})")

    print("\nStage 1 complete.\n")


if __name__ == "__main__":
    main()
