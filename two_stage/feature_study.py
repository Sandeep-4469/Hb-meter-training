"""
Feature Study: minimal channel-mean representation
===================================================
Channels: R, G, B, H, S, V, L, A, Gray  (9 channels)
Features: for each segment, mean of per-frame normalised channel sum
          → 3 segments × 9 channels = 27 features

Analysis outputs
----------------
  results/feature_study_corr.csv         — per-feature Pearson r vs HB
  results/feature_study_model.txt        — ridge / RF test performance
  results/feature_study_scatter.png      — scatter plots top-16 features
  results/feature_study_heatmap.png      — correlation heatmap  segs × channels
  results/feature_study_features.csv     — full 27-col feature matrix (train+test)
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import multiprocessing as mp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ROI_Y, ROI_X

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
RESULTS   = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INDEX_CSV    = DATA_DIR / "index.csv"
SEGMENTS_CSV = DATA_DIR / "segments.csv"

# 9 channels the user requested
CHANNELS = ["R", "G", "B", "H", "S", "V", "L", "A", "Gray"]
N_SEGS   = 3
N_FEATS  = N_SEGS * len(CHANNELS)   # 27

ANEMIA_THR = 12.0

# ── helpers ──────────────────────────────────────────────────────────────────

def crop_roi(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    y0, y1 = int(h * ROI_Y[0]), int(h * ROI_Y[1])
    x0, x1 = int(w * ROI_X[0]), int(w * ROI_X[1])
    return bgr[y0:y1, x0:x1]


def frame_channel_means(bgr: np.ndarray) -> dict[str, float]:
    """Return mean ROI pixel value (0–255 scale) for each of 9 channels."""
    roi  = crop_roi(bgr)
    blur = cv2.GaussianBlur(roi, (7, 7), 0)

    rgb  = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab  = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab).astype(np.float32)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY).astype(np.float32)

    return {
        "R":    float(rgb[:, :, 0].mean()),
        "G":    float(rgb[:, :, 1].mean()),
        "B":    float(rgb[:, :, 2].mean()),
        "H":    float(hsv[:, :, 0].mean()),
        "S":    float(hsv[:, :, 1].mean()),
        "V":    float(hsv[:, :, 2].mean()),
        "L":    float(lab[:, :, 0].mean()),
        "A":    float(lab[:, :, 1].mean()),
        "Gray": float(gray.mean()),
    }


def extract_video_features(row: dict) -> dict | None:
    """Read all frames sequentially; return 27 segment-mean features."""
    vpath = row["video_path"]
    segs  = [(row["s0_start"], row["s0_end"]),
             (row["s1_start"], row["s1_end"]),
             (row["s2_start"], row["s2_end"])]

    # collect per-channel time-series for each segment
    seg_accum = [{ch: [] for ch in CHANNELS} for _ in range(N_SEGS)]

    max_fid = max(hi for _, hi in segs) - 1
    all_fids = set()
    for lo, hi in segs:
        all_fids.update(range(lo, hi))

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return None

    fid = 0
    while fid <= max_fid:
        ret, frame = cap.read()
        if not ret:
            break
        if fid in all_fids:
            means = frame_channel_means(frame)
            for s_idx, (lo, hi) in enumerate(segs):
                if lo <= fid < hi:
                    for ch in CHANNELS:
                        seg_accum[s_idx][ch].append(means[ch])
        fid += 1
    cap.release()

    # aggregate: mean over frames per segment per channel
    feat = {"video_id": row["video_id"],
            "hb_value": row["hb_value"],
            "split":    row["split"]}
    for s_idx in range(N_SEGS):
        for ch in CHANNELS:
            vals = seg_accum[s_idx][ch]
            key  = f"s{s_idx}_{ch}"
            feat[key] = float(np.mean(vals)) if vals else np.nan
    return feat


def _worker(args):
    row_dict, = args
    return extract_video_features(row_dict)


# ── extraction ────────────────────────────────────────────────────────────────

def extract_all() -> pd.DataFrame:
    idx  = pd.read_csv(INDEX_CSV)
    segs = pd.read_csv(SEGMENTS_CSV)
    df   = idx.merge(segs, on=["video_id", "protocol"])

    rows = [dict(r) for _, r in df.iterrows()]
    print(f"\nExtracting {len(rows)} videos with {mp.cpu_count()} workers …")

    with mp.Pool(min(16, mp.cpu_count())) as pool:
        results = list(tqdm(pool.imap(_worker, [(r,) for r in rows]),
                            total=len(rows), ncols=80))

    records = [r for r in results if r is not None]
    return pd.DataFrame(records)


# ── analysis ──────────────────────────────────────────────────────────────────

def correlation_analysis(feats: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in feats.columns if c[:2] in ("s0", "s1", "s2")]
    hb        = feats["hb_value"].values
    rows = []
    for col in feat_cols:
        v = feats[col].astype(float).values
        mask = ~np.isnan(v)
        r, p   = pearsonr(v[mask], hb[mask])
        rho, _ = spearmanr(v[mask], hb[mask])
        rows.append({"feature": col, "pearson_r": r, "spearman_rho": rho, "p_value": p,
                     "abs_r": abs(r)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


def scatter_plots(feats: pd.DataFrame, corr_df: pd.DataFrame):
    top16 = corr_df.head(16)["feature"].tolist()
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.ravel()
    hb = feats["hb_value"].values
    colors = np.where(hb < ANEMIA_THR, "tomato", "steelblue")
    for i, col in enumerate(top16):
        ax = axes[i]
        v = feats[col].values
        ax.scatter(v, hb, c=colors, alpha=0.4, s=12)
        r = corr_df.loc[corr_df["feature"] == col, "pearson_r"].values[0]
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("HB (g/dL)", fontsize=8)
        ax.set_title(f"r = {r:.3f}", fontsize=9)
        # best-fit line
        mask = ~np.isnan(v)
        m, b = np.polyfit(v[mask], hb[mask], 1)
        xs = np.array([v[mask].min(), v[mask].max()])
        ax.plot(xs, m * xs + b, "k--", lw=1)
    fig.suptitle("Top-16 features (red=anaemic, blue=normal)", fontsize=12)
    plt.tight_layout()
    out = RESULTS / "feature_study_scatter.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out}")


def correlation_heatmap(corr_df: pd.DataFrame):
    # reshape into segs × channels matrix
    mat = np.zeros((N_SEGS, len(CHANNELS)))
    for _, row in corr_df.iterrows():
        parts = row["feature"].split("_", 1)   # s0_R → ['s0', 'R']
        s_idx = int(parts[0][1])
        ch    = parts[1]
        if ch in CHANNELS:
            mat[s_idx, CHANNELS.index(ch)] = row["pearson_r"]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(len(CHANNELS))); ax.set_xticklabels(CHANNELS)
    ax.set_yticks(range(N_SEGS)); ax.set_yticklabels(["Seg0 (RED)", "Seg1 (ORANGE)", "Seg2 (YELLOW)"])
    for i in range(N_SEGS):
        for j in range(len(CHANNELS)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r with HB")
    ax.set_title("Feature–HB correlation: segment × channel")
    plt.tight_layout()
    out = RESULTS / "feature_study_heatmap.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out}")


def within2(true, pred, thr=2.0):
    return float(np.mean(np.abs(true - pred) <= thr)) * 100


def model_evaluation(feats: pd.DataFrame) -> str:
    feat_cols = [c for c in feats.columns if c[:2] in ("s0", "s1", "s2")]
    train_df  = feats[feats["split"] == "train"].copy()
    test_df   = feats[feats["split"] == "test"].copy()

    X_train = train_df[feat_cols].fillna(0).values
    y_train = train_df["hb_value"].values
    X_test  = test_df[feat_cols].fillna(0).values
    y_test  = test_df["hb_value"].values
    groups  = train_df["video_id"].values   # each video is its own group

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lines = []
    lines.append("=" * 65)
    lines.append("  FEATURE STUDY — Model Comparison (27 segment-mean features)")
    lines.append("=" * 65)
    lines.append(f"  Train: {len(X_train)}   Test: {len(X_test)}   Features: {len(feat_cols)}")

    models = [
        ("Ridge (alpha=1)",    Ridge(alpha=1.0)),
        ("Ridge (alpha=10)",   Ridge(alpha=10.0)),
        ("RF (100 trees)",     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GBR (200 trees)",    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                         max_depth=3, random_state=42)),
    ]

    # 5-fold OOF CV on train
    kf = GroupKFold(n_splits=5)

    for name, mdl in models:
        if "Ridge" in name:
            Xtr, Xte = X_tr_sc, X_te_sc
        else:
            Xtr, Xte = X_train, X_test

        # OOF
        oof = np.zeros(len(y_train))
        for tr_idx, va_idx in kf.split(Xtr, y_train, groups):
            mdl.fit(Xtr[tr_idx], y_train[tr_idx])
            oof[va_idx] = mdl.predict(Xtr[va_idx])

        # retrain on all train, evaluate test
        mdl.fit(Xtr, y_train)
        pred = mdl.predict(Xte)

        anemia_true = (y_test < ANEMIA_THR).astype(int)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(anemia_true, -pred)   # lower pred → more likely anemic

        lines.append("")
        lines.append(f"  ── {name}")
        lines.append(f"  OOF : MAE={mean_absolute_error(y_train, oof):.3f}  "
                     f"RMSE={mean_squared_error(y_train, oof)**0.5:.3f}  "
                     f"W±2={within2(y_train, oof):.1f}%")
        lines.append(f"  Test: MAE={mean_absolute_error(y_test, pred):.3f}  "
                     f"RMSE={mean_squared_error(y_test, pred)**0.5:.3f}  "
                     f"W±2={within2(y_test, pred):.1f}%  "
                     f"AUC={auc:.3f}")
        lines.append(f"        PredStd={pred.std():.3f}  TrueStd={y_test.std():.3f}  "
                     f"Ratio={pred.std()/y_test.std():.3f}")

    lines.append("")
    lines.append("  ── Baseline: predict training mean always")
    pred_mean = np.full(len(y_test), y_train.mean())
    lines.append(f"  Test: MAE={mean_absolute_error(y_test, pred_mean):.3f}  "
                 f"RMSE={mean_squared_error(y_test, pred_mean)**0.5:.3f}  "
                 f"W±2={within2(y_test, pred_mean):.1f}%")
    lines.append("")
    lines.append("  ── Reference (1060-feat Ensemble from prior run)")
    lines.append("  Test: MAE=2.093  RMSE=2.681  W±2=55.8%  AUC=0.624  Sens=0.914")
    lines.append("=" * 65)
    return "\n".join(lines)


# ── feature importance (RF) ───────────────────────────────────────────────────

def rf_importance_plot(feats: pd.DataFrame):
    feat_cols = [c for c in feats.columns if c[:2] in ("s0", "s1", "s2")]
    train_df  = feats[feats["split"] == "train"]
    X = train_df[feat_cols].fillna(0).values
    y = train_df["hb_value"].values

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    colors = []
    for feat in imp.index:
        ch = feat.split("_", 1)[1]
        colors.append({
            "R": "#e74c3c", "G": "#2ecc71", "B": "#3498db",
            "H": "#9b59b6", "S": "#f39c12", "V": "#1abc9c",
            "L": "#7f8c8d", "A": "#e67e22", "Gray": "#95a5a6"
        }.get(ch, "#bdc3c7"))
    imp.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("RF feature importance", fontsize=10)
    ax.set_title("Random Forest feature importance\n(27 segment-mean features)", fontsize=11)
    plt.tight_layout()
    out = RESULTS / "feature_study_importance.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feat_csv = RESULTS / "feature_study_features.csv"

    if feat_csv.exists():
        print(f"Loading cached features from {feat_csv}")
        feats = pd.read_csv(feat_csv)
    else:
        feats = extract_all()
        feats.to_csv(feat_csv, index=False)
        print(f"Saved features → {feat_csv}")

    print(f"\nShape: {feats.shape}  |  NaNs: {feats.isna().sum().sum()}")

    # ── correlation analysis
    print("\n── Correlation analysis …")
    corr_df = correlation_analysis(feats)
    corr_csv = RESULTS / "feature_study_corr.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"  Saved → {corr_csv}")

    print("\nTop-15 features by |Pearson r|:")
    print(corr_df[["feature", "pearson_r", "spearman_rho"]].head(15).to_string(index=False))

    print("\nBottom-5 features (weakest signal):")
    print(corr_df[["feature", "pearson_r"]].tail(5).to_string(index=False))

    # ── plots
    print("\n── Generating plots …")
    scatter_plots(feats, corr_df)
    correlation_heatmap(corr_df)
    rf_importance_plot(feats)

    # ── model evaluation
    print("\n── Model evaluation …")
    report = model_evaluation(feats)
    print(report)
    report_path = RESULTS / "feature_study_model.txt"
    report_path.write_text(report)
    print(f"\n  Saved → {report_path}")

    print("\nDone.")
