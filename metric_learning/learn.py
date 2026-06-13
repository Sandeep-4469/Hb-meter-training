"""
Step 2 — Metric learning (LMNN).

Goal: learn a linear transformation W such that in the projected space
W×x, patients with similar HB values are close and patients with
different HB values are far apart.

Pipeline:
  1. Load features from Step 1
  2. Per-bin outlier removal (k=3, vote>20%)
  3. Bin HB into 1 g/dL labels for LMNN supervision
  4. Fit LMNN on training set
  5. Compare discriminability before vs after transformation
  6. Save W matrix and projected train/test features

Outputs:
  results/W_transform.npy          — learned projection matrix (n_components × n_features)
  results/train_projected.csv
  results/test_projected.csv
  results/discriminability_comparison.png

Run:
    python metric_learn.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).parent / "results"

META = {"video_id", "hb_value", "split", "protocol"}


# ── outlier removal ───────────────────────────────────────────────────────────

def per_bin_outlier_removal(df, feat_cols, k=3.0, vote_thresh=0.20):
    bin_edges  = np.arange(int(df["hb_value"].min()),
                            int(df["hb_value"].max()) + 2)
    bin_labels = [f"{int(e)}–{int(e)+1}" for e in bin_edges[:-1]]
    df = df.copy()
    df["hb_bin"] = pd.cut(df["hb_value"], bins=bin_edges,
                           labels=bin_labels, right=False)

    outlier_count = pd.Series(0, index=df.index, dtype=int)
    n_checked     = pd.Series(0, index=df.index, dtype=int)

    for _, grp in df.groupby("hb_bin", observed=True):
        if len(grp) < 4:
            continue
        for col in feat_cols:
            q1, q3 = grp[col].quantile(0.25), grp[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            is_out = ~grp[col].between(q1 - k*iqr, q3 + k*iqr)
            outlier_count[grp.index] += is_out.astype(int)
            n_checked[grp.index]     += 1

    frac_out = outlier_count / n_checked.replace(0, 1)
    keep = frac_out <= vote_thresh
    n_before = len(df)
    df_clean = df[keep].drop(columns=["hb_bin"])
    print(f"  Outlier removal: {n_before} → {len(df_clean)} "
          f"({n_before - len(df_clean)} removed, {len(df_clean)/n_before:.0%} kept)")
    return df_clean


# ── discriminability helper ───────────────────────────────────────────────────

def compute_discriminability(X, y_hb, label=""):
    """Average |Pearson r| and average F-score across all dimensions."""
    abs_r, f_scores = [], []
    bin_edges  = np.arange(int(y_hb.min()), int(y_hb.max()) + 2)
    bin_labels = [f"{int(e)}–{int(e)+1}" for e in bin_edges[:-1]]
    hb_bin = pd.cut(y_hb, bins=bin_edges, labels=bin_labels, right=False)

    for d in range(X.shape[1]):
        col = X[:, d]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        r, _ = sp_stats.pearsonr(col[valid], y_hb[valid])
        abs_r.append(abs(r))

        groups = [col[np.array(hb_bin == b) & valid]
                  for b in bin_labels
                  if (hb_bin == b).sum() >= 2]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                f, _ = sp_stats.f_oneway(*groups)
                if not np.isnan(f):
                    f_scores.append(f)
            except Exception:
                pass

    mean_r = np.mean(abs_r)   if abs_r    else 0.0
    mean_f = np.mean(f_scores) if f_scores else 0.0
    n_good = sum(r > 0.20 for r in abs_r)
    print(f"  {label:<30s}  mean|r|={mean_r:.4f}  "
          f"meanF={mean_f:.2f}  dims|r|>0.20: {n_good}/{X.shape[1]}")
    return mean_r, mean_f, abs_r, f_scores


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("STEP 2 — LMNN Metric Learning")
    print("=" * 55)

    train_raw = pd.read_csv(OUT_DIR / "features_train.csv")
    test_raw  = pd.read_csv(OUT_DIR / "features_test.csv")

    feat_cols = [c for c in train_raw.columns if c not in META]
    print(f"\nFeatures: {len(feat_cols)}  "
          f"Train: {len(train_raw)}  Test: {len(test_raw)}")

    # ── outlier removal ────────────────────────────────────────────────────────
    print("\n── Per-bin outlier removal ──")
    train_df = per_bin_outlier_removal(train_raw, feat_cols)
    # test set: no removal (evaluate on real distribution)

    # ── preprocess ────────────────────────────────────────────────────────────
    imp   = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
                  imp.fit_transform(train_df[feat_cols].values.astype(np.float32)))
    y_train = train_df["hb_value"].values

    X_test  = scaler.transform(
                  imp.transform(test_raw[feat_cols].values.astype(np.float32)))
    y_test  = test_raw["hb_value"].values

    # HB bin labels for LMNN (1 g/dL bins)
    y_train_bins = np.floor(y_train).astype(int)
    print(f"\nBins in training: {np.unique(y_train_bins)}")

    # ── baseline discriminability ──────────────────────────────────────────────
    print("\n── Discriminability BEFORE metric learning ──")
    r_before, f_before, abs_r_before, f_before_list = compute_discriminability(
        X_train, y_train, "Raw scaled features")

    # ── LMNN ──────────────────────────────────────────────────────────────────
    print("\n── Fitting LMNN ──")
    try:
        from metric_learn import LMNN
        from collections import Counter
        n_comp  = min(30, X_train.shape[1])
        k_lmnn  = 3
        bin_counts = Counter(y_train_bins)
        lmnn_mask  = np.array([bin_counts[b] >= k_lmnn + 1 for b in y_train_bins])
        print(f"  Training on {lmnn_mask.sum()}/{len(lmnn_mask)} samples "
              f"(bins with ≥{k_lmnn+1} members)")
        lmnn = LMNN(k=k_lmnn, n_components=n_comp, max_iter=500,
                    convergence_tol=1e-5, verbose=False)
        lmnn.fit(X_train[lmnn_mask], y_train_bins[lmnn_mask])
        X_train_proj = lmnn.transform(X_train)
        X_test_proj  = lmnn.transform(X_test)
        W = lmnn.components_
        method = "LMNN"
        print(f"  LMNN done — projection shape: {W.shape}")
    except Exception as e:
        print(f"  LMNN failed ({e}), falling back to NCA")
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        n_comp = min(30, X_train.shape[1])
        nca = NeighborhoodComponentsAnalysis(
                  n_components=n_comp, max_iter=500,
                  random_state=42, verbose=0)
        nca.fit(X_train, y_train_bins)
        X_train_proj = nca.transform(X_train)
        X_test_proj  = nca.transform(X_test)
        W = nca.components_
        method = "NCA"
        print(f"  NCA done — projection shape: {W.shape}")
    np.save(OUT_DIR / "W_transform.npy", W)

    # ── discriminability AFTER ─────────────────────────────────────────────────
    print("\n── Discriminability AFTER metric learning ──")
    r_after, f_after, abs_r_after, f_after_list = compute_discriminability(
        X_train_proj, y_train, f"{method} projected ({n_comp}D)")

    # ── comparison plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(abs_r_before, bins=20, alpha=0.6, label=f"Before (mean={r_before:.3f})",
                 color="steelblue", edgecolor="white")
    axes[0].hist(abs_r_after,  bins=20, alpha=0.6, label=f"After  (mean={r_after:.3f})",
                 color="darkorange", edgecolor="white")
    axes[0].axvline(0.20, color="red", linestyle="--", linewidth=1, label="|r|=0.20")
    axes[0].set_xlabel("|Pearson r| with HB", fontsize=10)
    axes[0].set_ylabel("Number of dimensions", fontsize=10)
    axes[0].set_title("Distribution of |r| — Before vs After", fontsize=11)
    axes[0].legend()

    axes[1].hist(f_before_list, bins=20, alpha=0.6, label=f"Before (mean={f_before:.2f})",
                 color="steelblue", edgecolor="white")
    axes[1].hist(f_after_list,  bins=20, alpha=0.6, label=f"After  (mean={f_after:.2f})",
                 color="darkorange", edgecolor="white")
    axes[1].axvline(3.0, color="red", linestyle="--", linewidth=1, label="F=3")
    axes[1].set_xlabel("ANOVA F-score", fontsize=10)
    axes[1].set_ylabel("Number of dimensions", fontsize=10)
    axes[1].set_title("Distribution of F-scores — Before vs After", fontsize=11)
    axes[1].legend()

    plt.suptitle(f"{method} Metric Learning — Discriminability Improvement",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "discriminability_comparison.png", dpi=140)
    plt.close()
    print(f"  saved → {OUT_DIR / 'discriminability_comparison.png'}")

    # ── save projected features ────────────────────────────────────────────────
    dim_names = [f"dim{i:02d}" for i in range(n_comp)]

    train_proj_df = train_df[list(META)].copy().reset_index(drop=True)
    for i, name in enumerate(dim_names):
        train_proj_df[name] = X_train_proj[:, i]
    train_proj_df.to_csv(OUT_DIR / "train_projected.csv", index=False)

    test_proj_df = test_raw[list(META)].copy().reset_index(drop=True)
    for i, name in enumerate(dim_names):
        test_proj_df[name] = X_test_proj[:, i]
    test_proj_df.to_csv(OUT_DIR / "test_projected.csv", index=False)

    print(f"\n  W matrix  → {OUT_DIR / 'W_transform.npy'}")
    print(f"  Train     → {OUT_DIR / 'train_projected.csv'}")
    print(f"  Test      → {OUT_DIR / 'test_projected.csv'}")

    # ── UMAP/PCA visualisation ─────────────────────────────────────────────────
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        emb_before = pca.fit_transform(X_train)
        emb_after  = PCA(n_components=2).fit_transform(X_train_proj)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, emb, title in [
                (axes[0], emb_before, "PCA of raw features"),
                (axes[1], emb_after,  f"PCA of {method} embedding")]:
            sc = ax.scatter(emb[:,0], emb[:,1], c=y_train,
                            cmap="RdYlGn", s=20, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="HB (g/dL)")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.suptitle("Feature space coloured by HB — Before vs After", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "embedding_visualisation.png", dpi=140)
        plt.close()
        print(f"  saved → {OUT_DIR / 'embedding_visualisation.png'}")
    except Exception:
        pass

    print("\nStep 2 complete.\n")


if __name__ == "__main__":
    main()
