"""
SmallHbNet — compact 3-branch network designed for the actual data size.

Design principles for 386 train samples with weak signal (max |r|=0.21):
  - SelectKBest(k=30) per branch before network → reduces input from 334 to 30
  - Tiny branch dim (32) + single hidden layer (64) → ~12K total params
  - Gaussian noise augmentation + feature dropout during training
  - StratifiedGroupKFold → balanced anemia/normal in each fold
  - 5-fold ensemble predictions on test set

Architecture:
  Input (30) → Branch (32) → Cat(96) → Hidden(64) → Output(1)
  ≈ 30*32*3 + 96*64 + 64*1 ≈ 9K parameters

Run:
    python train_small.py
"""

from __future__ import annotations
import sys, warnings, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                              mean_absolute_error, mean_squared_error, r2_score,
                              confusion_matrix, classification_report, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import RANDOM_SEED, ACCURACY_BANDS

OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR = ROOT / "metric_learning" / "results"

# ── hyperparameters ────────────────────────────────────────────────────────
ANEMIA_THRESH = 12.0
HB_MIN, HB_MAX = 3.0, 20.0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FEATURES  = 30      # per-branch feature selection
BRANCH_DIM  = 32
HIDDEN_DIM  = 64
DROPOUT     = 0.4
NOISE_STD   = 0.05    # Gaussian noise augmentation (in normalised space)
FEAT_DROP   = 0.05    # probability of zeroing a feature during training
BATCH_SIZE  = 32
MAX_EPOCHS  = 500
PATIENCE    = 60
LR          = 3e-4
WEIGHT_DECAY= 1e-2
N_FOLDS     = 5

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

META = {"video_id", "hb_value", "split", "protocol"}


# ── feature split ──────────────────────────────────────────────────────────

def get_feature_groups(df):
    feat = [c for c in df.columns if c not in META and c != "anemia"]
    seg0   = [c for c in feat if c.startswith("seg0_")]
    seg1   = [c for c in feat if c.startswith("seg1_")]
    seg2   = [c for c in feat if c.startswith("seg2_")]
    shared = [c for c in feat if c not in seg0 + seg1 + seg2]
    return seg0, seg1, seg2, shared


def raw_X(df, seg_cols, shared_cols):
    return df[seg_cols + shared_cols].values.astype(np.float32)


def make_preprocessor(X_train, y_train, k=K_FEATURES):
    """Impute → scale → select top-k by mutual info."""
    imp  = SimpleImputer(strategy="median").fit(X_train)
    X_i  = imp.transform(X_train)
    scl  = StandardScaler().fit(X_i)
    X_s  = scl.transform(X_i)
    sel  = SelectKBest(mutual_info_regression, k=k).fit(X_s, y_train)
    return imp, scl, sel


def apply_preprocessor(X, imp, scl, sel):
    return sel.transform(scl.transform(imp.transform(X))).astype(np.float32)


# ── dataset with augmentation ──────────────────────────────────────────────

class HbDataset(Dataset):
    def __init__(self, X0, X1, X2, y, augment=False):
        self.X0 = torch.tensor(X0, dtype=torch.float32)
        self.X1 = torch.tensor(X1, dtype=torch.float32)
        self.X2 = torch.tensor(X2, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _aug(self, x):
        if not self.augment:
            return x
        # Gaussian noise
        x = x + torch.randn_like(x) * NOISE_STD
        # Feature dropout (randomly zero some features)
        mask = torch.bernoulli(torch.full_like(x, 1 - FEAT_DROP))
        return x * mask

    def __getitem__(self, idx):
        return (self._aug(self.X0[idx]), self._aug(self.X1[idx]),
                self._aug(self.X2[idx]), self.y[idx])


# ── model ──────────────────────────────────────────────────────────────────

class SmallHbNet(nn.Module):
    """
    3-branch network: one branch per LED segment.
    Total params ≈ 12K — matched to 386-sample dataset.
    """
    def __init__(self, input_size, branch_dim=BRANCH_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT, task="regression"):
        super().__init__()
        self.task = task

        def branch(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.red_branch    = branch(input_size, branch_dim)
        self.orange_branch = branch(input_size, branch_dim)
        self.yellow_branch = branch(input_size, branch_dim)

        combined = branch_dim * 3
        self.head = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x0, x1, x2):
        r = self.red_branch(x0)
        o = self.orange_branch(x1)
        y = self.yellow_branch(x2)
        x = torch.cat([r, o, y], dim=1)
        out = self.head(x).squeeze(1)
        if self.task == "regression":
            return HB_MIN + (HB_MAX - HB_MIN) * torch.sigmoid(out)
        return out   # raw logit for BCEWithLogitsLoss


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── train/eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, crit, device):
    model.train()
    total, n = 0.0, 0
    for x0, x1, x2, y in loader:
        x0, x1, x2, y = x0.to(device), x1.to(device), x2.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x0, x1, x2), y).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x0, x1, x2, y in loader:
        preds.extend(model(x0.to(device), x1.to(device), x2.to(device)).cpu().numpy())
        trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


def make_sample_weights(hb, max_w=4.0):
    bins = np.floor(hb).astype(int)
    cnt  = {b: (bins == b).sum() for b in np.unique(bins)}
    w    = np.array([1.0 / cnt[b] for b in bins], dtype=np.float32)
    return np.clip(w / w.mean(), 0, max_w)


# ── 5-fold stratified ensemble ─────────────────────────────────────────────

def run_task(task, train_df, test_df, seg0, seg1, seg2, shared):
    print(f"\n{'='*65}")
    print(f"  SmallHbNet — {task.upper()}  ({N_FOLDS}-fold stratified ensemble)")
    print(f"{'='*65}")

    y_tr_hb = train_df["hb_value"].values.astype(np.float32)
    y_te_hb = test_df["hb_value"].values.astype(np.float32)
    groups   = train_df["video_id"].values

    # stratify label for StratifiedGroupKFold
    strat = pd.cut(y_tr_hb, bins=[0,7,10,12,25], labels=[0,1,2,3]).astype(int)

    if task == "classification":
        y_tr = (y_tr_hb < ANEMIA_THRESH).astype(np.float32)
        y_te = (y_te_hb < ANEMIA_THRESH).astype(np.float32)
    else:
        y_tr = y_tr_hb
        y_te = y_te_hb

    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(sgkf.split(y_tr, strat, groups=groups))

    te_preds_all = []
    val_metrics  = []

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        # ── per-fold preprocessors (fit only on training sub-fold) ────────
        imp0, scl0, sel0 = make_preprocessor(
            raw_X(train_df.iloc[tr_idx], seg0, shared), y_tr_hb[tr_idx])
        imp1, scl1, sel1 = make_preprocessor(
            raw_X(train_df.iloc[tr_idx], seg1, shared), y_tr_hb[tr_idx])
        imp2, scl2, sel2 = make_preprocessor(
            raw_X(train_df.iloc[tr_idx], seg2, shared), y_tr_hb[tr_idx])

        X0_tr = apply_preprocessor(raw_X(train_df.iloc[tr_idx], seg0, shared), imp0, scl0, sel0)
        X1_tr = apply_preprocessor(raw_X(train_df.iloc[tr_idx], seg1, shared), imp1, scl1, sel1)
        X2_tr = apply_preprocessor(raw_X(train_df.iloc[tr_idx], seg2, shared), imp2, scl2, sel2)

        X0_va = apply_preprocessor(raw_X(train_df.iloc[val_idx], seg0, shared), imp0, scl0, sel0)
        X1_va = apply_preprocessor(raw_X(train_df.iloc[val_idx], seg1, shared), imp1, scl1, sel1)
        X2_va = apply_preprocessor(raw_X(train_df.iloc[val_idx], seg2, shared), imp2, scl2, sel2)

        X0_te = apply_preprocessor(raw_X(test_df, seg0, shared), imp0, scl0, sel0)
        X1_te = apply_preprocessor(raw_X(test_df, seg1, shared), imp1, scl1, sel1)
        X2_te = apply_preprocessor(raw_X(test_df, seg2, shared), imp2, scl2, sel2)

        tr_ds = HbDataset(X0_tr, X1_tr, X2_tr, y_tr[tr_idx], augment=True)
        va_ds = HbDataset(X0_va, X1_va, X2_va, y_tr[val_idx], augment=False)
        te_ds = HbDataset(X0_te, X1_te, X2_te, y_te, augment=False)

        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=len(tr_ds) > BATCH_SIZE)
        va_ld = DataLoader(va_ds, batch_size=128, shuffle=False)
        te_ld = DataLoader(te_ds, batch_size=128, shuffle=False)

        # ── model ──────────────────────────────────────────────────────────
        model = SmallHbNet(input_size=K_FEATURES, task=task).to(DEVICE)

        if task == "classification":
            pos  = float((y_tr[tr_idx] == 1).sum())
            neg  = float((y_tr[tr_idx] == 0).sum())
            pw   = torch.tensor([neg / (pos + 1e-8)],
                                dtype=torch.float32).to(DEVICE)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        else:
            crit = nn.HuberLoss(delta=1.5, reduction="none")

        opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

        best_val, best_state, pat = (
            float("inf") if task == "regression" else -float("inf"),
            None, 0)

        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, tr_ld, opt, crit, DEVICE)
            vp, vt = predict(model, va_ld, DEVICE)
            if task == "classification":
                prob = torch.sigmoid(torch.tensor(vp)).numpy()
                try:    vm = roc_auc_score(vt, prob)
                except: vm = 0.5
                improved = vm > best_val
            else:
                vm       = mean_absolute_error(vt, vp)
                improved = vm < best_val

            if improved:
                best_val, pat = vm, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            if pat >= PATIENCE:
                break
            sched.step()

        model.load_state_dict(best_state)
        tp, _ = predict(model, te_ld, DEVICE)
        te_preds_all.append(tp)
        val_metrics.append(best_val)
        tag = "AUC" if task == "classification" else "MAE"
        print(f"    Fold {fold_i+1}/{N_FOLDS}  best_val_{tag}={best_val:.4f}  "
              f"params={count_params(model):,}  epoch={MAX_EPOCHS-pat+PATIENCE}")

    ensemble = np.mean(te_preds_all, axis=0)
    mean_vm  = np.mean(val_metrics)
    std_vm   = np.std(val_metrics)
    print(f"  CV {tag}: {mean_vm:.4f} ± {std_vm:.4f}")

    # save raw predictions for analysis
    pred_df = pd.DataFrame({
        "video_id":  test_df["video_id"].values,
        "hb_true":   y_te_hb,
        "hb_pred":   ensemble if task == "regression"
                     else torch.sigmoid(torch.tensor(ensemble)).numpy(),
        "error":     (ensemble - y_te_hb) if task == "regression"
                     else np.zeros(len(y_te_hb)),
        "abs_error": np.abs(ensemble - y_te_hb) if task == "regression"
                     else np.zeros(len(y_te_hb)),
        "protocol":  test_df["protocol"].values,
    })
    pred_df.to_csv(OUT_DIR / f"small_predictions_{task}.csv", index=False)

    # ── metrics ────────────────────────────────────────────────────────────
    metrics = {}
    if task == "regression":
        metrics["MAE"]  = mean_absolute_error(y_te_hb, ensemble)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_te_hb, ensemble))
        metrics["R2"]   = r2_score(y_te_hb, ensemble)
        for b in ACCURACY_BANDS:
            metrics[f"within_{str(b).replace('.','_')}"] = float(
                (np.abs(y_te_hb - ensemble) <= b).mean())
        try:
            metrics["AUC"] = roc_auc_score(
                (y_te_hb < ANEMIA_THRESH).astype(int), -ensemble)
        except: metrics["AUC"] = float("nan")

        print(f"\n  Test — Regression:")
        print(f"    MAE      : {metrics['MAE']:.4f} g/dL")
        print(f"    RMSE     : {metrics['RMSE']:.4f} g/dL")
        print(f"    R²       : {metrics['R2']:.4f}")
        print(f"    Within±1 : {100*metrics['within_1_0']:.1f}%")
        print(f"    Within±2 : {100*metrics['within_2_0']:.1f}%")
        print(f"    AUC      : {metrics['AUC']:.4f}")
        print(f"\n  Error by HB range:")
        print(f"  {'Range':>6}  {'n':>4}  {'MAE':>6}  {'Within±1':>9}")
        for lo, hi in [(0,7),(7,10),(10,12),(12,25)]:
            m = (y_te_hb >= lo) & (y_te_hb < hi)
            if m.sum() == 0: continue
            r  = mean_absolute_error(y_te_hb[m], ensemble[m])
            w1 = (np.abs(y_te_hb[m] - ensemble[m]) <= 1.0).mean()
            print(f"  {lo:>3}–{hi:<3}  {m.sum():>4}  {r:>6.3f}  {100*w1:>8.1f}%")

        # plots
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        lim = (min(y_te_hb.min(), ensemble.min())-0.5,
               max(y_te_hb.max(), ensemble.max())+0.5)
        ax = axes[0]
        ax.scatter(y_te_hb, ensemble, alpha=0.75, s=40, color="steelblue",
                   edgecolors="white", lw=0.4)
        ax.plot(lim, lim, "k--", lw=1, label="Identity")
        ax.plot(lim, [l+1 for l in lim], color="orange", lw=1, linestyle=":", label="±1 g/dL")
        ax.plot(lim, [l-1 for l in lim], color="orange", lw=1, linestyle=":")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("True HB (g/dL)"); ax.set_ylabel("Predicted HB (g/dL)")
        ax.set_title(f"True vs Predicted\nMAE={metrics['MAE']:.3f}  R²={metrics['R2']:.3f}")
        ax.legend(fontsize=8)
        err = ensemble - y_te_hb
        ax = axes[1]
        ax.scatter(y_te_hb, err, alpha=0.75, s=40, color="tomato",
                   edgecolors="white", lw=0.4)
        ax.axhline(0, color="black", lw=1)
        ax.axhline(err.mean(), color="red", lw=1.5, linestyle="--",
                   label=f"Bias={err.mean():+.3f}")
        ax.axhline(1,  color="orange", lw=0.7, linestyle=":")
        ax.axhline(-1, color="orange", lw=0.7, linestyle=":")
        ax.set_xlabel("True HB (g/dL)"); ax.set_ylabel("Residual (pred−true)")
        ax.set_title("Residual Plot"); ax.legend(fontsize=8)
        plt.suptitle("SmallHbNet — Regression (5-fold ensemble)", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "small_predictions_regression.png", dpi=130)
        plt.close()

    else:
        prob   = torch.sigmoid(torch.tensor(ensemble)).numpy()
        pred_b = (prob >= 0.5).astype(int)
        true_b = y_te.astype(int)
        metrics["AUC"]         = roc_auc_score(true_b, prob)
        metrics["F1_macro"]    = f1_score(true_b, pred_b, average="macro", zero_division=0)
        metrics["F1_anemic"]   = f1_score(true_b, pred_b, average="binary", zero_division=0)
        cm = confusion_matrix(true_b, pred_b, labels=[1,0])
        metrics["Sensitivity"] = cm[0,0] / float(true_b.sum())
        metrics["Specificity"] = cm[1,1] / float((true_b==0).sum())
        metrics["BalAcc"]      = balanced_accuracy_score(true_b, pred_b)
        metrics["Accuracy"]    = float((pred_b == true_b).mean())

        print(f"\n  Test — Classification:")
        print(f"    AUC         : {metrics['AUC']:.4f}")
        print(f"    Accuracy    : {100*metrics['Accuracy']:.1f}%")
        print(f"    Sensitivity : {metrics['Sensitivity']:.4f}")
        print(f"    Specificity : {metrics['Specificity']:.4f}")
        print(f"    F1-anemic   : {metrics['F1_anemic']:.4f}")
        print(f"    Bal.Acc     : {metrics['BalAcc']:.4f}")
        print()
        print(classification_report(true_b, pred_b,
                                    target_names=["Normal (≥12)", "Anemic (<12)"],
                                    zero_division=0))
        print(f"  Confusion matrix:")
        print(f"                    Pred-Anemic  Pred-Normal")
        print(f"    True-Anemic         {cm[0,0]:>4}         {cm[0,1]:>4}")
        print(f"    True-Normal         {cm[1,0]:>4}         {cm[1,1]:>4}")

        # ROC
        fpr, tpr, _ = roc_curve(true_b, prob)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, lw=2, color="steelblue",
                 label=f"SmallHbNet AUC={metrics['AUC']:.3f}")
        plt.plot([0,1],[0,1],"k--",lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC — Anemia ({N_FOLDS}-fold Ensemble)")
        plt.legend(fontsize=10); plt.tight_layout()
        plt.savefig(OUT_DIR / "small_roc_classification.png", dpi=130)
        plt.close()

    return metrics


# ── detailed prediction analysis ───────────────────────────────────────────

def prediction_analysis(reg_csv):
    df = pd.read_csv(reg_csv)
    true = df["hb_true"].values
    pred = df["hb_pred"].values
    err  = df["error"].values

    print("\n" + "=" * 75)
    print("  PREDICTION ANALYSIS — per HB level")
    print("=" * 75)

    # ── overall prediction stats ───────────────────────────────────────────
    print(f"\n  Overall (n={len(df)}):")
    print(f"  {'':20s}  {'Min':>7}  {'Max':>7}  {'Mean':>7}  {'SD':>7}")
    print(f"  {'True HB':<20s}  {true.min():7.2f}  {true.max():7.2f}  {true.mean():7.2f}  {true.std():7.2f}")
    print(f"  {'Predicted HB':<20s}  {pred.min():7.2f}  {pred.max():7.2f}  {pred.mean():7.2f}  {pred.std():7.2f}")
    print(f"  {'Error (pred−true)':<20s}  {err.min():7.2f}  {err.max():7.2f}  {err.mean():7.2f}  {err.std():7.2f}")
    print(f"  {'|Error|':<20s}  {df['abs_error'].min():7.2f}  {df['abs_error'].max():7.2f}  {df['abs_error'].mean():7.2f}  {df['abs_error'].std():7.2f}")

    # ── per 1 g/dL bin ────────────────────────────────────────────────────
    print(f"\n  Per HB level (1 g/dL bins):")
    hdr = f"  {'HB bin':>8}  {'n':>4}  {'TrueMin':>8} {'TrueMx':>7}  "   \
          f"{'PredMin':>8} {'PredMx':>7} {'PredMn':>7} {'PredSD':>7}  "   \
          f"{'Bias':>7} {'MAE':>7}  {'W±1':>6} {'W±2':>6}"
    print(hdr)
    print("  " + "─" * 100)

    bins = np.arange(3, 19, 1)
    for lo in bins:
        hi  = lo + 1
        m   = (true >= lo) & (true < hi)
        if m.sum() == 0:
            continue
        t   = true[m]; p = pred[m]; e = err[m]
        w1  = (np.abs(e) <= 1.0).mean()
        w2  = (np.abs(e) <= 2.0).mean()
        flag = " ◄ fail" if np.abs(e).mean() > 2.5 else ""
        print(f"  {lo:>4.0f}–{hi:<4.0f}  {m.sum():>4}  "
              f"{t.min():>8.2f} {t.max():>7.2f}  "
              f"{p.min():>8.2f} {p.max():>7.2f} {p.mean():>7.2f} {p.std():>7.2f}  "
              f"{e.mean():>+7.2f} {np.abs(e).mean():>7.2f}  "
              f"{100*w1:>5.0f}% {100*w2:>5.0f}%{flag}")

    # ── severity band summary ──────────────────────────────────────────────
    print(f"\n  Severity band summary:")
    print(f"  {'Band':<22}  {'n':>4}  {'Pred range':>15}  {'Pred mean':>10}  {'Bias':>7}  {'MAE':>7}  {'W±1':>6}  {'W±2':>6}")
    print("  " + "─" * 90)
    bands = [("<7  severe",      0,  7),
             ("7–10 moderate",   7, 10),
             ("10–12 mild",     10, 12),
             ("≥12 normal",     12, 25)]
    for lbl, lo, hi in bands:
        m = (true >= lo) & (true < hi)
        if m.sum() == 0: continue
        p_ = pred[m]; e_ = err[m]
        w1 = (np.abs(e_) <= 1.0).mean()
        w2 = (np.abs(e_) <= 2.0).mean()
        print(f"  {lbl:<22}  {m.sum():>4}  "
              f"{p_.min():>6.2f}–{p_.max():<6.2f}  "
              f"{p_.mean():>10.2f}  "
              f"{e_.mean():>+7.2f}  {np.abs(e_).mean():>7.2f}  "
              f"{100*w1:>5.0f}%  {100*w2:>5.0f}%")

    # ── worst predictions ─────────────────────────────────────────────────
    df_sorted = df.reindex(df["abs_error"].sort_values(ascending=False).index)
    print(f"\n  Worst 10 predictions:")
    print(f"  {'video_id':<35}  {'True':>7}  {'Pred':>7}  {'Error':>8}  {'Protocol'}")
    print("  " + "─" * 75)
    for _, row in df_sorted.head(10).iterrows():
        print(f"  {str(row['video_id']):<35}  {row['hb_true']:>7.2f}  "
              f"{row['hb_pred']:>7.2f}  {row['error']:>+8.2f}  {row['protocol']}")

    # ── prediction compression check ──────────────────────────────────────
    print(f"\n  Range compression check:")
    print(f"    True HB std   : {true.std():.3f} g/dL")
    print(f"    Predicted std : {pred.std():.3f} g/dL")
    ratio = pred.std() / (true.std() + 1e-8)
    print(f"    Pred/True std : {ratio:.3f}  (1.0=perfect, <1=compressed toward mean)")

    # ── 2D plot: true vs pred coloured by error ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # scatter coloured by abs error
    ax = axes[0]
    sc = ax.scatter(true, pred, c=df["abs_error"], cmap="RdYlGn_r",
                    vmin=0, vmax=4, s=50, edgecolors="white", lw=0.4)
    lim = (min(true.min(), pred.min())-0.5, max(true.max(), pred.max())+0.5)
    ax.plot(lim, lim, "k--", lw=1, label="Identity")
    ax.plot(lim, [l+2 for l in lim], "orange", lw=0.7, linestyle=":", label="±2 g/dL")
    ax.plot(lim, [l-2 for l in lim], "orange", lw=0.7, linestyle=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True HB (g/dL)"); ax.set_ylabel("Predicted HB (g/dL)")
    ax.set_title("True vs Predicted\n(colour = |error|)")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="|Error| g/dL")

    # MAE by HB bin bar chart
    ax = axes[1]
    bin_labels, bin_maes, bin_ns, bin_w1 = [], [], [], []
    for lo in np.arange(3, 19, 1):
        hi = lo + 1
        m  = (true >= lo) & (true < hi)
        if m.sum() < 2: continue
        bin_labels.append(f"{lo:.0f}–{hi:.0f}")
        bin_maes.append(np.abs(err[m]).mean())
        bin_ns.append(m.sum())
        bin_w1.append((np.abs(err[m]) <= 1.0).mean() * 100)

    colors = ["#e74c3c" if m > 2.5 else "#2ecc71" if m < 1.5 else "#f39c12"
              for m in bin_maes]
    bars = ax.bar(range(len(bin_labels)), bin_maes, color=colors, alpha=0.8,
                  edgecolor="white")
    ax.axhline(1.0, color="green",  lw=1.2, linestyle="--", label="1 g/dL target")
    ax.axhline(2.0, color="orange", lw=1.2, linestyle="--", label="2 g/dL")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
    ax.set_xlabel("HB bin (g/dL)"); ax.set_ylabel("MAE (g/dL)")
    ax.set_title("MAE by HB Level\n(red=fail, orange=ok, green=good)")
    ax.legend(fontsize=8)
    for i, (n, w) in enumerate(zip(bin_ns, bin_w1)):
        ax.text(i, bin_maes[i] + 0.05, f"n={n}\n{w:.0f}%", ha="center",
                fontsize=7, color="black")

    # prediction std vs true HB (compression check per bin)
    ax = axes[2]
    bin_true_means, bin_pred_means, bin_pred_stds = [], [], []
    for lo in np.arange(3, 19, 1):
        hi = lo + 1
        m  = (true >= lo) & (true < hi)
        if m.sum() < 2: continue
        bin_true_means.append(true[m].mean())
        bin_pred_means.append(pred[m].mean())
        bin_pred_stds.append(pred[m].std())

    ax.plot(bin_true_means, bin_true_means, "k--", lw=1, label="True mean")
    ax.errorbar(bin_true_means, bin_pred_means,
                yerr=bin_pred_stds, fmt="o-", color="steelblue",
                capsize=4, lw=1.5, markersize=6, label="Pred mean ± SD")
    ax.set_xlabel("True HB bin mean (g/dL)")
    ax.set_ylabel("Predicted HB (g/dL)")
    ax.set_title("Prediction Mean ± SD per True HB Bin\n(compression toward center visible here)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("SmallHbNet — Detailed Prediction Analysis (5-fold Ensemble)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "small_analysis_detail.png", dpi=140)
    plt.close()
    print(f"\n  Saved → {OUT_DIR / 'small_analysis_detail.png'}")


# ── entry point ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SmallHbNet — Small Network for 386-sample Dataset")
    print("=" * 65)
    print(f"  Device       : {DEVICE}")
    print(f"  Features/branch: {K_FEATURES} (SelectKBest from 334)")
    print(f"  Architecture : {K_FEATURES}→{BRANCH_DIM} × 3 → {HIDDEN_DIM} → 1")

    # quick param estimate
    p = K_FEATURES*BRANCH_DIM + BRANCH_DIM + BRANCH_DIM*3*HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM
    print(f"  Est. params  : ~{3*p//3 + BRANCH_DIM*3*HIDDEN_DIM:,}")
    print(f"  Augmentation : noise_std={NOISE_STD}  feat_drop={FEAT_DROP}")
    print(f"  Regularize   : dropout={DROPOUT}  weight_decay={WEIGHT_DECAY}")

    train_df = pd.read_csv(FEAT_DIR / "features_train.csv")
    test_df  = pd.read_csv(FEAT_DIR / "features_test.csv")
    seg0, seg1, seg2, shared = get_feature_groups(train_df)

    print(f"  Train/Test   : {len(train_df)} / {len(test_df)}")
    print(f"  Anemia(train): {int((train_df['hb_value']<12).sum())}/{len(train_df)}")

    # quick model param count
    demo = SmallHbNet(input_size=K_FEATURES, task="regression")
    print(f"  Actual params: {count_params(demo):,}")

    reg_m = run_task("regression",     train_df, test_df, seg0, seg1, seg2, shared)
    cls_m = run_task("classification", train_df, test_df, seg0, seg1, seg2, shared)

    prediction_analysis(OUT_DIR / "small_predictions_regression.csv")

    print("\n" + "=" * 65)
    print("  FINAL COMPARISON")
    print("=" * 65)
    print(f"  {'Model':<30s}  {'MAE':>6}  {'R²':>7}  {'W±2':>6}  {'AUC-cls':>8}  {'Sens':>6}  {'F1-an':>6}")
    print(f"  {'-'*70}")
    print(f"  {'SmallHbNet (NN)':<30s}  "
          f"{reg_m['MAE']:>6.3f}  {reg_m['R2']:>7.3f}  "
          f"{100*reg_m['within_2_0']:>5.1f}%  "
          f"{cls_m['AUC']:>8.3f}  {cls_m['Sensitivity']:>6.3f}  {cls_m['F1_anemic']:>6.3f}")
    print(f"  {'GradientBoosting (ML)':<30s}  "
          f"{'2.203':>6}  {'-0.083':>7}  {'57.9%':>6}  {'0.614':>8}  {'0.871':>6}  {'0.824':>6}")
    print(f"  {'RandomForest (ML)':<30s}  "
          f"{'2.203':>6}  {'-0.083':>7}  {'53.7%':>6}  {'0.603':>8}  {'0.771':>6}  {'0.777':>6}")
    print()


if __name__ == "__main__":
    main()
