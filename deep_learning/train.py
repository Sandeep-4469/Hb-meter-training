"""
Deep learning HB estimation using DeepHbNetFlexible.

Three-branch architecture (one branch per LED segment):
  - Red branch   : seg0 features (243) + shared cross-seg (91) = 334
  - Orange branch: seg1 features (243) + shared cross-seg (91) = 334
  - Yellow branch: seg2 features (243) + shared cross-seg (91) = 334

Fixes vs naive approach:
  - Bounded regression output: 3 + 17*sigmoid(logit) → forces valid HB range
  - 5-fold CV ensemble: train 5 models, average test predictions → reduces overfitting
  - Smaller architecture (128 branch dim, 2-3 blocks) for small dataset
  - Dropout 0.5 + weight_decay 5e-3
  - GroupKFold by video_id (no patient leakage)

Run:
    python train.py
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                              mean_absolute_error, mean_squared_error, r2_score,
                              confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import RANDOM_SEED, ACCURACY_BANDS

OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR = ROOT / "metric_learning" / "results"

ANEMIA_THRESH = 12.0
HB_MIN, HB_MAX = 3.0, 20.0           # bounded output range
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = 32
MAX_EPOCHS     = 400
PATIENCE       = 50
LR             = 5e-4
WEIGHT_DECAY   = 5e-3
DROPOUT        = 0.5
NUM_BLOCKS     = 2
N_FOLDS        = 5

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

META = {"video_id", "hb_value", "split", "protocol"}


# ── feature split ──────────────────────────────────────────────────────────

def get_feature_groups(df):
    feat_cols = [c for c in df.columns if c not in META and c != "anemia"]
    seg0   = [c for c in feat_cols if c.startswith("seg0_")]
    seg1   = [c for c in feat_cols if c.startswith("seg1_")]
    seg2   = [c for c in feat_cols if c.startswith("seg2_")]
    shared = [c for c in feat_cols if c not in seg0 + seg1 + seg2]
    return seg0, seg1, seg2, shared


def build_X(df, seg_cols, shared_cols):
    return df[seg_cols + shared_cols].values.astype(np.float32)


def make_scaler():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler())])


def make_sample_weights(hb_values, max_w=5.0):
    bins   = np.floor(hb_values).astype(int)
    counts = {b: (bins == b).sum() for b in np.unique(bins)}
    w = np.array([1.0 / counts[b] for b in bins], dtype=np.float32)
    w = w / w.mean()
    return np.clip(w, 0, max_w)


# ── dataset ────────────────────────────────────────────────────────────────

class HbDataset(Dataset):
    def __init__(self, X0, X1, X2, y):
        self.X0 = torch.tensor(X0, dtype=torch.float32)
        self.X1 = torch.tensor(X1, dtype=torch.float32)
        self.X2 = torch.tensor(X2, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X0[idx], self.X1[idx], self.X2[idx], self.y[idx]


# ── model ──────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_f, out_f, dropout=0.4):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(out_f),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_f, out_f),
            nn.BatchNorm1d(out_f),
        )
        self.skip = nn.Linear(in_f, out_f) if in_f != out_f else nn.Identity()
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.main(x) + self.skip(x)))


class DeepHbNetFlexible(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5, num_blocks=2,
                 task="regression"):
        super().__init__()
        self.task      = task
        branch_dim = 128

        def branch(in_f, out_f, dr):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dr),
            )

        self.red_branch    = branch(input_size, branch_dim, dropout_rate)
        self.orange_branch = branch(input_size, branch_dim, dropout_rate)
        self.yellow_branch = branch(input_size, branch_dim, dropout_rate)

        combined   = branch_dim * 3
        hidden_seq = [256, 128, 64, 32][:num_blocks]

        self.blocks = nn.ModuleList()
        in_f = combined
        for h in hidden_seq:
            self.blocks.append(ResidualBlock(in_f, h, dropout_rate))
            in_f = h

        self.output_layer = nn.Linear(in_f, 1)

    def forward(self, x_red, x_orange, x_yellow):
        r = self.red_branch(x_red)
        o = self.orange_branch(x_orange)
        y = self.yellow_branch(x_yellow)
        x = torch.cat([r, o, y], dim=1)
        for blk in self.blocks:
            x = blk(x)
        logit = self.output_layer(x).squeeze(1)

        if self.task == "regression":
            # bounded output: forces predictions into [HB_MIN, HB_MAX]
            return HB_MIN + (HB_MAX - HB_MIN) * torch.sigmoid(logit)
        else:
            return logit   # raw logit for BCEWithLogitsLoss


# ── train / eval helpers ───────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x0, x1, x2, y in loader:
        x0, x1, x2, y = x0.to(device), x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x0, x1, x2)
        loss = criterion(pred, y).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x0, x1, x2, y in loader:
        p = model(x0.to(device), x1.to(device), x2.to(device))
        preds.extend(p.cpu().numpy())
        trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


# ── 5-fold ensemble training ───────────────────────────────────────────────

def train_ensemble(task, X0_tr, X1_tr, X2_tr, y_tr_task, y_tr_hb,
                   X0_te, X1_te, X2_te, y_te_task, groups):
    gkf     = GroupKFold(n_splits=N_FOLDS)
    splits  = list(gkf.split(X0_tr, y_tr_task, groups=groups))

    test_preds_folds = []
    val_metrics = []

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        tr_ds  = HbDataset(X0_tr[tr_idx], X1_tr[tr_idx], X2_tr[tr_idx],
                           y_tr_task[tr_idx])
        val_ds = HbDataset(X0_tr[val_idx], X1_tr[val_idx], X2_tr[val_idx],
                           y_tr_task[val_idx])
        te_ds  = HbDataset(X0_te, X1_te, X2_te, y_te_task)

        tr_loader  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                                drop_last=len(tr_ds) > BATCH_SIZE)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        te_loader  = DataLoader(te_ds,  batch_size=128, shuffle=False)

        model = DeepHbNetFlexible(input_size=X0_tr.shape[1],
                                   dropout_rate=DROPOUT,
                                   num_blocks=NUM_BLOCKS,
                                   task=task).to(DEVICE)

        if task == "classification":
            pos  = float((y_tr_task[tr_idx] == 1).sum())
            neg  = float((y_tr_task[tr_idx] == 0).sum())
            pw   = torch.tensor([neg / (pos + 1e-8)],
                                dtype=torch.float32).to(DEVICE)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        else:
            crit = nn.HuberLoss(delta=1.5, reduction="none")

        opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

        best_val, best_state, patience_cnt = (
            float("inf") if task == "regression" else -float("inf"),
            None, 0
        )

        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, tr_loader, opt, crit, DEVICE)
            val_pred, val_true = predict(model, val_loader, DEVICE)

            if task == "classification":
                prob = torch.sigmoid(torch.tensor(val_pred)).numpy()
                try:
                    vm = roc_auc_score(val_true, prob)
                except Exception:
                    vm = 0.5
                improved = vm > best_val
            else:
                vm       = mean_absolute_error(val_true, val_pred)
                improved = vm < best_val

            if improved:
                best_val, patience_cnt = vm, 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if patience_cnt >= PATIENCE:
                break
            sched.step()

        model.load_state_dict(best_state)
        te_pred, _ = predict(model, te_loader, DEVICE)
        test_preds_folds.append(te_pred)
        val_metrics.append(best_val)
        print(f"    Fold {fold_i+1}/{N_FOLDS}  best_val={'AUC' if task=='classification' else 'MAE'}="
              f"{best_val:.4f}  stopped_early")

    # ensemble: average predictions across folds
    ensemble_pred = np.mean(test_preds_folds, axis=0)
    print(f"  Mean val {'AUC' if task=='classification' else 'MAE'} across folds: "
          f"{np.mean(val_metrics):.4f} ± {np.std(val_metrics):.4f}")
    return ensemble_pred, val_metrics


# ── full task runner ───────────────────────────────────────────────────────

def run_task(task, train_df, test_df, seg0, seg1, seg2, shared):
    print(f"\n{'='*65}")
    print(f"  TASK: {task.upper()}  ({N_FOLDS}-fold ensemble)")
    print(f"{'='*65}")

    # ── scalers fitted on train ────────────────────────────────────────────
    sc0 = make_scaler().fit(build_X(train_df, seg0, shared))
    sc1 = make_scaler().fit(build_X(train_df, seg1, shared))
    sc2 = make_scaler().fit(build_X(train_df, seg2, shared))

    X0_tr = sc0.transform(build_X(train_df, seg0, shared))
    X1_tr = sc1.transform(build_X(train_df, seg1, shared))
    X2_tr = sc2.transform(build_X(train_df, seg2, shared))

    X0_te = sc0.transform(build_X(test_df, seg0, shared))
    X1_te = sc1.transform(build_X(test_df, seg1, shared))
    X2_te = sc2.transform(build_X(test_df, seg2, shared))

    y_tr_hb = train_df["hb_value"].values.astype(np.float32)
    y_te_hb = test_df["hb_value"].values.astype(np.float32)

    if task == "classification":
        y_tr_task = (y_tr_hb < ANEMIA_THRESH).astype(np.float32)
        y_te_task = (y_te_hb < ANEMIA_THRESH).astype(np.float32)
    else:
        y_tr_task = y_tr_hb
        y_te_task = y_te_hb

    groups = train_df["video_id"].values

    te_pred, val_metrics = train_ensemble(
        task, X0_tr, X1_tr, X2_tr, y_tr_task, y_tr_hb,
        X0_te, X1_te, X2_te, y_te_task, groups)

    # ── test metrics ───────────────────────────────────────────────────────
    metrics = {}
    if task == "regression":
        metrics["MAE"]  = mean_absolute_error(y_te_hb, te_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_te_hb, te_pred))
        metrics["R2"]   = r2_score(y_te_hb, te_pred)
        for b in ACCURACY_BANDS:
            k = f"within_{str(b).replace('.','_')}"
            metrics[k] = float((np.abs(y_te_hb - te_pred) <= b).mean())
        try:
            metrics["AUC"] = roc_auc_score((y_te_hb < ANEMIA_THRESH).astype(int),
                                           -te_pred)
        except Exception:
            metrics["AUC"] = float("nan")

        print(f"\n  Test results (Regression):")
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
            mae_r = mean_absolute_error(y_te_hb[m], te_pred[m])
            w1    = (np.abs(y_te_hb[m] - te_pred[m]) <= 1.0).mean()
            print(f"  {lo:>3}–{hi:<3}  {m.sum():>4}  {mae_r:>6.3f}  {100*w1:>8.1f}%")

        # plots
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        lim = (min(y_te_hb.min(), te_pred.min()) - 0.5,
               max(y_te_hb.max(), te_pred.max()) + 0.5)
        a1.scatter(y_te_hb, te_pred, alpha=0.7, s=35,
                   color="steelblue", edgecolors="white", lw=0.4)
        a1.plot(lim, lim, "k--", lw=1, label="Identity")
        a1.plot(lim, [l+1 for l in lim], "orange", lw=0.8, linestyle=":", label="±1")
        a1.plot(lim, [l-1 for l in lim], "orange", lw=0.8, linestyle=":")
        a1.set_xlim(lim); a1.set_ylim(lim)
        a1.set_xlabel("True HB"); a1.set_ylabel("Predicted HB")
        a1.set_title(f"True vs Predicted\nMAE={metrics['MAE']:.3f}  R²={metrics['R2']:.3f}")
        a1.legend(fontsize=8)
        err = te_pred - y_te_hb
        a2.scatter(y_te_hb, err, alpha=0.7, s=35,
                   color="tomato", edgecolors="white", lw=0.4)
        a2.axhline(0, color="black", lw=1)
        a2.axhline(err.mean(), color="red", lw=1.5, linestyle="--",
                   label=f"Bias={err.mean():+.3f}")
        a2.set_xlabel("True HB"); a2.set_ylabel("Residual")
        a2.set_title("Residuals"); a2.legend(fontsize=8)
        plt.suptitle(f"DeepHbNet — Regression ({N_FOLDS}-fold ensemble)", fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "predictions_regression.png", dpi=130)
        plt.close()

    else:
        prob    = torch.sigmoid(torch.tensor(te_pred)).numpy()
        pred_b  = (prob >= 0.5).astype(int)
        true_b  = y_te_task.astype(int)
        metrics["AUC"]         = roc_auc_score(true_b, prob)
        metrics["F1_macro"]    = f1_score(true_b, pred_b, average="macro", zero_division=0)
        metrics["F1_anemic"]   = f1_score(true_b, pred_b, average="binary", zero_division=0)
        cm = confusion_matrix(true_b, pred_b, labels=[1, 0])
        metrics["Sensitivity"] = cm[0, 0] / float(true_b.sum())
        metrics["Specificity"] = cm[1, 1] / float((true_b == 0).sum())
        metrics["BalAcc"]      = balanced_accuracy_score(true_b, pred_b)
        metrics["Accuracy"]    = float((pred_b == true_b).mean())

        print(f"\n  Test results (Classification):")
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

        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(true_b, prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, color="steelblue",
                 label=f"AUC = {metrics['AUC']:.3f}")
        plt.plot([0,1],[0,1],"k--",lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — Anemia Classification\n({N_FOLDS}-fold Ensemble)")
        plt.legend(fontsize=10); plt.tight_layout()
        plt.savefig(OUT_DIR / "roc_classification.png", dpi=130)
        plt.close()

    return metrics


# ── entry point ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  DeepHbNetFlexible  —  5-fold Ensemble Training")
    print("=" * 65)
    print(f"  Device     : {DEVICE}")

    train_df = pd.read_csv(FEAT_DIR / "features_train.csv")
    test_df  = pd.read_csv(FEAT_DIR / "features_test.csv")

    seg0, seg1, seg2, shared = get_feature_groups(train_df)
    branch_in = len(seg0) + len(shared)

    print(f"  Train      : {len(train_df)}  |  Test: {len(test_df)}")
    print(f"  Per-seg    : {len(seg0)}  |  Shared: {len(shared)}")
    print(f"  Branch in  : {branch_in}")
    print(f"  Anemia(trn): {int((train_df['hb_value']<12).sum())}/{len(train_df)}")
    print(f"  Config     : dropout={DROPOUT}  blocks={NUM_BLOCKS}  "
          f"lr={LR}  wd={WEIGHT_DECAY}  patience={PATIENCE}")

    reg_metrics = run_task("regression",     train_df, test_df, seg0, seg1, seg2, shared)
    cls_metrics = run_task("classification", train_df, test_df, seg0, seg1, seg2, shared)

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — DeepHbNetFlexible (5-fold ensemble)")
    print("=" * 65)
    print(f"  Regression     MAE={reg_metrics['MAE']:.3f} g/dL  "
          f"RMSE={reg_metrics['RMSE']:.3f}  R²={reg_metrics['R2']:.3f}  "
          f"Within±1={100*reg_metrics['within_1_0']:.1f}%  "
          f"Within±2={100*reg_metrics['within_2_0']:.1f}%  "
          f"AUC={reg_metrics['AUC']:.3f}")
    print(f"  Classification AUC={cls_metrics['AUC']:.3f}  "
          f"Sens={cls_metrics['Sensitivity']:.3f}  "
          f"Spec={cls_metrics['Specificity']:.3f}  "
          f"F1-anemic={cls_metrics['F1_anemic']:.3f}  "
          f"Acc={100*cls_metrics['Accuracy']:.1f}%")

    # comparison vs best ML
    print("\n  vs. Best ML models (GradientBoosting / RandomForest):")
    print(f"  ML Regression  MAE=2.203  R²=-0.083  Within±2=57.9%  AUC=0.616")
    print(f"  ML Classif     AUC=0.614  Sens=0.871  Spec=0.320  F1-anemic=0.824")


if __name__ == "__main__":
    main()
