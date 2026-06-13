"""
Best HB estimation pipeline — designed for this specific dataset.

Dataset reality:
  Train 386 / Test 95
  Severe (<7): 33 train, 7 test  ← hardest, fewest
  Moderate (7-10): 117 / 41
  Mild (10-12): 105 / 22
  Normal (>12): 131 / 25

Techniques:
  1. SMOTER augmentation  — upsample severe + high-normal extremes per fold
  2. Extreme-weighted Huber loss — samples far from mean get 5× more weight
  3. Multi-task ordinal heads — predict P(HB<7), P(HB<10), P(HB<12) alongside regression
  4. StratifiedGroupKFold — HB-range-balanced folds, no patient leakage
  5. 5-fold ensemble predictions on test set
  6. Final stacking: NN ensemble + GradientBoosting blended for best combined output

Run:
    python train_best.py
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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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

# ── config ─────────────────────────────────────────────────────────────────
ANEMIA_THRESH  = 12.0
HB_MIN, HB_MAX = 3.0, 20.0
ORD_THRESHOLDS = [7.0, 10.0, 12.0]      # clinical ordinal boundaries
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_FEATURES  = 40      # per-branch SelectKBest
BRANCH_DIM  = 48
HIDDEN_DIM  = 96
DROPOUT     = 0.4
NOISE_STD   = 0.04
BATCH_SIZE  = 32
MAX_EPOCHS  = 500
PATIENCE    = 60
LR          = 3e-4
WEIGHT_DECAY= 1e-2
N_FOLDS     = 5
EXTREME_W   = 3.0     # extreme-weight multiplier in loss
ORD_COEFF   = 0.4     # ordinal loss contribution

# SMOTER: (lo, hi, target_n_per_fold)  — applied inside each fold's training data
# fold train ≈ 308 samples; we augment extremes to balance
SMOTER_CONFIG = [
    (0,   7, 65),   # severe: ~26/fold → 65
    (7,   9, 55),   # lower-moderate: ~38/fold → 55
    (13,  16, 60),  # above-normal: ~35/fold → 60
    (16,  25, 45),  # high: ~24/fold → 45
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

META = {"video_id", "hb_value", "split", "protocol"}


# ── feature utils ──────────────────────────────────────────────────────────

def get_groups(df):
    feat = [c for c in df.columns if c not in META and c != "anemia"]
    seg0   = [c for c in feat if c.startswith("seg0_")]
    seg1   = [c for c in feat if c.startswith("seg1_")]
    seg2   = [c for c in feat if c.startswith("seg2_")]
    shared = [c for c in feat if c not in seg0 + seg1 + seg2]
    return seg0, seg1, seg2, shared

def raw(df, seg, shared):
    return df[seg + shared].values.astype(np.float32)

def fit_pre(X, y):
    imp = SimpleImputer(strategy="median").fit(X)
    Xi  = imp.transform(X)
    scl = StandardScaler().fit(Xi)
    Xs  = scl.transform(Xi)
    sel = SelectKBest(mutual_info_regression, k=K_FEATURES).fit(Xs, y)
    return imp, scl, sel

def apply_pre(X, imp, scl, sel):
    return sel.transform(scl.transform(imp.transform(X))).astype(np.float32)


# ── SMOTER augmentation ────────────────────────────────────────────────────

def smoter(X0, X1, X2, y, config, k_nn=5, noise=NOISE_STD):
    """
    SMOTER for regression: nearest-neighbour interpolation in feature space
    for under-represented HB bins.
    """
    X0o, X1o, X2o, yo = [X0.copy()], [X1.copy()], [X2.copy()], [y.copy()]

    for lo, hi, tgt in config:
        mask = (y >= lo) & (y < hi)
        n    = mask.sum()
        if n < 2:
            continue
        need = max(0, tgt - n)
        if need == 0:
            continue
        idx = np.where(mask)[0]
        for _ in range(need):
            i   = np.random.choice(idx)
            # k nearest neighbours within the same bin (euclidean on X0)
            d   = np.sum((X0[idx] - X0[i]) ** 2, axis=1)
            d[idx == i] = np.inf
            nn  = idx[np.argsort(d)[:min(k_nn, len(idx) - 1)]]
            j   = np.random.choice(nn)
            a   = np.random.uniform(0.2, 0.8)
            n0  = np.random.randn(X0.shape[1]) * noise
            n1  = np.random.randn(X1.shape[1]) * noise
            n2  = np.random.randn(X2.shape[1]) * noise
            X0o.append((a * X0[i] + (1 - a) * X0[j] + n0).reshape(1, -1))
            X1o.append((a * X1[i] + (1 - a) * X1[j] + n1).reshape(1, -1))
            X2o.append((a * X2[i] + (1 - a) * X2[j] + n2).reshape(1, -1))
            yo.append([a * y[i] + (1 - a) * y[j]])

    return (np.vstack(X0o), np.vstack(X1o), np.vstack(X2o),
            np.concatenate(yo).astype(np.float32))


# ── dataset ────────────────────────────────────────────────────────────────

class HbDataset(Dataset):
    def __init__(self, X0, X1, X2, y, augment=False):
        self.X0 = torch.tensor(X0, dtype=torch.float32)
        self.X1 = torch.tensor(X1, dtype=torch.float32)
        self.X2 = torch.tensor(X2, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32)
        self.aug = augment

    def _noise(self, x):
        if not self.aug: return x
        x = x + torch.randn_like(x) * NOISE_STD
        return x * torch.bernoulli(torch.full_like(x, 0.95))  # 5% feat dropout

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self._noise(self.X0[idx]), self._noise(self.X1[idx]), \
               self._noise(self.X2[idx]), self.y[idx]


# ── model ──────────────────────────────────────────────────────────────────

class BestHbNet(nn.Module):
    """
    3-branch net (one per LED) + ordinal multi-task head.
    Branch: K_FEATURES → BRANCH_DIM
    Trunk:  BRANCH_DIM*3 → HIDDEN_DIM
    Heads:  regression (bounded) + ordinal (3 binary logits)
    """
    def __init__(self):
        super().__init__()
        def branch(in_f):
            return nn.Sequential(
                nn.Linear(in_f, BRANCH_DIM),
                nn.BatchNorm1d(BRANCH_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            )
        self.red    = branch(K_FEATURES)
        self.orange = branch(K_FEATURES)
        self.yellow = branch(K_FEATURES)

        self.trunk = nn.Sequential(
            nn.Linear(BRANCH_DIM * 3, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.BatchNorm1d(HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.reg_head = nn.Linear(HIDDEN_DIM // 2, 1)
        self.ord_head = nn.Linear(HIDDEN_DIM // 2, len(ORD_THRESHOLDS))

    def forward(self, x0, x1, x2):
        shared = self.trunk(torch.cat([self.red(x0),
                                       self.orange(x1),
                                       self.yellow(x2)], dim=1))
        reg = HB_MIN + (HB_MAX - HB_MIN) * torch.sigmoid(
                self.reg_head(shared).squeeze(1))
        ord_logits = self.ord_head(shared)           # (B, 3)
        return reg, ord_logits


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── loss ───────────────────────────────────────────────────────────────────

def extreme_weighted_loss(pred, target, y_mean, y_std, delta=1.5):
    """Huber loss weighted by distance from dataset mean — extremes penalised more."""
    huber = nn.functional.huber_loss(pred, target, delta=delta, reduction="none")
    w     = 1.0 + EXTREME_W * torch.abs(target - y_mean) / (y_std + 1e-8)
    return (huber * w).mean()


def ordinal_loss(ord_logits, y_hb):
    """BCE loss for each clinical threshold."""
    total = 0.0
    for i, thr in enumerate(ORD_THRESHOLDS):
        lbl   = (y_hb < thr).float()
        pos   = lbl.sum() + 1e-8
        neg   = len(lbl) - pos + 1e-8
        pw    = torch.tensor([neg / pos]).to(y_hb.device)
        total += nn.functional.binary_cross_entropy_with_logits(
            ord_logits[:, i], lbl, pos_weight=pw)
    return total / len(ORD_THRESHOLDS)


# ── train / predict ────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader):
    model.eval()
    regs, ords, trues = [], [], []
    for x0, x1, x2, y in loader:
        r, o = model(x0.to(DEVICE), x1.to(DEVICE), x2.to(DEVICE))
        regs.extend(r.cpu().numpy())
        ords.extend(torch.sigmoid(o).cpu().numpy())
        trues.extend(y.numpy())
    return np.array(regs), np.array(ords), np.array(trues)


def train_one_epoch(model, loader, opt, y_mean, y_std):
    model.train()
    total, n = 0.0, 0
    ym = torch.tensor(y_mean, dtype=torch.float32).to(DEVICE)
    ys = torch.tensor(y_std,  dtype=torch.float32).to(DEVICE)
    for x0, x1, x2, y in loader:
        x0, x1, x2, y = x0.to(DEVICE), x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        reg, ord_l = model(x0, x1, x2)
        loss = extreme_weighted_loss(reg, y, ym, ys) + ORD_COEFF * ordinal_loss(ord_l, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


# ── 5-fold ensemble ────────────────────────────────────────────────────────

def run_nn_ensemble(train_df, test_df, seg0, seg1, seg2, shared):
    print(f"\n{'='*65}")
    print(f"  BestHbNet  —  5-fold SMOTER ensemble")
    print(f"{'='*65}")

    y_tr_hb = train_df["hb_value"].values.astype(np.float32)
    y_te_hb = test_df["hb_value"].values.astype(np.float32)
    groups  = train_df["video_id"].values
    strat   = pd.cut(y_tr_hb, bins=[0,7,10,12,25], labels=[0,1,2,3]).astype(int)

    sgkf   = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
    splits = list(sgkf.split(y_tr_hb, strat, groups=groups))

    te_reg_folds = []
    te_ord_folds = []
    val_maes     = []

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        # ── per-fold preprocessors ────────────────────────────────────────
        imp0, scl0, sel0 = fit_pre(raw(train_df.iloc[tr_idx], seg0, shared),
                                    y_tr_hb[tr_idx])
        imp1, scl1, sel1 = fit_pre(raw(train_df.iloc[tr_idx], seg1, shared),
                                    y_tr_hb[tr_idx])
        imp2, scl2, sel2 = fit_pre(raw(train_df.iloc[tr_idx], seg2, shared),
                                    y_tr_hb[tr_idx])

        X0r = apply_pre(raw(train_df.iloc[tr_idx], seg0, shared), imp0, scl0, sel0)
        X1r = apply_pre(raw(train_df.iloc[tr_idx], seg1, shared), imp1, scl1, sel1)
        X2r = apply_pre(raw(train_df.iloc[tr_idx], seg2, shared), imp2, scl2, sel2)
        y_r = y_tr_hb[tr_idx]

        # ── SMOTER augmentation ───────────────────────────────────────────
        X0a, X1a, X2a, ya = smoter(X0r, X1r, X2r, y_r, SMOTER_CONFIG)
        n_synth = len(ya) - len(y_r)
        y_mean, y_std = float(ya.mean()), float(ya.std())

        # shuffle
        perm = np.random.permutation(len(ya))
        X0a, X1a, X2a, ya = X0a[perm], X1a[perm], X2a[perm], ya[perm]

        X0v = apply_pre(raw(train_df.iloc[val_idx], seg0, shared), imp0, scl0, sel0)
        X1v = apply_pre(raw(train_df.iloc[val_idx], seg1, shared), imp1, scl1, sel1)
        X2v = apply_pre(raw(train_df.iloc[val_idx], seg2, shared), imp2, scl2, sel2)
        y_v = y_tr_hb[val_idx]

        X0t = apply_pre(raw(test_df, seg0, shared), imp0, scl0, sel0)
        X1t = apply_pre(raw(test_df, seg1, shared), imp1, scl1, sel1)
        X2t = apply_pre(raw(test_df, seg2, shared), imp2, scl2, sel2)

        tr_ld = DataLoader(HbDataset(X0a, X1a, X2a, ya, augment=True),
                           batch_size=BATCH_SIZE, shuffle=True,
                           drop_last=len(ya) > BATCH_SIZE)
        va_ld = DataLoader(HbDataset(X0v, X1v, X2v, y_v), batch_size=128)
        te_ld = DataLoader(HbDataset(X0t, X1t, X2t, y_te_hb), batch_size=128)

        # ── model ────────────────────────────────────────────────────────
        model = BestHbNet().to(DEVICE)
        opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

        best_mae, best_state, pat = float("inf"), None, 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_one_epoch(model, tr_ld, opt, y_mean, y_std)
            vp, _, vt = predict(model, va_ld)
            vm = mean_absolute_error(vt, vp)
            if vm < best_mae:
                best_mae, pat = vm, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            if pat >= PATIENCE:
                break
            sched.step()

        model.load_state_dict(best_state)
        tp, to, _ = predict(model, te_ld)
        te_reg_folds.append(tp)
        te_ord_folds.append(to)
        val_maes.append(best_mae)
        print(f"    Fold {fold_i+1}  val_MAE={best_mae:.4f}  "
              f"train_n={len(ya)} (+{n_synth} synthetic)  params={count_params(model):,}")

    reg_pred = np.mean(te_reg_folds, axis=0)
    ord_pred = np.mean(te_ord_folds, axis=0)  # shape (95, 3)
    print(f"  CV MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    return reg_pred, ord_pred, y_te_hb


# ── ML baseline for stacking ───────────────────────────────────────────────

def run_ml_baseline(train_df, test_df, seg0, seg1, seg2, shared):
    """GradientBoosting on all 820 features — used for final blend."""
    from sklearn.pipeline import Pipeline as SKPipe
    meta_cols = {"video_id","hb_value","split","protocol"}
    feat = [c for c in train_df.columns if c not in meta_cols and c != "anemia"]

    X_tr = train_df[feat].values.astype(np.float32)
    y_tr = train_df["hb_value"].values
    X_te = test_df[feat].values.astype(np.float32)
    y_te = test_df["hb_value"].values

    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr = imp.transform(X_tr); X_te = imp.transform(X_te)

    # sample weights: extreme-focused
    y_mean, y_std = y_tr.mean(), y_tr.std()
    sw = 1 + EXTREME_W * np.abs(y_tr - y_mean) / (y_std + 1e-8)
    sw = sw / sw.mean()

    gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=RANDOM_SEED)
    gb.fit(X_tr, y_tr, sample_weight=sw)
    ml_pred = gb.predict(X_te)
    ml_mae  = mean_absolute_error(y_te, ml_pred)
    print(f"  GradientBoosting (extreme-weighted) MAE={ml_mae:.4f}")
    return ml_pred


# ── blended prediction ─────────────────────────────────────────────────────

def blend_and_evaluate(nn_pred, ml_pred, ord_pred, y_true, alpha=0.5):
    """
    Blend NN + ML regression predictions.
    Also use ordinal P(HB<12) for anemia classification.
    """
    blended = alpha * nn_pred + (1 - alpha) * ml_pred

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")

    results = {}
    for label, pred in [("NN only", nn_pred),
                         ("ML only", ml_pred),
                         ("Blend (50/50)", blended)]:
        mae  = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        r2   = r2_score(y_true, pred)
        w1   = (np.abs(y_true - pred) <= 1.0).mean()
        w2   = (np.abs(y_true - pred) <= 2.0).mean()
        try:
            auc = roc_auc_score((y_true < ANEMIA_THRESH).astype(int), -pred)
        except: auc = float("nan")
        results[label] = dict(mae=mae, rmse=rmse, r2=r2, w1=w1, w2=w2, auc=auc)

    print(f"\n  {'Model':<22}  {'MAE':>6}  {'RMSE':>6}  {'R²':>7}  {'W±1':>6}  {'W±2':>6}  {'AUC':>6}")
    print(f"  {'─'*68}")
    for label, m in results.items():
        print(f"  {label:<22}  {m['mae']:>6.3f}  {m['rmse']:>6.3f}  "
              f"{m['r2']:>7.3f}  {100*m['w1']:>5.1f}%  {100*m['w2']:>5.1f}%  {m['auc']:>6.3f}")

    # ── use best regression for range analysis ─────────────────────────────
    best_label = min(results, key=lambda k: results[k]["mae"])
    best_pred  = {"NN only": nn_pred, "ML only": ml_pred,
                  "Blend (50/50)": blended}[best_label]
    print(f"\n  Best: {best_label}  →  MAE={results[best_label]['mae']:.3f} g/dL")

    print(f"\n  Error by HB range ({best_label}):")
    print(f"  {'Band':<18}  {'n':>4}  {'PredMn':>7}  {'PredSD':>7}  {'Bias':>7}  {'MAE':>6}  {'W±1':>6}  {'W±2':>6}")
    print(f"  {'─'*75}")
    for lo, hi, lbl in [(0,7,"<7  severe"),(7,10,"7–10 mod"),(10,12,"10–12 mild"),(12,25,">12 normal")]:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0: continue
        p   = best_pred[m]; t = y_true[m]; e = p - t
        w1  = (np.abs(e) <= 1.0).mean()
        w2  = (np.abs(e) <= 2.0).mean()
        print(f"  {lbl:<18}  {m.sum():>4}  {p.mean():>7.2f}  {p.std():>7.2f}  "
              f"{e.mean():>+7.2f}  {np.abs(e).mean():>6.3f}  {100*w1:>5.0f}%  {100*w2:>5.0f}%")

    # ── ordinal anemia classification ──────────────────────────────────────
    # P(HB < 12) is the 3rd ordinal output (index 2)
    anemia_prob = ord_pred[:, 2]
    true_bin    = (y_true < ANEMIA_THRESH).astype(int)

    print(f"\n  Anemia Classification (from ordinal P(HB<12)):")
    auc_cls = roc_auc_score(true_bin, anemia_prob)
    pred_b  = (anemia_prob >= 0.5).astype(int)
    f1_an   = f1_score(true_bin, pred_b, average="binary", zero_division=0)
    cm      = confusion_matrix(true_bin, pred_b, labels=[1, 0])
    sens    = cm[0, 0] / float(true_bin.sum())
    spec    = cm[1, 1] / float((true_bin == 0).sum())
    bal_acc = balanced_accuracy_score(true_bin, pred_b)
    acc     = float((pred_b == true_bin).mean())

    print(f"    AUC         : {auc_cls:.4f}")
    print(f"    Accuracy    : {100*acc:.1f}%")
    print(f"    Sensitivity : {sens:.4f}  (catches {cm[0,0]}/{true_bin.sum()} anemic)")
    print(f"    Specificity : {spec:.4f}")
    print(f"    F1-anemic   : {f1_an:.4f}")
    print(f"    Bal.Acc     : {bal_acc:.4f}")
    print()
    print(classification_report(true_bin, pred_b,
                                target_names=["Normal (≥12)", "Anemic (<12)"],
                                zero_division=0))
    print(f"  Confusion matrix:")
    print(f"                    Pred-Anemic  Pred-Normal")
    print(f"    True-Anemic         {cm[0,0]:>4}         {cm[0,1]:>4}")
    print(f"    True-Normal         {cm[1,0]:>4}         {cm[1,1]:>4}")

    # ── range compression check ───────────────────────────────────────────
    print(f"\n  Range compression (best={best_label}):")
    print(f"    True std    : {y_true.std():.3f}  range [{y_true.min():.1f}, {y_true.max():.1f}]")
    print(f"    Pred std    : {best_pred.std():.3f}  range [{best_pred.min():.1f}, {best_pred.max():.1f}]")
    print(f"    Pred/True   : {best_pred.std()/y_true.std():.3f}  (1.0=perfect)")

    # ── save predictions ──────────────────────────────────────────────────
    pd.DataFrame({
        "hb_true": y_true,
        "nn_pred": nn_pred,
        "ml_pred": ml_pred,
        "blend_pred": blended,
        "p_anemia_7":  ord_pred[:, 0],
        "p_anemia_10": ord_pred[:, 1],
        "p_anemia_12": ord_pred[:, 2],
    }).to_csv(OUT_DIR / "best_predictions.csv", index=False)

    # ── plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. True vs Pred (blend)
    ax = axes[0, 0]
    err_c = np.abs(blended - y_true)
    sc = ax.scatter(y_true, blended, c=err_c, cmap="RdYlGn_r",
                    vmin=0, vmax=4, s=55, edgecolors="white", lw=0.4)
    lim = (min(y_true.min(), blended.min())-0.5,
           max(y_true.max(), blended.max())+0.5)
    ax.plot(lim, lim, "k--", lw=1.2)
    ax.plot(lim, [l+2 for l in lim], "orange", lw=0.8, linestyle=":")
    ax.plot(lim, [l-2 for l in lim], "orange", lw=0.8, linestyle=":", label="±2 g/dL")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True HB (g/dL)"); ax.set_ylabel("Predicted HB (g/dL)")
    ax.set_title(f"True vs Pred (Blend)\nMAE={results['Blend (50/50)']['mae']:.3f}  "
                 f"R²={results['Blend (50/50)']['r2']:.3f}")
    plt.colorbar(sc, ax=ax, label="|Error|")
    ax.legend(fontsize=8)

    # 2. MAE by HB bin
    ax = axes[0, 1]
    bins_lo = np.arange(3, 18, 1)
    bmaes, bns, bw1s, blbls = [], [], [], []
    for lo in bins_lo:
        hi = lo + 1
        m  = (y_true >= lo) & (y_true < hi)
        if m.sum() < 2: continue
        bmaes.append(np.abs(y_true[m] - blended[m]).mean())
        bns.append(m.sum())
        bw1s.append((np.abs(y_true[m]-blended[m])<=1.0).mean()*100)
        blbls.append(f"{lo}–{hi}")
    colors = ["#e74c3c" if v>2.5 else "#f39c12" if v>1.5 else "#2ecc71" for v in bmaes]
    ax.bar(range(len(blbls)), bmaes, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="green",  lw=1.2, linestyle="--", label="1 g/dL")
    ax.axhline(2.0, color="orange", lw=1.2, linestyle="--", label="2 g/dL")
    ax.set_xticks(range(len(blbls))); ax.set_xticklabels(blbls, rotation=45, fontsize=8)
    ax.set_ylabel("MAE (g/dL)"); ax.set_title("MAE per HB bin\n(green<1.5, orange<2.5, red≥2.5)")
    ax.legend(fontsize=8)
    for i, (n, w) in enumerate(zip(bns, bw1s)):
        ax.text(i, bmaes[i]+0.05, f"n={n}", ha="center", fontsize=7)

    # 3. Model comparison bar
    ax = axes[0, 2]
    models = list(results.keys())
    maes   = [results[k]["mae"] for k in models]
    w2s    = [100*results[k]["w2"] for k in models]
    x = np.arange(len(models))
    b1 = ax.bar(x - 0.2, maes, 0.35, label="MAE (g/dL)", color="steelblue", alpha=0.8)
    ax2b = ax.twinx()
    b2 = ax2b.bar(x + 0.2, w2s, 0.35, label="Within ±2 (%)", color="coral", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("MAE (g/dL)"); ax2b.set_ylabel("Within ±2 (%)")
    ax.set_title("Model Comparison")
    lines = [b1, b2]
    ax.legend(lines, ["MAE", "Within ±2%"], fontsize=8)
    for bar, v in zip(b1, maes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{v:.3f}", ha="center", fontsize=8)

    # 4. Prediction mean ± SD per bin (compression check)
    ax = axes[1, 0]
    bin_true_m, bin_pred_m, bin_pred_s, bin_x = [], [], [], []
    for lo in bins_lo:
        hi = lo + 1
        m  = (y_true >= lo) & (y_true < hi)
        if m.sum() < 2: continue
        bin_x.append(y_true[m].mean())
        bin_pred_m.append(blended[m].mean())
        bin_pred_s.append(blended[m].std())
        bin_true_m.append(y_true[m].mean())
    ax.plot(bin_x, bin_x, "k--", lw=1.2, label="Identity (no compression)")
    ax.errorbar(bin_x, bin_pred_m, yerr=bin_pred_s, fmt="o-",
                color="steelblue", capsize=4, lw=1.5, ms=6, label="Pred mean ± SD")
    ax.fill_between(bin_x,
                    [m-s for m,s in zip(bin_pred_m, bin_pred_s)],
                    [m+s for m,s in zip(bin_pred_m, bin_pred_s)],
                    alpha=0.15, color="steelblue")
    ax.set_xlabel("True HB bin mean"); ax.set_ylabel("Predicted HB")
    ax.set_title("Range Compression Check\n(ideal: pred mean = true mean)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. ROC curve
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(true_bin, anemia_prob)
    ax.plot(fpr, tpr, lw=2.5, color="steelblue", label=f"Ordinal P(HB<12)  AUC={auc_cls:.3f}")
    # also plot from regression
    try:
        fpr2, tpr2, _ = roc_curve(true_bin, -blended)
        ax.plot(fpr2, tpr2, lw=1.5, color="coral", linestyle="--",
                label=f"Regression score  AUC={results['Blend (50/50)']['auc']:.3f}")
    except: pass
    ax.plot([0,1],[0,1],"gray",lw=1,linestyle=":")
    ax.axvline(1-spec, color="green", lw=0.8, linestyle=":",
               label=f"Operating point\nSens={sens:.2f} Spec={spec:.2f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — Anemia Detection (HB < 12 g/dL)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 6. Severity band summary
    ax = axes[1, 2]
    bands = ["<7\nSevere", "7–10\nModerate", "10–12\nMild", ">12\nNormal"]
    band_lims = [(0,7),(7,10),(10,12),(12,25)]
    bm_maes, bm_ns, bm_bias, bm_w1 = [], [], [], []
    for lo, hi in band_lims:
        m = (y_true>=lo)&(y_true<hi)
        bm_ns.append(m.sum())
        if m.sum():
            bm_maes.append(np.abs(y_true[m]-blended[m]).mean())
            bm_bias.append((blended[m]-y_true[m]).mean())
            bm_w1.append((np.abs(y_true[m]-blended[m])<=1.0).mean()*100)
        else:
            bm_maes.append(0); bm_bias.append(0); bm_w1.append(0)
    x = np.arange(len(bands))
    bars = ax.bar(x, bm_maes, color=["#e74c3c","#e67e22","#2ecc71","#3498db"],
                  alpha=0.8, edgecolor="white")
    ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax.axhline(2.0, color="black", lw=1, linestyle=":", alpha=0.5)
    for i, (b, w, n, bias) in enumerate(zip(bars, bm_w1, bm_ns, bm_bias)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f"W±1:{w:.0f}%\nn={n}\nbias{bias:+.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(bands, fontsize=9)
    ax.set_ylabel("MAE (g/dL)"); ax.set_title("MAE by Severity Band (Blend)")

    plt.suptitle("NISHAD — Best Results: SMOTER + Extreme-Weighted Loss + Ordinal Multi-task + ML Blend\n"
                 f"MAE={results[best_label]['mae']:.3f} g/dL  "
                 f"AUC={auc_cls:.3f}  Sens={sens:.3f}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "best_results.png", dpi=140)
    plt.close()
    print(f"\n  Saved → {OUT_DIR / 'best_results.png'}")
    print(f"  Saved → {OUT_DIR / 'best_predictions.csv'}")

    return results, auc_cls, sens, spec


# ── entry point ────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  NISHAD — Best HB Estimation Pipeline")
    print("=" * 65)
    print(f"  Device: {DEVICE}")

    train_df = pd.read_csv(FEAT_DIR / "features_train.csv")
    test_df  = pd.read_csv(FEAT_DIR / "features_test.csv")
    seg0, seg1, seg2, shared = get_groups(train_df)

    # ── dataset summary ────────────────────────────────────────────────────
    print(f"\n  Dataset:")
    print(f"    Train: {len(train_df)}  |  Test: {len(test_df)}")
    y_tr = train_df["hb_value"].values
    y_te = test_df["hb_value"].values
    for lo, hi, lbl in [(0,7,"<7  severe"),(7,10,"7-10 mod"),(10,12,"10-12 mild"),(12,25,">12 normal")]:
        mt = (y_tr>=lo)&(y_tr<hi); me = (y_te>=lo)&(y_te<hi)
        print(f"    {lbl:<14}: train={mt.sum():3d} ({100*mt.mean():.0f}%)  "
              f"test={me.sum():3d} ({100*me.mean():.0f}%)")
    print(f"\n  SMOTER augmentation targets (per fold):")
    for lo, hi, tgt in SMOTER_CONFIG:
        print(f"    HB {lo:>2}–{hi:<2}: augment to {tgt}")
    print(f"\n  Model: BestHbNet  params={count_params(BestHbNet()):,}")
    print(f"  Loss : extreme-weighted Huber + {ORD_COEFF}×ordinal BCE")
    print(f"  Blend: 50% NN + 50% GradientBoosting")

    # ── run ────────────────────────────────────────────────────────────────
    nn_pred, ord_pred, y_te_hb = run_nn_ensemble(train_df, test_df,
                                                   seg0, seg1, seg2, shared)
    print(f"\n  Running ML baseline...")
    ml_pred = run_ml_baseline(train_df, test_df, seg0, seg1, seg2, shared)

    blend_and_evaluate(nn_pred, ml_pred, ord_pred, y_te_hb, alpha=0.5)


if __name__ == "__main__":
    main()
