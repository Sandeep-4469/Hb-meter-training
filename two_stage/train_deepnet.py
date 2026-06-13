"""
SmallDeepHBNet — compact 3-branch net on 1062 combined features
================================================================
Architecture  : 3 branches (seg0/seg1/seg2) + shared features
                branch_dim=48 (was 128), 1 residual block (was 2-3)
                Bounded output: 3 + 17*sigmoid(logit) → valid HB range

Features      : 1060 spectral + gender + age = 1062 total
                Shared  = cross-segment + protocol flag + gender + age
                Per-seg = histogram bins + scalar stats + temporal

Imbalance     : band-inverse sample weights (5 bands) +
                oversample extremes (<7 and >14 g/dL) per fold

Training      : 5-fold GroupKFold CV → per-fold mean±std
                AdamW + CosineAnnealing + early stopping (patience=40)
                Huber loss (delta=1.5) for regression

Saves         : saved_models/deepnet_fold{i}.pt  (5 fold models)
                saved_models/deepnet_scalers.pkl  (3 scalers)
                saved_models/deepnet_meta.pkl     (feature columns etc)
"""
import warnings; warnings.filterwarnings("ignore")
import random, sys
from pathlib import Path

import numpy as np, pandas as pd, joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              roc_auc_score, f1_score, balanced_accuracy_score)

AIIMS_CSV  = Path("/data2/sandeep/AIIMS_DATA/hb_values.csv")
RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# ── hyperparameters ───────────────────────────────────────────────────────────
SEED        = 42
N_FOLDS     = 5
BRANCH_DIM  = 48          # was 128 — smaller for 386 samples
N_BLOCKS    = 1           # was 2-3
DROPOUT     = 0.45
BATCH_SIZE  = 32
MAX_EPOCHS  = 300
PATIENCE    = 40
LR          = 3e-4
WEIGHT_DECAY= 1e-2
HB_MIN, HB_MAX = 3.0, 20.0
ANEMIA_THR  = 12.0
BANDS       = [(0,7,"0–7  Severe"),(7,10,"7–10 Moderate"),
               (10,14,"10–14 Mild/Norm"),(14,99,"14+  High")]

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
demo = pd.read_csv(AIIMS_CSV)
demo.columns = demo.columns.str.strip().str.lower().str.replace(" ","_")
demo["gender"] = demo["gender"].str.strip().str.upper()
demo["age"]    = pd.to_numeric(demo["age"], errors="coerce")
demo.loc[demo["age"] > 120, "age"] /= 10
demo["vid_norm"] = (demo["video_name"].str.strip().str.lower()
                    .str.replace(" ","_",regex=False)
                    .str.replace(r"\.(mkv|mp4|avi)$","",regex=True))
demo = demo.dropna(subset=["vid_norm","gender","age"])
demo = demo[demo["gender"].isin(["M","F"])].copy()
demo["gender_bin"] = (demo["gender"]=="M").astype(float)
demo = demo[["vid_norm","gender_bin","age"]].drop_duplicates("vid_norm")

tr_feat = pd.read_csv(RESULTS/"features_combined_train.csv")
te_feat = pd.read_csv(RESULTS/"features_combined_test.csv")
MC = {"video_id","hb_value","split","protocol"}
SPEC_COLS = [c for c in tr_feat.columns if c not in MC]

def merge_demo(df):
    m = df.merge(demo, left_on="video_id", right_on="vid_norm", how="left")
    m["gender_bin"] = m["gender_bin"].fillna(0.0)
    m["age"]        = m["age"].fillna(30.0)
    return m

tr = merge_demo(tr_feat); te = merge_demo(te_feat)
ALL_FEAT = SPEC_COLS + ["gender_bin","age"]

X_tr_raw = tr[ALL_FEAT].astype(float).fillna(0).values
y_tr     = tr["hb_value"].astype(float).values
X_te_raw = te[ALL_FEAT].astype(float).fillna(0).values
y_te     = te["hb_value"].astype(float).values
ids_tr   = tr["video_id"].values

# winsorise
win_lo = np.percentile(X_tr_raw,1,axis=0)
win_hi = np.percentile(X_tr_raw,99,axis=0)
X_tr   = np.clip(X_tr_raw,win_lo,win_hi)
X_te   = np.clip(X_te_raw,win_lo,win_hi)

# ── feature split by segment ─────────────────────────────────────────────────
seg0_idx = [i for i,c in enumerate(ALL_FEAT) if c.startswith("seg0_")]
seg1_idx = [i for i,c in enumerate(ALL_FEAT) if c.startswith("seg1_")]
seg2_idx = [i for i,c in enumerate(ALL_FEAT) if c.startswith("seg2_")]
shared_idx = [i for i in range(len(ALL_FEAT))
              if i not in seg0_idx+seg1_idx+seg2_idx]

print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Total feats: {len(ALL_FEAT)}")
print(f"Seg0: {len(seg0_idx)}  Seg1: {len(seg1_idx)}  "
      f"Seg2: {len(seg2_idx)}  Shared: {len(shared_idx)}")

BRANCH_IN = len(seg0_idx) + len(shared_idx)   # same size for all 3 branches

# ══════════════════════════════════════════════════════════════════════════════
# IMBALANCE
# ══════════════════════════════════════════════════════════════════════════════
def band_weights(y, cap=5.0):
    bins = [0,7,10,12,14,100]; lbl = np.digitize(y,bins)-1
    cnt  = np.bincount(lbl,minlength=5); freq = cnt[lbl]/len(y)
    w    = 1.0/(freq+1e-8); w /= w.min()
    return np.clip(w,None,cap).astype(np.float32)

def oversample_extremes(X, y, lo=7.0, hi=14.0,
                         lo_frac=0.12, hi_frac=0.08, seed=0):
    rng = np.random.default_rng(seed); Xo,yo = X.copy(), y.copy()
    for idx,frac,clo,chi in [(np.where(y<lo)[0], lo_frac, y.min(), lo),
                              (np.where(y>hi)[0], hi_frac, hi, y.max())]:
        n_want = int(len(y)*frac)
        if len(idx)==0 or len(idx)>=n_want: continue
        extra  = n_want-len(idx); ch = rng.choice(idx,extra,replace=True)
        noise  = rng.normal(0, X[idx].std(0)*0.03+1e-8, (extra,X.shape[1]))
        Xo = np.vstack([Xo, X[ch]+noise])
        yo = np.concatenate([yo, np.clip(y[ch]+rng.normal(0,.15,extra),clo,chi)])
    return Xo, yo

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class HbDataset(Dataset):
    def __init__(self, X, y, w=None,
                 s0=seg0_idx, s1=seg1_idx, s2=seg2_idx, sh=shared_idx):
        self.X0 = torch.tensor(X[:,s0+sh], dtype=torch.float32)
        self.X1 = torch.tensor(X[:,s1+sh], dtype=torch.float32)
        self.X2 = torch.tensor(X[:,s2+sh], dtype=torch.float32)
        self.y  = torch.tensor(y,          dtype=torch.float32)
        self.w  = torch.tensor(w if w is not None else np.ones(len(y)),
                               dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.X2[i], self.y[i], self.w[i]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, drop):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_f,out_f), nn.BatchNorm1d(out_f), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_f,out_f), nn.BatchNorm1d(out_f),
        )
        self.skip = nn.Linear(in_f,out_f) if in_f!=out_f else nn.Identity()
        self.act  = nn.GELU(); self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.act(self.main(x) + self.skip(x)))

class SmallDeepHBNet(nn.Module):
    def __init__(self, branch_in, branch_dim=BRANCH_DIM,
                 n_blocks=N_BLOCKS, dropout=DROPOUT):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Linear(branch_in, branch_dim),
                nn.BatchNorm1d(branch_dim), nn.GELU(), nn.Dropout(dropout))
        self.b0 = branch(); self.b1 = branch(); self.b2 = branch()

        combined = branch_dim * 3
        hidden   = [128, 64][:n_blocks]
        self.blocks = nn.ModuleList()
        in_f = combined
        for h in hidden:
            self.blocks.append(ResBlock(in_f, h, dropout)); in_f = h

        self.head = nn.Linear(in_f, 1)

    def forward(self, x0, x1, x2):
        x = torch.cat([self.b0(x0), self.b1(x1), self.b2(x2)], dim=1)
        for blk in self.blocks: x = blk(x)
        logit = self.head(x).squeeze(1)
        return HB_MIN + (HB_MAX - HB_MIN) * torch.sigmoid(logit)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def within(yt,yp,k): return np.mean(np.abs(yt-yp)<=k)*100

def full_metrics(yt, yp):
    mae  = mean_absolute_error(yt,yp)
    rmse = mean_squared_error(yt,yp)**.5
    auc  = roc_auc_score((yt<ANEMIA_THR).astype(int),-yp)
    sens = np.mean(yp[yt<ANEMIA_THR]<ANEMIA_THR)
    ypl  = (yp<ANEMIA_THR).astype(int); ytl=(yt<ANEMIA_THR).astype(int)
    f1   = f1_score(ytl,ypl,zero_division=0)
    bal  = balanced_accuracy_score(ytl,ypl)
    return dict(mae=mae,rmse=rmse,w1=within(yt,yp,1.),w2=within(yt,yp,2.),
                auc=auc,sens=sens,f1=f1,bal=bal,bias=(yp-yt).mean(),
                ratio=yp.std()/(yt.std()+1e-8))

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / INFER
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, opt, crit):
    model.train(); total = 0.0
    for x0,x1,x2,y,w in loader:
        x0,x1,x2,y,w = (t.to(DEVICE) for t in (x0,x1,x2,y,w))
        opt.zero_grad()
        loss = (crit(model(x0,x1,x2), y) * w).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); total += loss.item()
    return total / len(loader)

@torch.no_grad()
def infer(model, loader):
    model.eval(); preds=[]
    for x0,x1,x2,*_ in loader:
        p = model(x0.to(DEVICE),x1.to(DEVICE),x2.to(DEVICE))
        preds.extend(p.cpu().numpy())
    return np.array(preds)

# ══════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV
# ══════════════════════════════════════════════════════════════════════════════
kf = GroupKFold(n_splits=N_FOLDS)
CV_METRICS = ["mae","rmse","w2","auc","sens","f1"]

oof_preds = np.zeros(len(X_tr))
fold_scores = {m:[] for m in CV_METRICS}
fold_models = []   # save state_dicts for ensemble inference

# scale once — fold-level re-scaling would leak through shared scalers
# use full-train scaler for branch inputs (mild leak but stable)
sc0 = StandardScaler().fit(X_tr[:,seg0_idx+shared_idx])
sc1 = StandardScaler().fit(X_tr[:,seg1_idx+shared_idx])
sc2 = StandardScaler().fit(X_tr[:,seg2_idx+shared_idx])

def apply_scalers(X, s0, s1, s2):
    X0 = s0.transform(X[:,seg0_idx+shared_idx])
    X1 = s1.transform(X[:,seg1_idx+shared_idx])
    X2 = s2.transform(X[:,seg2_idx+shared_idx])
    # stack back as single array (dataset will re-split by index)
    n = X.shape[0]; d = X0.shape[1]
    Xs = np.zeros((n, len(ALL_FEAT)), dtype=np.float32)
    # We pass X scaled per branch directly in a combined tensor
    return X0, X1, X2

Xtr0,Xtr1,Xtr2 = apply_scalers(X_tr,sc0,sc1,sc2)
Xte0,Xte1,Xte2 = apply_scalers(X_te,sc0,sc1,sc2)

# build combined scaled arrays for Dataset (it will re-split internally)
# Simpler: concatenate scaled branches back and let Dataset index correctly
# Actually, pass X0/X1/X2 separately to Dataset

class HbDatasetBranch(Dataset):
    def __init__(self, X0, X1, X2, y, w=None):
        self.X0=torch.tensor(X0,dtype=torch.float32)
        self.X1=torch.tensor(X1,dtype=torch.float32)
        self.X2=torch.tensor(X2,dtype=torch.float32)
        self.y =torch.tensor(y, dtype=torch.float32)
        self.w =torch.tensor(w if w is not None else np.ones(len(y)),
                             dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return self.X0[i],self.X1[i],self.X2[i],self.y[i],self.w[i]

print(f"\nModel params: {count_params(SmallDeepHBNet(BRANCH_IN)):,}")
print(f"\n{'='*65}")
print(f"  5-FOLD CV — SmallDeepHBNet")
print(f"{'='*65}")

te_ds = HbDatasetBranch(Xte0,Xte1,Xte2,y_te)
te_loader = DataLoader(te_ds,batch_size=128,shuffle=False)

for fi,(tri,vai) in enumerate(kf.split(X_tr,y_tr,ids_tr)):
    # oversample extremes in this fold
    # We need unscaled X to oversample, then scale
    Xf_raw,yf = oversample_extremes(X_tr[tri],y_tr[tri],seed=SEED+fi)
    wf         = band_weights(yf)
    Xv0        = Xtr0[vai]; Xv1=Xtr1[vai]; Xv2=Xtr2[vai]; yv=y_tr[vai]

    # scale augmented fold train
    sc0f = StandardScaler().fit(Xf_raw[:,seg0_idx+shared_idx])
    sc1f = StandardScaler().fit(Xf_raw[:,seg1_idx+shared_idx])
    sc2f = StandardScaler().fit(Xf_raw[:,seg2_idx+shared_idx])
    Xf0  = sc0f.transform(Xf_raw[:,seg0_idx+shared_idx])
    Xf1  = sc1f.transform(Xf_raw[:,seg1_idx+shared_idx])
    Xf2  = sc2f.transform(Xf_raw[:,seg2_idx+shared_idx])

    # val: use fold's own scaler
    Xv0f = sc0f.transform(X_tr[vai][:,seg0_idx+shared_idx])
    Xv1f = sc1f.transform(X_tr[vai][:,seg1_idx+shared_idx])
    Xv2f = sc2f.transform(X_tr[vai][:,seg2_idx+shared_idx])

    tr_ds  = HbDatasetBranch(Xf0,Xf1,Xf2,yf,wf)
    val_ds = HbDatasetBranch(Xv0f,Xv1f,Xv2f,yv)
    tr_ld  = DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True,
                        drop_last=len(tr_ds)>BATCH_SIZE)
    val_ld = DataLoader(val_ds,batch_size=128,shuffle=False)

    model = SmallDeepHBNet(BRANCH_IN).to(DEVICE)
    crit  = nn.HuberLoss(delta=1.5, reduction="none")
    opt   = optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EPOCHS)

    best_mae, best_state, wait = float("inf"), None, 0
    for ep in range(1,MAX_EPOCHS+1):
        train_epoch(model,tr_ld,opt,crit)
        val_pred = infer(model,val_ld)
        vm = mean_absolute_error(yv,val_pred)
        if vm < best_mae:
            best_mae=vm; wait=0
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
        if wait>=PATIENCE: break
        sched.step()

    model.load_state_dict(best_state)
    oof_preds[vai] = infer(model,val_ld)
    fold_models.append((best_state,sc0f,sc1f,sc2f))

    fm = full_metrics(yv,oof_preds[vai])
    for m in CV_METRICS: fold_scores[m].append(fm[m])
    print(f"  Fold {fi+1}/5  MAE={best_mae:.3f}  AUC={fm['auc']:.3f}  "
          f"Sens={fm['sens']:.3f}  ep={ep}")

means = {m:np.mean(fold_scores[m]) for m in CV_METRICS}
stds  = {m:np.std(fold_scores[m])  for m in CV_METRICS}

print(f"\n{'='*65}")
print("  CV SUMMARY — Mean ± Std")
print(f"{'='*65}")
print(f"  {'Metric':<8}  {'F1':>6}  {'F2':>6}  {'F3':>6}  {'F4':>6}  {'F5':>6}  │  {'Mean':>6}  {'±Std':>6}")
print(f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  │  {'─'*6}  {'─'*6}")
for m in CV_METRICS:
    vals = fold_scores[m]
    row  = f"  {m:<8}  "+"  ".join(f"{v:>6.3f}" for v in vals)
    row += f"  │  {means[m]:>6.3f}  ±{stds[m]:>5.3f}"
    print(row)

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE INFERENCE ON TEST (average all 5 fold models)
# ══════════════════════════════════════════════════════════════════════════════
print("\n  Ensemble test inference (5 fold models) …")
test_fold_preds = []
for fi,(state,s0f,s1f,s2f) in enumerate(fold_models):
    model = SmallDeepHBNet(BRANCH_IN).to(DEVICE)
    model.load_state_dict({k:v.to(DEVICE) for k,v in state.items()})
    Xt0 = s0f.transform(X_te[:,seg0_idx+shared_idx])
    Xt1 = s1f.transform(X_te[:,seg1_idx+shared_idx])
    Xt2 = s2f.transform(X_te[:,seg2_idx+shared_idx])
    ds  = HbDatasetBranch(Xt0,Xt1,Xt2,y_te)
    ld  = DataLoader(ds,batch_size=128,shuffle=False)
    test_fold_preds.append(infer(model,ld))

p_ens = np.mean(test_fold_preds,axis=0)
m_test = full_metrics(y_te,p_ens)

print(f"\n{'='*65}")
print("  TEST SET RESULTS — SmallDeepHBNet (5-fold ensemble)")
print(f"{'='*65}")
print(f"  MAE    = {m_test['mae']:.3f} g/dL")
print(f"  RMSE   = {m_test['rmse']:.3f} g/dL")
print(f"  W±1    = {m_test['w1']:.1f}%")
print(f"  W±2    = {m_test['w2']:.1f}%")
print(f"  AUC    = {m_test['auc']:.3f}")
print(f"  Sens   = {m_test['sens']:.3f}")
print(f"  F1     = {m_test['f1']:.3f}")
print(f"  BalAcc = {m_test['bal']:.3f}")
print(f"  Bias   = {m_test['bias']:+.3f} g/dL")
print(f"  Ratio  = {m_test['ratio']:.3f}  (pred_std/true_std)")

print(f"\n  Per-range breakdown:")
print(f"  {'Range':<16}  {'n':>3}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>6}  {'W±1':>5}  {'W±2':>5}")
print(f"  {'─'*15}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}")
for lo,hi,label in BANDS:
    m=(y_te>=lo)&(y_te<hi); n=int(m.sum())
    if n==0: continue
    mae=mean_absolute_error(y_te[m],p_ens[m])
    rmse=mean_squared_error(y_te[m],p_ens[m])**.5
    bias=(p_ens[m]-y_te[m]).mean()
    w1=within(y_te[m],p_ens[m],1.); w2=within(y_te[m],p_ens[m],2.)
    print(f"  {label:<16}  {n:>3}  {mae:>6.3f}  {rmse:>6.3f}  {bias:>+6.2f}  {w1:>4.0f}%  {w2:>4.0f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
for fi,(state,s0f,s1f,s2f) in enumerate(fold_models):
    torch.save(state, MODELS_DIR/f"deepnet_fold{fi}.pt")

joblib.dump({"scalers_per_fold":[(s0,s1,s2) for _,s0,s1,s2 in fold_models],
             "win_lo":win_lo,"win_hi":win_hi,
             "feat_cols":ALL_FEAT,
             "seg0_idx":seg0_idx,"seg1_idx":seg1_idx,
             "seg2_idx":seg2_idx,"shared_idx":shared_idx,
             "branch_in":BRANCH_IN,
             "hparams":dict(branch_dim=BRANCH_DIM,n_blocks=N_BLOCKS,
                            dropout=DROPOUT)},
            MODELS_DIR/"deepnet_meta.pkl")

print(f"\n  Saved → {MODELS_DIR}/deepnet_fold{{0-4}}.pt")
print(f"  Saved → {MODELS_DIR}/deepnet_meta.pkl")

print(f"\n  Reference (best tree model):")
print(f"  RF (tree): MAE=2.106  AUC=0.651  Sens=0.929  W±2=57%")
print(f"\nDone.")
