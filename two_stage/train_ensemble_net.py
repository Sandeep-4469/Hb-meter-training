"""
Neural Net + Tree Ensemble — Stacking, Cascading, Blending
===========================================================
Step 1 : TinyDeepHBNet  — branch_dim=16, no residual blocks (~28K params)
Step 2 : Base models     — TinyNet, RF, LGB, SVR, PLS+Ridge  (5-fold OOF)
Step 3 : Ensemble strategies
           A. WeightedMean    — inverse-OOF-MAE weights
           B. Stack-Ridge     — Ridge meta on 5-dim OOF
           C. Stack-LGB       — LGB meta on 5-dim OOF
           D. Cascade         — LGB residual correction on (features + TinyNet OOF)
           E. DoubleStack     — Stack-Ridge on [Stack-Ridge, RF, LGB] second level

All models saved; full test + OOF metrics with per-range breakdown.
"""
import warnings; warnings.filterwarnings("ignore")
import random
from pathlib import Path

import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              roc_auc_score, f1_score, balanced_accuracy_score)
import lightgbm as lgb

AIIMS_CSV  = Path("/data2/sandeep/AIIMS_DATA/hb_values.csv")
RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

SEED=42; N_FOLDS=5; ANEMIA_THR=12.0
BRANCH_DIM=16; DROPOUT=0.5
BATCH=32; MAX_EP=300; PATIENCE=40; LR=3e-4; WD=1e-2
HB_MIN,HB_MAX=3.0,20.0
BANDS=[(0,7,"0–7  Severe"),(7,10,"7–10 Moderate"),
       (10,14,"10–14 Mild/Norm"),(14,99,"14+  High")]

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
demo=pd.read_csv(AIIMS_CSV)
demo.columns=demo.columns.str.strip().str.lower().str.replace(" ","_")
demo["gender"]=demo["gender"].str.strip().str.upper()
demo["age"]=pd.to_numeric(demo["age"],errors="coerce")
demo.loc[demo["age"]>120,"age"]/=10
demo["vid_norm"]=(demo["video_name"].str.strip().str.lower()
                  .str.replace(" ","_",regex=False)
                  .str.replace(r"\.(mkv|mp4|avi)$","",regex=True))
demo=demo.dropna(subset=["vid_norm","gender","age"])
demo=demo[demo["gender"].isin(["M","F"])].copy()
demo["gender_bin"]=(demo["gender"]=="M").astype(float)
demo=demo[["vid_norm","gender_bin","age"]].drop_duplicates("vid_norm")

def merge(df):
    m=df.merge(demo,left_on="video_id",right_on="vid_norm",how="left")
    m["gender_bin"]=m["gender_bin"].fillna(0.); m["age"]=m["age"].fillna(30.)
    return m

tr=merge(pd.read_csv(RESULTS/"features_combined_train.csv"))
te=merge(pd.read_csv(RESULTS/"features_combined_test.csv"))
MC={"video_id","hb_value","split","protocol","vid_norm"}
SPEC=[c for c in tr.columns if c not in MC]
ALL_FEAT=SPEC+["gender_bin","age"]

X_tr_raw=tr[ALL_FEAT].astype(float).fillna(0).values; y_tr=tr["hb_value"].astype(float).values
X_te_raw=te[ALL_FEAT].astype(float).fillna(0).values; y_te=te["hb_value"].astype(float).values
ids_tr=tr["video_id"].values

lo=np.percentile(X_tr_raw,1,axis=0); hi=np.percentile(X_tr_raw,99,axis=0)
X_tr=np.clip(X_tr_raw,lo,hi); X_te=np.clip(X_te_raw,lo,hi)

seg0=[i for i,c in enumerate(ALL_FEAT) if c.startswith("seg0_")]
seg1=[i for i,c in enumerate(ALL_FEAT) if c.startswith("seg1_")]
seg2=[i for i,c in enumerate(ALL_FEAT) if c.startswith("seg2_")]
shr =[i for i in range(len(ALL_FEAT)) if i not in seg0+seg1+seg2]
BIN =len(seg0)+len(shr)   # branch input size

print(f"Train:{len(X_tr)} Test:{len(X_te)} Feats:{len(ALL_FEAT)}")
print(f"Seg0:{len(seg0)} Seg1:{len(seg1)} Seg2:{len(seg2)} Shared:{len(shr)}")

# ── imbalance helpers ─────────────────────────────────────────────────────────
def bw(y,cap=5.):
    b=[0,7,10,12,14,100]; l=np.digitize(y,b)-1
    c=np.bincount(l,minlength=5); f=c[l]/len(y)
    w=1/(f+1e-8); w/=w.min(); return np.clip(w,None,cap).astype(np.float32)

def oversample(X,y,seed=0):
    rng=np.random.default_rng(seed); Xo,yo=X.copy(),y.copy()
    for idx,frac,clo,chi in [(np.where(y<7)[0],.12,y.min(),7.),
                              (np.where(y>14)[0],.08,14.,y.max())]:
        nw=int(len(y)*frac)
        if len(idx)==0 or len(idx)>=nw: continue
        ex=nw-len(idx); ch=rng.choice(idx,ex,replace=True)
        noise=rng.normal(0,X[idx].std(0)*.03+1e-8,(ex,X.shape[1]))
        Xo=np.vstack([Xo,X[ch]+noise])
        yo=np.concatenate([yo,np.clip(y[ch]+rng.normal(0,.15,ex),clo,chi)])
    return Xo,yo

# ── metrics ───────────────────────────────────────────────────────────────────
def W(yt,yp,k): return np.mean(np.abs(yt-yp)<=k)*100
def metrics(yt,yp):
    mae=mean_absolute_error(yt,yp); rmse=mean_squared_error(yt,yp)**.5
    auc=roc_auc_score((yt<ANEMIA_THR).astype(int),-yp)
    sens=np.mean(yp[yt<ANEMIA_THR]<ANEMIA_THR)
    ypl=(yp<ANEMIA_THR).astype(int); ytl=(yt<ANEMIA_THR).astype(int)
    return dict(mae=mae,rmse=rmse,w1=W(yt,yp,1.),w2=W(yt,yp,2.),
                auc=auc,sens=sens,f1=f1_score(ytl,ypl,zero_division=0),
                bal=balanced_accuracy_score(ytl,ypl),
                bias=(yp-yt).mean(),ratio=yp.std()/(yt.std()+1e-8))

def print_m(tag,m):
    print(f"  {tag}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
          f"W±2={m['w2']:.0f}% AUC={m['auc']:.3f} "
          f"Sens={m['sens']:.3f} F1={m['f1']:.3f} "
          f"Bias={m['bias']:+.3f} Ratio={m['ratio']:.3f}")

def band_print(yt,yp,prefix=""):
    for lo_,hi_,lb in BANDS:
        m=(yt>=lo_)&(yt<hi_); n=int(m.sum())
        if n==0: continue
        mae=mean_absolute_error(yt[m],yp[m])
        bias=(yp[m]-yt[m]).mean()
        w1=W(yt[m],yp[m],1.); w2=W(yt[m],yp[m],2.)
        print(f"  {prefix}{lb:<16} n={n:3d}  MAE={mae:.3f}  "
              f"Bias={bias:+.2f}  W±1={w1:.0f}%  W±2={w2:.0f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TINY NET
# ══════════════════════════════════════════════════════════════════════════════
class TinyDeepHBNet(nn.Module):
    def __init__(self,branch_in,dim=BRANCH_DIM,drop=DROPOUT):
        super().__init__()
        def br(): return nn.Sequential(
            nn.Linear(branch_in,dim), nn.BatchNorm1d(dim),
            nn.GELU(), nn.Dropout(drop))
        self.b0=br(); self.b1=br(); self.b2=br()
        self.head=nn.Sequential(
            nn.Linear(dim*3,dim*2), nn.BatchNorm1d(dim*2),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim*2,1))
    def forward(self,x0,x1,x2):
        x=torch.cat([self.b0(x0),self.b1(x1),self.b2(x2)],1)
        return HB_MIN+(HB_MAX-HB_MIN)*torch.sigmoid(self.head(x).squeeze(1))

n_params=sum(p.numel() for p in TinyDeepHBNet(BIN).parameters() if p.requires_grad)
print(f"TinyDeepHBNet params: {n_params:,}")

class DS(Dataset):
    def __init__(self,X0,X1,X2,y,w=None):
        self.X0=torch.tensor(X0,dtype=torch.float32)
        self.X1=torch.tensor(X1,dtype=torch.float32)
        self.X2=torch.tensor(X2,dtype=torch.float32)
        self.y =torch.tensor(y, dtype=torch.float32)
        self.w =torch.tensor(w if w is not None else np.ones(len(y)),dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X0[i],self.X1[i],self.X2[i],self.y[i],self.w[i]

def mk_loaders(X0,X1,X2,y,w=None,shuffle=True):
    ds=DS(X0,X1,X2,y,w)
    return DataLoader(ds,batch_size=BATCH,shuffle=shuffle,
                      drop_last=shuffle and len(ds)>BATCH)

@torch.no_grad()
def infer_net(model,X0,X1,X2,y):
    model.eval()
    ds=DS(X0,X1,X2,y); ld=DataLoader(ds,128,shuffle=False)
    p=[]
    for x0,x1,x2,*_ in ld:
        p.extend(model(x0.to(DEV),x1.to(DEV),x2.to(DEV)).cpu().numpy())
    return np.array(p)

def train_fold_net(X0tr,X1tr,X2tr,ytr,wtr, X0va,X1va,X2va,yva):
    model=TinyDeepHBNet(BIN).to(DEV)
    crit=nn.HuberLoss(delta=1.5,reduction="none")
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EP)
    tr_ld=mk_loaders(X0tr,X1tr,X2tr,ytr,wtr)
    best_mae,best_st,wait=float("inf"),None,0
    for ep in range(1,MAX_EP+1):
        model.train()
        for x0,x1,x2,y,w in tr_ld:
            x0,x1,x2,y,w=(t.to(DEV) for t in (x0,x1,x2,y,w))
            opt.zero_grad()
            loss=(crit(model(x0,x1,x2),y)*w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        sch.step()
        vp=infer_net(model,X0va,X1va,X2va,yva)
        vm=mean_absolute_error(yva,vp)
        if vm<best_mae:
            best_mae=vm; wait=0
            best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
        if wait>=PATIENCE: break
    model.load_state_dict(best_st)
    return model,best_mae,ep

# ── per-fold scalers for net ──────────────────────────────────────────────────
def scale_branches(X,sc0,sc1,sc2):
    return (sc0.transform(X[:,seg0+shr]),
            sc1.transform(X[:,seg1+shr]),
            sc2.transform(X[:,seg2+shr]))

# ══════════════════════════════════════════════════════════════════════════════
# 5-FOLD BASE MODEL OOF
# ══════════════════════════════════════════════════════════════════════════════
kf=GroupKFold(n_splits=N_FOLDS)
oof={"net":np.zeros(len(X_tr)),"rf":np.zeros(len(X_tr)),
     "lgb":np.zeros(len(X_tr)),"svr":np.zeros(len(X_tr)),
     "pls":np.zeros(len(X_tr))}
net_fold_models=[]   # (state, sc0, sc1, sc2)
tree_fold_models={"rf":[],"lgb":[]}

print(f"\n{'='*65}\n  5-FOLD BASE MODEL OOF\n{'='*65}")

for fi,(tri,vai) in enumerate(kf.split(X_tr,y_tr,ids_tr)):
    Xf,yf=oversample(X_tr[tri],y_tr[tri],seed=SEED+fi)
    wf=bw(yf); Xv=X_tr[vai]; yv=y_tr[vai]

    # ── TinyNet ──────────────────────────────────────────────────────────────
    sc0f=StandardScaler().fit(Xf[:,seg0+shr])
    sc1f=StandardScaler().fit(Xf[:,seg1+shr])
    sc2f=StandardScaler().fit(Xf[:,seg2+shr])
    X0f,X1f,X2f=scale_branches(Xf,sc0f,sc1f,sc2f)
    X0v,X1v,X2v=scale_branches(Xv,sc0f,sc1f,sc2f)
    net_m,net_mae,ep=train_fold_net(X0f,X1f,X2f,yf,wf, X0v,X1v,X2v,yv)
    oof["net"][vai]=infer_net(net_m,X0v,X1v,X2v,yv)
    net_fold_models.append(({k:v.cpu().clone() for k,v in net_m.state_dict().items()},
                             sc0f,sc1f,sc2f))

    # ── RF ───────────────────────────────────────────────────────────────────
    rf_f=RandomForestRegressor(n_estimators=300,max_features=.3,
                                min_samples_leaf=4,random_state=SEED,n_jobs=-1)
    rf_f.fit(Xf,yf,sample_weight=wf); oof["rf"][vai]=rf_f.predict(Xv)
    tree_fold_models["rf"].append(rf_f)

    # ── LGB ──────────────────────────────────────────────────────────────────
    lg_f=lgb.LGBMRegressor(n_estimators=400,learning_rate=.05,num_leaves=63,
                             subsample=.8,colsample_bytree=.6,reg_lambda=1.,
                             random_state=SEED,verbose=-1,n_jobs=-1)
    lg_f.fit(Xf,yf,sample_weight=wf); oof["lgb"][vai]=lg_f.predict(Xv)
    tree_fold_models["lgb"].append(lg_f)

    # ── SVR ──────────────────────────────────────────────────────────────────
    sel=SelectKBest(f_regression,k=80); sc=StandardScaler()
    Xfs=sc.fit_transform(sel.fit_transform(Xf,yf))
    svr_f=SVR(kernel="rbf",C=10,epsilon=.2)
    svr_f.fit(Xfs,yf)
    oof["svr"][vai]=svr_f.predict(sc.transform(sel.transform(Xv)))

    # ── PLS+Ridge ────────────────────────────────────────────────────────────
    pls_f=PLSRegression(n_components=5,scale=True); pls_f.fit(Xf,yf)
    rdg_f=Ridge(alpha=.1); rdg_f.fit(pls_f.transform(Xf),yf,sample_weight=wf)
    oof["pls"][vai]=rdg_f.predict(pls_f.transform(Xv))

    print(f"  Fold {fi+1}/5  Net={net_mae:.3f}(ep={ep})  "
          f"RF={mean_absolute_error(yv,oof['rf'][vai]):.3f}  "
          f"LGB={mean_absolute_error(yv,oof['lgb'][vai]):.3f}")

print("\n  OOF summary:")
for name,p in oof.items():
    m=metrics(y_tr,p)
    print(f"  {name:<8} MAE={m['mae']:.3f}  AUC={m['auc']:.3f}  Sens={m['sens']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL TEST PREDICTIONS (full-train base models)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}\n  FINAL BASE MODELS — test predictions\n{'='*65}")

Xaug,yaug=oversample(X_tr,y_tr,seed=SEED); waug=bw(yaug)

# TinyNet ensemble = average of 5 fold models on test
te_net_preds=[]
for fi,(state,sc0f,sc1f,sc2f) in enumerate(net_fold_models):
    m=TinyDeepHBNet(BIN).to(DEV)
    m.load_state_dict({k:v.to(DEV) for k,v in state.items()})
    X0t,X1t,X2t=scale_branches(X_te,sc0f,sc1f,sc2f)
    te_net_preds.append(infer_net(m,X0t,X1t,X2t,y_te))
te_preds={"net":np.mean(te_net_preds,axis=0)}

# RF final
rf_fin=RandomForestRegressor(n_estimators=300,max_features=.3,
                              min_samples_leaf=4,random_state=SEED,n_jobs=-1)
rf_fin.fit(Xaug,yaug,sample_weight=waug); te_preds["rf"]=rf_fin.predict(X_te)

# LGB final
lg_fin=lgb.LGBMRegressor(n_estimators=400,learning_rate=.05,num_leaves=63,
                           subsample=.8,colsample_bytree=.6,reg_lambda=1.,
                           random_state=SEED,verbose=-1,n_jobs=-1)
lg_fin.fit(Xaug,yaug,sample_weight=waug); te_preds["lgb"]=lg_fin.predict(X_te)

# SVR final
sel_fin=SelectKBest(f_regression,k=80); sc_fin=StandardScaler()
Xaug_sel=sc_fin.fit_transform(sel_fin.fit_transform(Xaug,yaug))
svr_fin=SVR(kernel="rbf",C=10,epsilon=.2); svr_fin.fit(Xaug_sel,yaug)
te_preds["svr"]=svr_fin.predict(sc_fin.transform(sel_fin.transform(X_te)))

# PLS+Ridge final
pls_fin=PLSRegression(n_components=5,scale=True); pls_fin.fit(Xaug,yaug)
rdg_fin=Ridge(alpha=.1)
rdg_fin.fit(pls_fin.transform(Xaug),yaug,sample_weight=waug)
te_preds["pls"]=rdg_fin.predict(pls_fin.transform(X_te))

for name,p in te_preds.items(): print_m(name,metrics(y_te,p))

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}\n  ENSEMBLE STRATEGIES\n{'='*65}")

BASE_NAMES=["net","rf","lgb","svr","pls"]
oof_mat  =np.column_stack([oof[n]  for n in BASE_NAMES])
te_mat   =np.column_stack([te_preds[n] for n in BASE_NAMES])
w_full   =bw(y_tr)

# ── A. Inverse-MAE weighted mean ─────────────────────────────────────────────
oof_maes=np.array([mean_absolute_error(y_tr,oof[n]) for n in BASE_NAMES])
inv_w=1/(oof_maes+1e-8); inv_w/=inv_w.sum()
p_wmean_oof=(oof_mat*inv_w).sum(1)
p_wmean_te =(te_mat *inv_w).sum(1)
print(f"\n  A. WeightedMean (inv-MAE weights):")
print(f"     weights: "+", ".join(f"{n}={w:.3f}" for n,w in zip(BASE_NAMES,inv_w)))
print_m("OOF ",metrics(y_tr,p_wmean_oof))
print_m("Test",metrics(y_te,p_wmean_te))

# ── B. Stack-Ridge ────────────────────────────────────────────────────────────
sc_b=StandardScaler()
meta_b=Ridge(alpha=.5)
meta_b.fit(sc_b.fit_transform(oof_mat),y_tr,sample_weight=w_full)
p_sridge_oof=meta_b.predict(sc_b.transform(oof_mat))
p_sridge_te =meta_b.predict(sc_b.transform(te_mat))
print(f"\n  B. Stack-Ridge (meta on 5-dim OOF):")
print_m("OOF ",metrics(y_tr,p_sridge_oof))
print_m("Test",metrics(y_te,p_sridge_te))

# ── C. Stack-LGB ─────────────────────────────────────────────────────────────
meta_c=lgb.LGBMRegressor(n_estimators=200,learning_rate=.03,num_leaves=7,
                           reg_lambda=5.,random_state=SEED,verbose=-1,n_jobs=-1)
meta_c.fit(sc_b.transform(oof_mat),y_tr,sample_weight=w_full)
p_slgb_oof=meta_c.predict(sc_b.transform(oof_mat))
p_slgb_te =meta_c.predict(sc_b.transform(te_mat))
print(f"\n  C. Stack-LGB (meta on 5-dim OOF):")
print_m("OOF ",metrics(y_tr,p_slgb_oof))
print_m("Test",metrics(y_te,p_slgb_te))

# ── D. Cascade: TinyNet → LGB residual correction ────────────────────────────
# Stage 1: TinyNet OOF predictions
# Stage 2: LGB trained on (original features + TinyNet OOF pred) → predicts residual
resid_tr = y_tr - oof["net"]   # residuals to correct
X_casc_tr = np.hstack([X_tr, oof["net"].reshape(-1,1)])
w_casc    = bw(y_tr)

resid_lgb=lgb.LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=31,
                              subsample=.8,colsample_bytree=.6,reg_lambda=2.,
                              random_state=SEED,verbose=-1,n_jobs=-1)
resid_lgb.fit(X_casc_tr, resid_tr, sample_weight=w_casc)

# OOF cascade
p_casc_oof = oof["net"] + resid_lgb.predict(X_casc_tr)
# Test cascade — use final net prediction (ensemble of 5 folds)
X_casc_te = np.hstack([X_te, te_preds["net"].reshape(-1,1)])
p_casc_te  = te_preds["net"] + resid_lgb.predict(X_casc_te)
print(f"\n  D. Cascade (TinyNet → LGB residual correction):")
print_m("OOF ",metrics(y_tr,p_casc_oof))
print_m("Test",metrics(y_te,p_casc_te))

# ── E. DoubleStack: best first-level metas → Ridge second level ───────────────
# Level-1 OOF: [Stack-Ridge, RF, LGB]  (using their OOF)
l1_oof = np.column_stack([p_sridge_oof, oof["rf"], oof["lgb"]])
l1_te  = np.column_stack([p_sridge_te,  te_preds["rf"], te_preds["lgb"]])
sc_e   = StandardScaler()
meta_e = Ridge(alpha=.5)
meta_e.fit(sc_e.fit_transform(l1_oof), y_tr, sample_weight=w_full)
p_dstack_oof=meta_e.predict(sc_e.transform(l1_oof))
p_dstack_te =meta_e.predict(sc_e.transform(l1_te))
print(f"\n  E. DoubleStack (Ridge on [StackRidge, RF, LGB]):")
print_m("OOF ",metrics(y_tr,p_dstack_oof))
print_m("Test",metrics(y_te,p_dstack_te))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
all_results=[
    ("TinyNet",     y_te, te_preds["net"]),
    ("RF",          y_te, te_preds["rf"]),
    ("LGB",         y_te, te_preds["lgb"]),
    ("SVR",         y_te, te_preds["svr"]),
    ("PLS+Ridge",   y_te, te_preds["pls"]),
    ("A.WeightedMean", y_te, p_wmean_te),
    ("B.StackRidge",   y_te, p_sridge_te),
    ("C.StackLGB",     y_te, p_slgb_te),
    ("D.Cascade",      y_te, p_casc_te),
    ("E.DoubleStack",  y_te, p_dstack_te),
]
print(f"\n{'='*90}")
print("  FINAL SUMMARY — Test set (n=95)")
print(f"{'='*90}")
print(f"  {'Model':<20} {'MAE':>6} {'RMSE':>6} {'W±2':>5} {'AUC':>6} "
      f"{'Sens':>6} {'F1':>6} {'Bias':>6} {'Ratio':>6}")
print(f"  {'─'*19} {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
for name,yt,yp in all_results:
    m=metrics(yt,yp)
    print(f"  {name:<20} {m['mae']:>6.3f} {m['rmse']:>6.3f} "
          f"{m['w2']:>4.0f}% {m['auc']:>6.3f} {m['sens']:>6.3f} "
          f"{m['f1']:>6.3f} {m['bias']:>+6.2f} {m['ratio']:>6.3f}")

best_name=min(all_results,key=lambda x:metrics(x[1],x[2])["mae"])[0]
print(f"\n  Best MAE: {best_name}")

print(f"\n  Per-range (WeightedMean — best ensemble):")
band_print(y_te, p_wmean_te)

print(f"\n  Per-range (RF — best single model):")
band_print(y_te, te_preds["rf"])

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
for fi,(state,sc0f,sc1f,sc2f) in enumerate(net_fold_models):
    torch.save(state, MODELS_DIR/f"tinynet_fold{fi}.pt")

joblib.dump({"scalers":[(s0,s1,s2) for _,s0,s1,s2 in net_fold_models],
             "win_lo":lo,"win_hi":hi,"feat_cols":ALL_FEAT,
             "seg0":seg0,"seg1":seg1,"seg2":seg2,"shr":shr,"BIN":BIN,
             "hparams":dict(branch_dim=BRANCH_DIM,dropout=DROPOUT)},
            MODELS_DIR/"tinynet_meta.pkl")

joblib.dump({"rf":rf_fin,"lgb":lg_fin,"svr":svr_fin,
             "svr_sel":sel_fin,"svr_sc":sc_fin,
             "pls":pls_fin,"pls_ridge":rdg_fin,
             "win_lo":lo,"win_hi":hi,"feat_cols":ALL_FEAT},
            MODELS_DIR/"ensemble_base_models.pkl")

joblib.dump({"stack_ridge":meta_b,"stack_ridge_sc":sc_b,
             "stack_lgb":meta_c,
             "cascade_lgb":resid_lgb,
             "dstack_ridge":meta_e,"dstack_sc":sc_e,
             "inv_weights":inv_w,"base_names":BASE_NAMES},
            MODELS_DIR/"ensemble_meta_models.pkl")

print(f"\n  Saved → tinynet_fold0-4.pt  +  tinynet_meta.pkl")
print(f"  Saved → ensemble_base_models.pkl")
print(f"  Saved → ensemble_meta_models.pkl")
print("\nDone.")
