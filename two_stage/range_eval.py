"""Per-range evaluation: 0-7, 7-10, 10-14, 14+"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
import lightgbm as lgb; import xgboost as xgb

RESULTS = Path(__file__).parent / "results"
SEED    = 42
BANDS   = [(0,7,"0–7   Severe"),(7,10,"7–10  Moderate"),
           (10,14,"10–14 Mild/Normal"),(14,99,"14+   High")]

# ── data ─────────────────────────────────────────────────────────────────────
tr = pd.read_csv(RESULTS/"features_combined_train.csv")
te = pd.read_csv(RESULTS/"features_combined_test.csv")
mc = {"video_id","hb_value","split","protocol"}
fc = [c for c in tr.columns if c not in mc]

X_tr = tr[fc].astype(float).fillna(0).values;  y_tr = tr["hb_value"].astype(float).values
X_te = te[fc].astype(float).fillna(0).values;  y_te = te["hb_value"].astype(float).values
ids  = tr["video_id"].values

# winsorise (fixes kurtosis blow-up)
lo = np.percentile(X_tr,1,axis=0); hi = np.percentile(X_tr,99,axis=0)
X_tr = np.clip(X_tr,lo,hi);        X_te = np.clip(X_te,lo,hi)

def weights(y,cap=5.):
    bins=[0,7,10,12,100]; lbl=np.digitize(y,bins)-1
    cnt=np.bincount(lbl,minlength=4); freq=cnt[lbl]/len(y)
    w=1/(freq+1e-8); w/=w.min(); return np.clip(w,None,cap)

def oversample(X,y,thr=7.,frac=.15,seed=0):
    rng=np.random.default_rng(seed); idx=np.where(y<thr)[0]
    want=int(len(y)*frac)
    if len(idx)>=want or len(idx)==0: return X,y
    extra=want-len(idx); ch=rng.choice(idx,extra,replace=True)
    noise=rng.normal(0,X[idx].std(0)*.03+1e-8,(extra,X.shape[1]))
    return np.vstack([X,X[ch]+noise]),np.concatenate([y,np.clip(y[ch]+rng.normal(0,.15,extra),3,7)])

w_tr = weights(y_tr)

# ── Model 1: Ensemble RF+LGB+XGB, SelectKBest(80) ────────────────────────────
sel = SelectKBest(f_regression,k=80)
Xtr1 = sel.fit_transform(X_tr,y_tr); Xte1 = sel.transform(X_te)

rf  = RandomForestRegressor(300,max_features=.3,min_samples_leaf=4,random_state=SEED,n_jobs=-1)
lg  = lgb.LGBMRegressor(400,learning_rate=.05,num_leaves=63,subsample=.8,
                         colsample_bytree=.6,reg_lambda=1,random_state=SEED,verbose=-1,n_jobs=-1)
xg  = xgb.XGBRegressor(400,learning_rate=.05,max_depth=5,subsample=.8,
                        colsample_bytree=.6,reg_lambda=1,random_state=SEED,verbosity=0,n_jobs=-1)
rf.fit(Xtr1,y_tr,sample_weight=w_tr)
lg.fit(Xtr1,y_tr,sample_weight=w_tr)
xg.fit(Xtr1,y_tr,sample_weight=w_tr)
p_ens = (rf.predict(Xte1)+lg.predict(Xte1)+xg.predict(Xte1))/3

# save ensemble artefacts
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump({"rf":rf,"lgb":lg,"xgb":xg,"selector":sel,
             "winsor_lo":lo,"winsor_hi":hi},
            MODELS_DIR/"ensemble_best_mae.pkl")
print(f"  Saved ensemble → {MODELS_DIR}/ensemble_best_mae.pkl")

# ── Model 2: PLS(n=5) direct ─────────────────────────────────────────────────
pls = PLSRegression(n_components=5,scale=True)
pls.fit(X_tr,y_tr); p_pls = pls.predict(X_te).ravel()
joblib.dump({"pls":pls,"winsor_lo":lo,"winsor_hi":hi},
            MODELS_DIR/"pls_best_auc.pkl")
print(f"  Saved PLS     → {MODELS_DIR}/pls_best_auc.pkl")

# ── Model 3: Meta-stacking Ridge on RF+LGB+XGB OOF ───────────────────────────
kf = GroupKFold(n_splits=5)
oof=np.zeros((len(X_tr),3)); tst=np.zeros((len(X_te),3))
for fi,(tri,vai) in enumerate(kf.split(X_tr,y_tr,ids)):
    Xf,yf = oversample(X_tr[tri],y_tr[tri],seed=SEED+fi)
    wf    = weights(yf)
    sf=SelectKBest(f_regression,k=80); Xfs=sf.fit_transform(Xf,yf)
    Xvs=sf.transform(X_tr[vai]); Xts=sf.transform(X_te)
    for mi,(m,Xfit,Xval,Xtest) in enumerate([
        (RandomForestRegressor(300,max_features=.3,min_samples_leaf=4,random_state=SEED,n_jobs=-1),Xf,X_tr[vai],X_te),
        (lgb.LGBMRegressor(400,learning_rate=.05,num_leaves=63,subsample=.8,colsample_bytree=.6,reg_lambda=1,random_state=SEED,verbose=-1,n_jobs=-1),Xf,X_tr[vai],X_te),
        (xgb.XGBRegressor(400,learning_rate=.05,max_depth=5,subsample=.8,colsample_bytree=.6,reg_lambda=1,random_state=SEED,verbosity=0,n_jobs=-1),Xf,X_tr[vai],X_te),
    ]):
        m.fit(Xfit,yf,sample_weight=wf)
        oof[vai,mi]=m.predict(Xval); tst[:,mi]+=m.predict(Xtest)/5
tst_fin=tst  # already averaged over folds
sc=StandardScaler(); meta=Ridge(.5)
meta.fit(sc.fit_transform(oof),y_tr,sample_weight=w_tr)
p_meta=meta.predict(sc.transform(tst_fin))
joblib.dump({"oof_models_per_fold":"rebuilt_at_inference","meta_ridge":meta,
             "meta_scaler":sc,"base_model_order":["rf","lgb","xgb"],
             "winsor_lo":lo,"winsor_hi":hi},
            MODELS_DIR/"meta_stack_best_sens.pkl")
print(f"  Saved meta    → {MODELS_DIR}/meta_stack_best_sens.pkl")

# ── evaluation helpers ────────────────────────────────────────────────────────
def w_pct(yt,yp,k): return np.mean(np.abs(yt-yp)<=k)*100
def auc(yt,yp):     return roc_auc_score((yt<12).astype(int),-yp)
def sens(yt,yp):    m=yt<12; return np.mean(yp[m]<12)

def band_row(yt,yp,lo,hi):
    m=(yt>=lo)&(yt<hi); n=m.sum()
    if n==0: return n,"-","-","-","-","-"
    mae=mean_absolute_error(yt[m],yp[m])
    rmse=mean_squared_error(yt[m],yp[m])**.5
    bias=(yp[m]-yt[m]).mean()
    return n,mae,rmse,bias,w_pct(yt[m],yp[m],1),w_pct(yt[m],yp[m],2)

# ── print ─────────────────────────────────────────────────────────────────────
models=[
    ("Ensemble  RF+LGB+XGB   (best MAE)",  y_te,p_ens),
    ("PLS Direct n=5          (best AUC)", y_te,p_pls),
    ("Meta-Stack Ridge OOF    (best Sens)",y_te,p_meta),
]

print("\n"+"="*88)
print("  PER-RANGE RESULTS — Test set (n=95)  |  95 samples total")
print("="*88)
print(f"\n  Test set breakdown:  0–7: n=7 (7%)  |  7–10: n=41 (43%)  |  10–14: n=38 (40%)  |  14+: n=9 (9%)")
print()

for mname,yt,yp in models:
    mae=mean_absolute_error(yt,yp)
    rmse=mean_squared_error(yt,yp)**.5
    w2=w_pct(yt,yp,2); a=auc(yt,yp); s=sens(yt,yp)
    print("─"*88)
    print(f"  {mname}")
    print(f"  Overall → MAE={mae:.3f}  RMSE={rmse:.3f}  W±2={w2:.0f}%  AUC={a:.3f}  Sens={s:.3f}")
    print(f"  {'Range':<22}  {'n':>3}  {'MAE':>6}  {'RMSE':>6}  {'Bias':>6}  {'W±1':>6}  {'W±2':>6}")
    print(f"  {'─'*21}  {'─'*3}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    for lo,hi,label in BANDS:
        n,mae_b,rmse_b,bias_b,w1_b,w2_b = band_row(yt,yp,lo,hi)
        if isinstance(mae_b,str):
            print(f"  {label:<22}  {n:>3}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>6}")
        else:
            print(f"  {label:<22}  {n:>3}  {mae_b:>6.3f}  {rmse_b:>6.3f}  {bias_b:>+6.2f}  {w1_b:>5.0f}%  {w2_b:>5.0f}%")
    print()

print("="*88)
print("  Bias = mean(pred − truth). Positive = over-estimate. Negative = under-estimate.")
print("="*88+"\n")
