"""
Final Regression Training — imbalance-aware, gender+age included
=================================================================
Features  : 1060 spectral + gender + age = 1062 total
Imbalance : 1) Inverse HB-band sample weights (5 bands: <7,7-10,10-12,12-14,14+)
             2) Oversampling extremes (<7 and >14 g/dL) to ≥12% each
             3) For linear models: SelectKBest(k=80) from winsorised features
             4) PLS(n=5) as supervised DR (best AUC from prior study)

Models    :
  Linear   : Ridge, ElasticNet, SVR(RBF), PLS(n=5)+Ridge
  Tree     : RandomForest, ExtraTrees, GradientBoosting
  Boosting : LightGBM, XGBoost
  Ensemble : RF+LGB+XGB mean (best MAE prior)
  Stack    : Ridge meta on RF+LGB+XGB OOF (best Sens prior)

Evaluation: 5-fold GroupKFold → per-fold mean±std + test set
            Metrics: MAE, RMSE, W±1, W±2, AUC, Sens, F1, BalAcc
            Per-range: 0-7, 7-10, 10-14, 14+

Saves     : saved_models/reg_<name>.pkl  for every model
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              roc_auc_score, f1_score, balanced_accuracy_score)
import lightgbm as lgb
import xgboost as xgb

AIIMS_CSV  = Path("/data2/sandeep/AIIMS_DATA/hb_values.csv")
RESULTS    = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

SEED       = 42
N_FOLDS    = 5
K_SELECT   = 80
N_PLS      = 5
ANEMIA_THR = 12.0
BANDS      = [(0,7,"0–7  Severe"),(7,10,"7–10 Moderate"),
              (10,14,"10–14 Mild/Norm"),(14,99,"14+  High")]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
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
demo["gender_bin"] = (demo["gender"]=="M").astype(int)
demo = demo[["vid_norm","gender_bin","age"]].drop_duplicates("vid_norm")

tr_feat = pd.read_csv(RESULTS/"features_combined_train.csv")
te_feat = pd.read_csv(RESULTS/"features_combined_test.csv")
feat_cols = [c for c in tr_feat.columns
             if c not in {"video_id","hb_value","split","protocol"}]

def merge(df):
    m = df.merge(demo, left_on="video_id", right_on="vid_norm", how="left")
    m["gender_bin"] = m["gender_bin"].fillna(0)
    m["age"]        = m["age"].fillna(30)
    return m

tr = merge(tr_feat); te = merge(te_feat)
FEAT_EXT = feat_cols + ["gender_bin","age"]

X_tr_raw = tr[FEAT_EXT].astype(float).fillna(0).values
y_tr     = tr["hb_value"].astype(float).values
X_te_raw = te[FEAT_EXT].astype(float).fillna(0).values
y_te     = te["hb_value"].astype(float).values
ids_tr   = tr["video_id"].values

# winsorise
win_lo = np.percentile(X_tr_raw,1,axis=0)
win_hi = np.percentile(X_tr_raw,99,axis=0)
X_tr   = np.clip(X_tr_raw,win_lo,win_hi)
X_te   = np.clip(X_te_raw,win_lo,win_hi)

print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Features: {X_tr.shape[1]}")
print(f"HB train  mean={y_tr.mean():.2f}  std={y_tr.std():.2f}  "
      f"range=[{y_tr.min():.1f},{y_tr.max():.1f}]")

# ══════════════════════════════════════════════════════════════════════════════
# IMBALANCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def band_weights(y, cap=5.0):
    """Inverse-frequency weights across 5 HB bands."""
    bins  = [0,7,10,12,14,100]
    lbl   = np.digitize(y,bins)-1
    cnt   = np.bincount(lbl, minlength=5)
    freq  = cnt[lbl]/len(y)
    w     = 1.0/(freq+1e-8); w /= w.min()
    return np.clip(w,None,cap)

def oversample_extremes(X, y, lo_thr=7.0, hi_thr=14.0,
                        lo_frac=0.12, hi_frac=0.08, seed=0):
    """Oversample severe (<lo_thr) and high (>hi_thr) cases with jitter."""
    rng  = np.random.default_rng(seed)
    Xo,yo = X.copy(), y.copy()
    for idx,frac,clip_lo,clip_hi in [
        (np.where(y<lo_thr)[0],  lo_frac, y.min(), lo_thr),
        (np.where(y>hi_thr)[0],  hi_frac, hi_thr,  y.max()),
    ]:
        n_want = int(len(y)*frac)
        if len(idx)==0 or len(idx)>=n_want: continue
        extra  = n_want - len(idx)
        chosen = rng.choice(idx,extra,replace=True)
        noise  = rng.normal(0, X[idx].std(0)*0.03+1e-8, (extra,X.shape[1]))
        y_noise= np.clip(y[chosen]+rng.normal(0,.15,extra), clip_lo, clip_hi)
        Xo = np.vstack([Xo, X[chosen]+noise])
        yo = np.concatenate([yo, y_noise])
    return Xo, yo

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def within(yt,yp,k): return np.mean(np.abs(yt-yp)<=k)*100

def full_metrics(yt, yp):
    mae  = mean_absolute_error(yt,yp)
    rmse = mean_squared_error(yt,yp)**.5
    w1   = within(yt,yp,1.); w2=within(yt,yp,2.)
    bias = (yp-yt).mean()
    ratio= yp.std()/(yt.std()+1e-8)
    auc  = roc_auc_score((yt<ANEMIA_THR).astype(int),-yp)
    sens = np.mean(yp[yt<ANEMIA_THR]<ANEMIA_THR)
    yp_lbl = (yp<ANEMIA_THR).astype(int)
    yt_lbl = (yt<ANEMIA_THR).astype(int)
    f1   = f1_score(yt_lbl,yp_lbl,zero_division=0)
    bal  = balanced_accuracy_score(yt_lbl,yp_lbl)
    return dict(mae=mae,rmse=rmse,w1=w1,w2=w2,bias=bias,
                ratio=ratio,auc=auc,sens=sens,f1=f1,bal=bal)

def band_row(yt,yp,lo,hi):
    m=(yt>=lo)&(yt<hi); n=int(m.sum())
    if n==0: return n,None,None,None,None,None
    return (n, mean_absolute_error(yt[m],yp[m]),
            mean_squared_error(yt[m],yp[m])**.5,
            (yp[m]-yt[m]).mean(), within(yt[m],yp[m],1.), within(yt[m],yp[m],2.))

def print_metrics(tag, m):
    print(f"  {tag}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
          f"W±1={m['w1']:.0f}% W±2={m['w2']:.0f}% "
          f"AUC={m['auc']:.3f} Sens={m['sens']:.3f} "
          f"F1={m['f1']:.3f} Bias={m['bias']:+.3f} Ratio={m['ratio']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# Each entry: (name, preproc_type, model, supports_sample_weight)
# preproc_type: 'linear'=SelectKBest+Scaler, 'pls'=PLS5, None=raw
# ══════════════════════════════════════════════════════════════════════════════
def model_configs():
    return [
        ("Ridge",     "linear", Ridge(alpha=1.0),                                          True),
        ("ElasticNet","linear", ElasticNet(alpha=0.005,l1_ratio=0.5,max_iter=5000),        True),
        ("SVR",       "linear", SVR(kernel="rbf",C=10,epsilon=0.2),                        False),
        ("PLS+Ridge", "pls",    Ridge(alpha=0.1),                                          True),
        ("RF",        None,     RandomForestRegressor(n_estimators=300,max_features=0.3,
                                    min_samples_leaf=4,random_state=SEED,n_jobs=-1),       True),
        ("ExtraTrees",None,     ExtraTreesRegressor(n_estimators=300,max_features=0.3,
                                    min_samples_leaf=4,random_state=SEED,n_jobs=-1),       True),
        ("GBM",       None,     GradientBoostingRegressor(n_estimators=300,
                                    learning_rate=0.05,max_depth=4,subsample=0.8,
                                    random_state=SEED),                                    True),
        ("LGB",       None,     lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,
                                    num_leaves=63,subsample=0.8,colsample_bytree=0.6,
                                    reg_lambda=1.0,min_child_samples=10,
                                    random_state=SEED,verbose=-1,n_jobs=-1),               True),
        ("XGB",       None,     xgb.XGBRegressor(n_estimators=400,learning_rate=0.05,
                                    max_depth=5,subsample=0.8,colsample_bytree=0.6,
                                    reg_lambda=1.0,min_child_weight=5,
                                    random_state=SEED,verbosity=0,n_jobs=-1),              True),
    ]

# ══════════════════════════════════════════════════════════════════════════════
# 5-FOLD CV
# ══════════════════════════════════════════════════════════════════════════════
kf      = GroupKFold(n_splits=N_FOLDS)
configs = model_configs()
CV_METRICS = ["mae","rmse","w2","auc","sens","f1"]

all_oof   = {}   # name → oof predictions
fold_data = {}   # name → {metric: [fold values]}

print(f"\n{'='*72}")
print("  5-FOLD CV (GroupKFold by video_id) — per-fold mean±std")
print(f"{'='*72}")

for mname, ptype, model, use_sw in configs:
    oof       = np.zeros(len(X_tr))
    fscores   = {m:[] for m in CV_METRICS}

    for fi,(tri,vai) in enumerate(kf.split(X_tr,y_tr,ids_tr)):
        Xf,yf = oversample_extremes(X_tr[tri],y_tr[tri],seed=SEED+fi)
        wf    = band_weights(yf)
        Xv    = X_tr[vai]; yv = y_tr[vai]

        if ptype=="linear":
            sc=StandardScaler(); sel=SelectKBest(f_regression,k=K_SELECT)
            Xf_t = sc.fit_transform(sel.fit_transform(Xf,yf))
            Xv_t = sc.transform(sel.transform(Xv))
            model.fit(Xf_t,yf,sample_weight=wf) if use_sw else model.fit(Xf_t,yf)
            oof[vai]=model.predict(Xv_t)
        elif ptype=="pls":
            pls=PLSRegression(n_components=N_PLS,scale=True); pls.fit(Xf,yf)
            model.fit(pls.transform(Xf),yf,sample_weight=wf)
            oof[vai]=model.predict(pls.transform(Xv))
        else:
            model.fit(Xf,yf,sample_weight=wf) if use_sw else model.fit(Xf,yf)
            oof[vai]=model.predict(Xv)

        fm = full_metrics(yv,oof[vai])
        for m in CV_METRICS: fscores[m].append(fm[m])

    means = {m:np.mean(fscores[m]) for m in CV_METRICS}
    stds  = {m:np.std(fscores[m])  for m in CV_METRICS}
    all_oof[mname]   = oof
    fold_data[mname] = {"means":means,"stds":stds,"folds":fscores}

    print(f"\n  ── {mname}")
    print(f"  {'Metric':<8} {'F1':>7} {'F2':>7} {'F3':>7} {'F4':>7} {'F5':>7}  │  {'Mean':>7} {'±Std':>7}")
    print(f"  {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}  │  {'─'*7} {'─'*7}")
    for m in CV_METRICS:
        vals=fscores[m]
        row=f"  {m:<8} "+"  ".join(f"{v:>7.3f}" for v in vals)
        row+=f"  │  {means[m]:>7.3f} ±{stds[m]:>6.3f}"
        print(row)

# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE + META-STACK OOF
# ══════════════════════════════════════════════════════════════════════════════
# Ensemble: mean of RF, LGB, XGB OOF
p_ens_oof = (all_oof["RF"]+all_oof["LGB"]+all_oof["XGB"])/3
m_ens_oof  = full_metrics(y_tr,p_ens_oof)
fold_data["Ensemble(RF+LGB+XGB)"] = {"oof_overall": m_ens_oof}

# Meta-stack: Ridge on [RF, LGB, XGB] OOF
oof_stack = np.column_stack([all_oof["RF"],all_oof["LGB"],all_oof["XGB"]])
sc_meta   = StandardScaler()
meta_rdg  = Ridge(alpha=0.5)
w_full    = band_weights(y_tr)
meta_rdg.fit(sc_meta.fit_transform(oof_stack),y_tr,sample_weight=w_full)
p_meta_oof = meta_rdg.predict(sc_meta.transform(oof_stack))
m_meta_oof = full_metrics(y_tr,p_meta_oof)
fold_data["MetaStack(Ridge)"] = {"oof_overall": m_meta_oof}

print(f"\n  ── Ensemble (RF+LGB+XGB mean)")
print_metrics("OOF", m_ens_oof)
print(f"\n  ── Meta-Stack (Ridge on RF+LGB+XGB OOF)")
print_metrics("OOF", m_meta_oof)

# ══════════════════════════════════════════════════════════════════════════════
# RETRAIN ON FULL TRAIN, EVALUATE TEST, SAVE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  FINAL MODELS — full train → test set + save")
print(f"{'='*72}")

X_aug, y_aug = oversample_extremes(X_tr,y_tr,seed=SEED)
w_aug        = band_weights(y_aug)
w_tr_full    = band_weights(y_tr)

test_preds = {}   # name → test predictions
saved_artefacts = {}

# individual models
for mname, ptype, model, use_sw in configs:
    if ptype=="linear":
        sc=StandardScaler(); sel=SelectKBest(f_regression,k=K_SELECT)
        Xtr_t=sc.fit_transform(sel.fit_transform(X_aug,y_aug))
        Xte_t=sc.transform(sel.transform(X_te))
        model.fit(Xtr_t,y_aug,sample_weight=w_aug) if use_sw else model.fit(Xtr_t,y_aug)
        p_te=model.predict(Xte_t)
        artefact={"model":model,"selector":sel,"scaler":sc,
                  "win_lo":win_lo,"win_hi":win_hi,"feat_cols":FEAT_EXT}
    elif ptype=="pls":
        pls=PLSRegression(n_components=N_PLS,scale=True); pls.fit(X_aug,y_aug)
        model.fit(pls.transform(X_aug),y_aug,sample_weight=w_aug)
        p_te=model.predict(pls.transform(X_te))
        artefact={"model":model,"pls":pls,
                  "win_lo":win_lo,"win_hi":win_hi,"feat_cols":FEAT_EXT}
    else:
        model.fit(X_aug,y_aug,sample_weight=w_aug) if use_sw else model.fit(X_aug,y_aug)
        p_te=model.predict(X_te)
        artefact={"model":model,"win_lo":win_lo,"win_hi":win_hi,"feat_cols":FEAT_EXT}

    test_preds[mname]=p_te
    save_path=MODELS_DIR/f"reg_{mname.lower().replace('+','_').replace(' ','_')}.pkl"
    joblib.dump(artefact,save_path)
    saved_artefacts[mname]=save_path

    m=full_metrics(y_te,p_te)
    oof_m=fold_data[mname]
    print(f"\n  ── {mname}  (saved → {save_path.name})")
    print(f"  OOF : MAE={oof_m['means']['mae']:.3f}±{oof_m['stds']['mae']:.3f}  "
          f"AUC={oof_m['means']['auc']:.3f}±{oof_m['stds']['auc']:.3f}  "
          f"Sens={oof_m['means']['sens']:.3f}±{oof_m['stds']['sens']:.3f}")
    print_metrics("Test", m)

# Ensemble
rf_m=joblib.load(MODELS_DIR/"reg_rf.pkl")["model"]
lg_m=joblib.load(MODELS_DIR/"reg_lgb.pkl")["model"]
xg_m=joblib.load(MODELS_DIR/"reg_xgb.pkl")["model"]
p_ens_te=(rf_m.predict(X_aug[:0+len(X_te)])*0   # placeholder
          + test_preds["RF"]+test_preds["LGB"]+test_preds["XGB"])/3
joblib.dump({"rf":rf_m,"lgb":lg_m,"xgb":xg_m,
             "win_lo":win_lo,"win_hi":win_hi,"feat_cols":FEAT_EXT},
            MODELS_DIR/"reg_ensemble_rf_lgb_xgb.pkl")
m_ens_te=full_metrics(y_te,p_ens_te)
print(f"\n  ── Ensemble RF+LGB+XGB  (saved → reg_ensemble_rf_lgb_xgb.pkl)")
print_metrics("Test", m_ens_te)

# Meta-stack — retrain base models on full train for consistent inference
rf_s=RandomForestRegressor(n_estimators=300,max_features=0.3,min_samples_leaf=4,
                            random_state=SEED,n_jobs=-1)
lg_s=lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=63,
                         subsample=0.8,colsample_bytree=0.6,reg_lambda=1.0,
                         random_state=SEED,verbose=-1,n_jobs=-1)
xg_s=xgb.XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=5,
                        subsample=0.8,colsample_bytree=0.6,reg_lambda=1.0,
                        random_state=SEED,verbosity=0,n_jobs=-1)
rf_s.fit(X_aug,y_aug,sample_weight=w_aug)
lg_s.fit(X_aug,y_aug,sample_weight=w_aug)
xg_s.fit(X_aug,y_aug,sample_weight=w_aug)
te_stack=np.column_stack([rf_s.predict(X_te),lg_s.predict(X_te),xg_s.predict(X_te)])
p_meta_te=meta_rdg.predict(sc_meta.transform(te_stack))
joblib.dump({"rf":rf_s,"lgb":lg_s,"xgb":xg_s,
             "meta_ridge":meta_rdg,"meta_scaler":sc_meta,
             "win_lo":win_lo,"win_hi":win_hi,"feat_cols":FEAT_EXT},
            MODELS_DIR/"reg_metastack_ridge.pkl")
m_meta_te=full_metrics(y_te,p_meta_te)
print(f"\n  ── Meta-Stack Ridge  (saved → reg_metastack_ridge.pkl)")
print_metrics("Test", m_meta_te)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  FINAL SUMMARY — Test set (n=95)  |  1062 features (1060 spectral + gender + age)")
print(f"  Imbalance: band-inverse weights + oversample extremes (<7 and >14 g/dL)")
print(f"{'='*90}")
print(f"  {'Model':<22} {'MAE':>6} {'RMSE':>6} {'W±1':>5} {'W±2':>5} "
      f"{'AUC':>6} {'Sens':>6} {'F1':>6} {'Bias':>6} {'Ratio':>6}")
print(f"  {'─'*21} {'─'*6} {'─'*6} {'─'*5} {'─'*5} "
      f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

all_test = list(test_preds.items()) + [
    ("Ensemble",  p_ens_te),
    ("MetaStack", p_meta_te),
]
for mname,yp in all_test:
    m=full_metrics(y_te,yp)
    print(f"  {mname:<22} {m['mae']:>6.3f} {m['rmse']:>6.3f} "
          f"{m['w1']:>4.0f}% {m['w2']:>4.0f}% "
          f"{m['auc']:>6.3f} {m['sens']:>6.3f} {m['f1']:>6.3f} "
          f"{m['bias']:>+6.2f} {m['ratio']:>6.3f}")

print(f"\n{'='*90}")
print("  Per-range breakdown (test set) — best model per metric marked")
print(f"{'='*90}")
print(f"  {'Model':<22} {'Range':<16} {'n':>3} {'MAE':>6} {'RMSE':>6} "
      f"{'Bias':>6} {'W±1':>5} {'W±2':>5}")
print(f"  {'─'*21} {'─'*15} {'─'*3} {'─'*6} {'─'*6} {'─'*6} {'─'*5} {'─'*5}")
for mname,yp in all_test:
    for lo,hi,label in BANDS:
        n,mae,rmse,bias,w1,w2=band_row(y_te,yp,lo,hi)
        if mae is None: continue
        print(f"  {mname:<22} {label:<16} {n:>3} {mae:>6.3f} {rmse:>6.3f} "
              f"{bias:>+6.2f} {w1:>4.0f}% {w2:>4.0f}%")
    print()

print(f"\n  Saved models → {MODELS_DIR}/reg_*.pkl")
print("Done.")
