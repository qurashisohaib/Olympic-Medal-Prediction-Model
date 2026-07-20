# eval_metrics.py — Olympic Medal Prediction Model: metric & validation audit
# Run inside the project folder (needs athlete_events.csv + noc_regions.csv).
#   pip install -r requirements.txt
#   python3 eval_metrics.py | tee results/metric_report.txt

import numpy as np, pandas as pd, sklearn
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, precision_recall_fscore_support, confusion_matrix)
print("sklearn", sklearn.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)

# ---------- load + features (mirrors fbladataanalysis.py exactly) ----------
ath = pd.read_csv("athlete_events.csv"); noc = pd.read_csv("noc_regions.csv")
df = ath.merge(noc, on="NOC", how="left")
for c in ["Height","Weight","Age","Year"]: df[c] = pd.to_numeric(df[c], errors="coerce")
df["NOC"] = df["NOC"].replace({"URS":"RUS"})
df["Medalist"] = (~df["Medal"].isna()).astype(int)
df["BMI"] = df["Weight"]/(df["Height"]/100)**2
df.loc[(df["BMI"]<10)|(df["BMI"]>60),"BMI"] = np.nan
df = df[(df["Season"]=="Summer") & (df["Year"]>=1960)].copy()
m = df.copy()
m["Years_since_1960"] = m["Year"]-1960
m["NOC_Year_Athletes"]       = m.groupby(["NOC","Year"])["ID"].transform("nunique")
m["NOC_Sport_Year_Athletes"] = m.groupby(["NOC","Sport","Year"])["ID"].transform("nunique")
m["Event_Year_Athletes"]     = m.groupby(["Event","Year"])["ID"].transform("nunique")
m["Athlete_Event_Count"]     = m.groupby(["ID","Year"])["Event"].transform("nunique")
m["NOC_Event_Year_Athletes"] = m.groupby(["NOC","Event","Year"])["ID"].transform("nunique")
def prior(keys):
    g = (m.groupby(keys+["Year"]).agg(e=("Medalist","size"), md=("Medalist","sum"))
           .reset_index().sort_values(keys+["Year"]))
    g["pe"] = g.groupby(keys)["e"].cumsum()-g["e"]; g["pm"] = g.groupby(keys)["md"].cumsum()-g["md"]
    g["rate"] = g["pm"]/g["pe"].replace(0,np.nan); return g
for keys,nm in [(["NOC"],"Prior_NOC_Medal_Rate"),(["NOC","Sport"],"Prior_NOC_Sport_Medal_Rate"),(["Event"],"Prior_Event_Medal_Rate")]:
    g = prior(keys); m = m.merge(g[keys+["Year","rate"]].rename(columns={"rate":nm}), on=keys+["Year"], how="left")

NUM = ["Age","Height","Weight","BMI","Years_since_1960","NOC_Year_Athletes","NOC_Sport_Year_Athletes",
       "Event_Year_Athletes","NOC_Event_Year_Athletes","Athlete_Event_Count",
       "Prior_NOC_Medal_Rate","Prior_NOC_Sport_Medal_Rate","Prior_Event_Medal_Rate"]
CAT = ["Sex","Sport","NOC"]
for c in CAT: m[c] = m[c].astype("category")
X = m[NUM+CAT].copy(); y = m["Medalist"].astype(int).values
ID = m["ID"].values; YEAR = m["Year"].values
catmask = [c in CAT for c in X.columns]

def gbm(mask):
    return HistGradientBoostingClassifier(categorical_features=mask, learning_rate=0.05, max_iter=600,
        max_leaf_nodes=63, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=42)

def panel(yte, p, tag):
    base = yte.mean(); order = np.argsort(p)[::-1]; ys = yte[order]
    roc = roc_auc_score(yte,p); pr = average_precision_score(yte,p)
    print(f"\n[{tag}]  n_test={len(yte):,}  base_rate={base:.3f}")
    print(f"  ROC-AUC {roc:.3f} (no-skill .500)  |  PR-AUC {pr:.3f} (no-skill {base:.3f} -> {pr/base:.1f}x chance)")
    print(f"  Brier {brier_score_loss(yte,p):.4f} (base {base*(1-base):.4f})  |  LogLoss {log_loss(yte,p):.4f} (base {log_loss(yte,np.full_like(p,base)):.4f})")
    for f in (0.05,0.10,0.20):
        k=int(len(ys)*f); print(f"  P@top{int(f*100):>2}% {ys[:k].mean():.3f}  (captures {ys[:k].sum()/ys.sum():.0%} of real medalists)")
    return roc, pr

# ---------- 1) FULL model, random 80/20 (their current setup) ----------
print("\n"+"="*70+"\n1) FULL MODEL — random 80/20 split (current setup)\n"+"="*70)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
mdl = gbm(catmask); mdl.fit(Xtr,ytr); p = mdl.predict_proba(Xte)[:,1]
roc_main, pr_main = panel(yte,p,"FULL / random split")
pr,rc,f1,_ = precision_recall_fscore_support(yte,(p>=0.5).astype(int),average="binary",zero_division=0)
tn,fp,fn,tp = confusion_matrix(yte,(p>=0.5).astype(int),labels=[0,1]).ravel()
print(f"  @0.50 threshold: precision={pr:.3f} recall={rc:.3f} F1={f1:.3f}")
print(f"  confusion@0.5: TP={tp} FP={fp} FN={fn} TN={tn} | accuracy={(tp+tn)/len(yte):.3f}  vs 'predict-none' {1-yte.mean():.3f}")
bins=np.clip((p*10).astype(int),0,9); ece=0.0
print("  calibration by decile (pred vs actual):")
for b in range(10):
    s=bins==b
    if s.sum()==0: continue
    mp=p[s].mean(); ap_=yte[s].mean(); ece+=s.mean()*abs(mp-ap_)
    print(f"    bin{b}: pred {mp:.3f}  actual {ap_:.3f}  n={s.sum()}")
print(f"  ECE (expected calibration error) = {ece:.4f}")

# ---------- 2) ABLATION: body vs country/context ----------
print("\n"+"="*70+"\n2) ABLATION — what carries the signal?\n"+"="*70)
BODY = ["Age","Height","Weight","BMI","Sex","Sport"]
CONTEXT = ["Years_since_1960","NOC_Year_Athletes","NOC_Sport_Year_Athletes","Event_Year_Athletes",
           "NOC_Event_Year_Athletes","Athlete_Event_Count","Prior_NOC_Medal_Rate",
           "Prior_NOC_Sport_Medal_Rate","Prior_Event_Medal_Rate","NOC"]
def fit_subset(cols, tag):
    cm=[c in CAT for c in cols]; mm=gbm(cm); mm.fit(Xtr[cols],ytr); pp=mm.predict_proba(Xte[cols])[:,1]
    print(f"  {tag:<38} ROC {roc_auc_score(yte,pp):.3f} | PR {average_precision_score(yte,pp):.3f}")
fit_subset(BODY, "BODY+demo (Age/Ht/Wt/BMI/Sex/Sport)")
fit_subset(CONTEXT, "CONTEXT (country/event/history)")
fit_subset([c for c in NUM+CAT if c!="NOC_Event_Year_Athletes"], "FULL minus team-size feature")
print(f"  {'FULL (all features)':<38} ROC {roc_main:.3f} | PR {pr_main:.3f}")

# ---------- 3) LEAKAGE-SAFE VALIDATION ----------
print("\n"+"="*70+"\n3) LEAKAGE-SAFE VALIDATION (does 0.92 survive a harder test?)\n"+"="*70)
tr=YEAR<=2008; te=YEAR>=2012
mt=gbm(catmask); mt.fit(X[tr],y[tr]); pt=mt.predict_proba(X[te])[:,1]
panel(y[te],pt,"TEMPORAL: train<=2008, test 2012+2016 (predicts the FUTURE)")
gkf=GroupKFold(n_splits=5); rocs=[]; prs=[]
for tri,tei in gkf.split(X,y,groups=ID):
    g=gbm(catmask); g.fit(X.iloc[tri],y[tri]); pp=g.predict_proba(X.iloc[tei])[:,1]
    rocs.append(roc_auc_score(y[tei],pp)); prs.append(average_precision_score(y[tei],pp))
print(f"\n[GROUP 5-fold by athlete ID — same athlete never in train+test]")
print(f"  ROC-AUC {np.mean(rocs):.3f} +/- {np.std(rocs):.3f}  |  PR-AUC {np.mean(prs):.3f} +/- {np.std(prs):.3f}")

# ---------- 4) PERMUTATION IMPORTANCE (full model) ----------
print("\n"+"="*70+"\n4) PERMUTATION IMPORTANCE (drop in ROC-AUC when feature shuffled)\n"+"="*70)
perm=permutation_importance(mdl,Xte,yte,scoring="roc_auc",n_repeats=5,random_state=42,n_jobs=-1)
print(pd.Series(perm.importances_mean,index=X.columns).sort_values(ascending=False).round(4).to_string())

# ---------- 5) UNIVARIATE AUC (each feature alone, exact) ----------
print("\n"+"="*70+"\n5) UNIVARIATE ROC-AUC (each numeric feature alone)\n"+"="*70)
for f in NUM:
    x=pd.to_numeric(m[f],errors="coerce").values; s=~np.isnan(x)
    r=pd.Series(x[s]).rank().values; yy=y[s]; n1=yy.sum(); n0=len(yy)-n1
    print(f"  {f:<28} {(r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0):.3f}")
print("\nDONE")
