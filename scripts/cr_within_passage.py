"""Strictest leakage control: AUROC computed ONLY over same-passage response pairs.
Removes every between-passage channel, stricter than passage-grouped CV."""
import sys, os, json, time, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from _cr_common import load, apriori_layer, auroc_signed as auroc, seeded_group_folds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

rows=[json.loads(l) for l in open('data/ragtruth.jsonl')]
norm=lambda x:" ".join(x.lower().split())
_c={}; ctx_id=np.array([_c.setdefault(norm(r['context']),len(_c)) for r in rows])

def within_auroc(s, y, g):
    """Mann-Whitney concordance restricted to same-group pairs, signed (no polarity flip)."""
    num=0.0; den=0.0
    for p in np.unique(g):
        m=g==p; sp=s[m]; yp=y[m]
        a=sp[yp==1]; b=sp[yp==0]
        if len(a)==0 or len(b)==0: continue
        d=a[:,None]-b[None,:]
        num+=(d>0).sum()+0.5*(d==0).sum(); den+=d.size
    v=num/den
    return float(v), int(den)

def fit_fold(X,y,tr,te):
    sc=StandardScaler().fit(X[tr])
    clf=LogisticRegression(C=0.5,max_iter=2000).fit(sc.transform(X[tr]),y[tr])
    return te, clf.decision_function(sc.transform(X[te]))

res={}
with Parallel(n_jobs=3,backend='loky',max_nbytes='1M') as par:
    for m in ("qwen25_7b","llama31_8b","gemma2_9b"):
        z,meta=load(f"{m}_rt"); L=apriori_layer(meta)
        X=z[f'last_{L}'].astype(np.float64); y=z['faithful'].astype(int); iid=z['item_id']
        gp=ctx_id[iid]; ov=z['lexical_overlap'].astype(np.float64); nli=z['nli_entail_prob'].astype(np.float64)
        conf=z['mean_maxsoftmax'].astype(np.float64)
        d={}
        for gname,g in (("item_id",iid),("passage",gp)):
            wa=[]
            for seed in range(3):
                folds=list(seeded_group_folds(g,5,seed))
                oof=np.zeros(len(y))
                for te,s in par(delayed(fit_fold)(X,y,tr,te) for tr,te in folds): oof[te]=s
                w,den=within_auroc(oof,y,gp); wa.append(w)
                print(f"  {m} {gname} seed{seed}: pooled={auroc(oof,y):.4f} within_passage={w:.4f} (pairs={den})",flush=True)
            d[gname]={"within_passage_mean":float(np.mean(wa)),"per_seed":[float(x) for x in wa]}
        for nm,sc_ in (("lexical_overlap",ov),("nli",nli),("max_softmax",conf)):
            w,den=within_auroc(sc_,y,gp)
            d[nm]={"pooled":auroc(sc_,y),"within_passage":w,"pairs":den}
            print(f"  {m} baseline {nm}: pooled={auroc(sc_,y):.4f} within_passage={w:.4f}",flush=True)
        res[m]=d
        json.dump(res,open('runs/cr_within_passage.json','w'),indent=2)
        z.drop(f'last_{L}'); del X
print("WITHIN DONE",flush=True)
