"""Passage-leakage audit of the RAGTruth in-domain probe (reviewer En8z, claims a-d)."""
import sys, os, json, time, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from _cr_common import load, apriori_layer, auroc, seeded_group_folds, stat
from groundlm.probe.confidence import purge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

T0 = time.time()
OUT = 'runs/cr_passage_leakage.json'
RT = ["qwen25_7b", "llama31_8b", "gemma2_9b"]

rows = [json.loads(l) for l in open('data/ragtruth.jsonl')]
norm = lambda x: " ".join(x.lower().split())
_c = {}
ctx_id = np.array([_c.setdefault(norm(r['context']), len(_c)) for r in rows])
task_of = np.array([r['task_type'] for r in rows])
y_all = np.array([r['faithful'] for r in rows], int)

# ---------- leakage-channel diagnostic: what can passage identity alone buy you? ----------
import collections
sums = collections.Counter(); cnts = collections.Counter()
for c, yy in zip(ctx_id, y_all):
    sums[c] += yy; cnts[c] += 1
loo = np.array([(sums[c] - yy) / (cnts[c] - 1) for c, yy in zip(ctx_id, y_all)])   # leave-one-out passage mean label
naive = np.array([sums[c] / cnts[c] for c in ctx_id])
diag = {"passage_mean_label_LOO_auroc": auroc(loo, y_all),
        "passage_mean_label_naive_auroc": auroc(naive, y_all),
        "n_passages": int(len(_c)), "n_rows": int(len(rows)),
        "passages_with_mixed_labels": int(sum(1 for c in range(len(_c))
                                              if 0 < sums[c] < cnts[c]))}
print("LEAKAGE CHANNEL:", json.dumps(diag), flush=True)

def fit_fold(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
    return te, clf.decision_function(sc.transform(X[te]))

def oof_seeded(X, y, groups, seed, par):
    folds = list(seeded_group_folds(groups, 5, seed))
    res = par(delayed(fit_fold)(X, y, tr, te) for tr, te in folds)
    oof = np.zeros(len(y))
    for te, s in res: oof[te] = s
    return oof

def boot_ci_paired(a, b, y, groups, B=2000, seed=1):
    """cluster bootstrap over `groups`; returns (ci_a, diff_mean, diff_ci, p)."""
    uniq = np.unique(groups); idx_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed); A = []; D = []
    for _ in range(B):
        gs = rng.choice(uniq, len(uniq))
        idx = np.concatenate([idx_of[g] for g in gs])
        if len(np.unique(y[idx])) == 2:
            aa = auroc(a[idx], y[idx]); A.append(aa)
            if b is not None: D.append(aa - auroc(b[idx], y[idx]))
    A = np.array(A)
    out = {"ci": [float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))]}
    if b is not None:
        D = np.array(D)
        out |= {"diff": float(D.mean()),
                "diff_ci": [float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))],
                "p": float((D <= 0).mean())}
    return out

RES = {"claims_abd": diag, "models": {}}
def save(): json.dump(RES, open(OUT, 'w'), indent=2, default=float)

with Parallel(n_jobs=5, backend='loky', max_nbytes='1M') as par:
    for m in RT:
        z, meta = load(f"{m}_rt"); L = apriori_layer(meta)
        X = z[f'last_{L}'].astype(np.float64)
        y = z['faithful'].astype(int); iid = z['item_id']
        g_item = iid                       # paper's grouping (unique per response)
        g_pass = ctx_id[iid]               # correct grouping (normalized source passage)
        ov = z['lexical_overlap'].astype(np.float64)
        nli = z['nli_entail_prob'].astype(np.float64)
        conf = z['mean_maxsoftmax'].astype(np.float64)
        tt = task_of[iid]
        surf = np.stack([ov, z['answer_tok_len'].astype(float)], 1)
        Xp = purge(X, surf)                # overlap+length residualized (paper's "no-overlap" arm)

        r = {"layer": int(L), "n": int(len(y)),
             "baselines_grouping_invariant": {
                 "lexical_overlap_auroc": auroc(ov, y),
                 "nli_auroc": auroc(nli, y),
                 "max_softmax_auroc": auroc(conf, y)}}
        for gname, g in (("item_id", g_item), ("passage", g_pass)):
            for fname, XX in (("raw", X), ("overlap_purged", Xp)):
                aus = []; oofs = {}
                for seed in range(10):
                    oof = oof_seeded(XX, y, g, seed, par)
                    aus.append(auroc(oof, y)); oofs[seed] = oof
                    print(f"  [{time.time()-T0:6.0f}s] {m} {gname:8s} {fname:14s} seed{seed} "
                          f"AUROC={aus[-1]:.4f}", flush=True)
                key = f"{gname}__{fname}"
                r[key] = stat(aus) | {"per_seed": [float(a) for a in aus]}
                if fname == "raw":
                    r[key]["per_task_mean_over_seeds"] = {
                        t: float(np.mean([auroc(oofs[s][tt == t], y[tt == t]) for s in range(10)]))
                        for t in sorted(set(tt.tolist()))}
                    # bootstrap: cluster at the SAME level as the grouping, seed-0 OOF
                    r[key]["boot_seed0_cluster_" + gname] = boot_ci_paired(oofs[0], ov, y, g)
                    r[key]["_oof_seed0"] = oofs[0].tolist()
                save()
        # cross: paper's item-grouped OOF, but honest passage-cluster bootstrap
        oof_item0 = np.array(r["item_id__raw"]["_oof_seed0"])
        r["item_grouped_oof_with_PASSAGE_cluster_boot"] = boot_ci_paired(oof_item0, ov, y, g_pass)
        r["overlap_vs_nli_passage_boot"] = boot_ci_paired(ov, nli, y, g_pass)
        for k in ("item_id__raw", "passage__raw"): r[k].pop("_oof_seed0", None)
        RES["models"][m] = r; save()
        z.drop(f'last_{L}'); del X, Xp
        print(f"=== {m} done [{time.time()-T0:.0f}s] ===", flush=True)
print("ALL DONE", time.time()-T0, flush=True)
