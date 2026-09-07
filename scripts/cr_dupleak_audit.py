"""Exact verification of reviewer En8z / AC claims (a)-(d) about CtrlPairs duplicate leakage.

Normalisation used throughout: norm(x) = " ".join(str(x).lower().split())
i.e. lowercase + collapse all runs of whitespace to a single space, strip ends.
This is the repo's OWN normaliser (scripts/reviewer_analyses.py / _cr_common.collision_map).
The script also reports the counts under three weaker normalisations so the sensitivity
of each number to the normalisation choice is explicit.
"""
from __future__ import annotations
import json, collections, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GroupKFold
from _cr_common import collision_map

rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
NORMS = {"lower+ws-collapse (repo norm)": lambda x: " ".join(str(x).lower().split()),
         "lowercase only": lambda x: str(x).lower(),
         "ws-collapse only": lambda x: " ".join(str(x).split()),
         "raw exact string": lambda x: str(x)}
norm = NORMS["lower+ws-collapse (repo norm)"]
cell = np.array([r["cell"] for r in rows]); item = np.array([r["item_id"] for r in rows])
O = np.array([r["overlap_measured"] for r in rows])
coll = collision_map()
keepC = ~((cell == "C") & np.array([bool(coll.get(int(i), False)) for i in item]))
m1 = (O == 1) & keepC                       # the Table-1 mask: O==1 AND collision-free

out = {}
print(f"rows={len(rows)} cells={dict(collections.Counter(cell.tolist()))} "
      f"items={len(set(item.tolist()))} donor-collision items={sum(coll.values())}")
print(f"Table-1 mask (O==1 & collision-free): {int(m1.sum())} rows "
      f"(A={int(((cell=='A')&m1).sum())}, C={int(((cell=='C')&m1).sum())})\n")

# ---- (a) + (b) -------------------------------------------------------------
A = [r for r in rows if r["cell"] == "A"]
print("(a)/(b) uniqueness of the 1200 cell-A (question, context, answer) triples")
for name, f in NORMS.items():
    c = collections.Counter((f(r["question"]), f(r["context"]), f(r["asserted_answer"])) for r in A)
    dup_rows = sum(v for v in c.values() if v > 1); dup_g = sum(1 for v in c.values() if v > 1)
    print(f"  {name:<32} unique={len(c):4d}  dup_groups={dup_g:4d}  rows_in_dup_groups={dup_rows:4d}"
          f"  sizes={dict(sorted(collections.Counter(c.values()).items()))}")
    if name.startswith("lower+ws"):
        out["a_unique_triples"] = len(c); out["b_rows_in_dup_groups"] = dup_rows
        out["dup_groups"] = dup_g

# ---- (c) -------------------------------------------------------------------
def fold_of(groups, n=5):
    f = np.empty(len(groups), int)
    for k, (_, te) in enumerate(GroupKFold(n).split(np.zeros((len(groups), 1)), None, groups)):
        f[te] = k
    return f

def spanning(fold_full, mask, tag):
    ia = np.where((cell == "A") & mask)[0]
    g = collections.defaultdict(list)
    for k in ia:
        g[(norm(rows[k]["question"]), norm(rows[k]["context"]), norm(rows[k]["asserted_answer"]))].append(k)
    dup = [v for v in g.values() if len(v) > 1]
    span = [v for v in dup if len(set(fold_full[np.array(v)])) > 1]
    print(f"  {tag:<52} multi-fold dup groups={len(span):4d}  A rows involved={sum(map(len, span)):4d}")
    return len(span), sum(map(len, span))

print("\n(c) exact A-prompt duplicate groups that straddle a GroupKFold(5)-on-item_id boundary")
ff = np.full(len(rows), -1); ff[np.where(m1)[0]] = fold_of(item[m1])
out["c_groups_spanning"], out["c_rows_spanning"] = spanning(ff, m1, "folds over the Table-1 subset (paper's actual split)")
spanning(fold_of(item), np.ones(len(rows), bool), "folds over all 3593 rows")
fa = np.full(len(rows), -1); fa[np.where(cell == "A")[0]] = fold_of(item[cell == "A"])
spanning(fa, cell == "A", "folds over the 1200 A rows only")

# ---- (d) -------------------------------------------------------------------
print("\n(d) item_id granularity")
byitem = {}
for r in rows:
    byitem.setdefault(r["item_id"], {})[r["cell"]] = r
g = collections.defaultdict(list)
for i, d in byitem.items():
    a = d["A"]; g[(norm(a["question"]), norm(a["context"]), norm(a["asserted_answer"]))].append(i)
dup = [v for v in g.values() if len(v) > 1]
alldistinctD = sum(1 for v in dup if len({norm(byitem[i]["D"]["context"]) for i in v if "D" in byitem[i]}) == len(v))
print(f"  1200 item_ids -> {len(g)} distinct source QA triples; {len(dup)} triples cover >1 item_id")
print(f"  {alldistinctD}/{len(dup)} of those groups have a DISTINCT cell-D (swapped) context per item_id,")
print(f"  i.e. the items are different NQ-Swap swap records of the SAME original QA.")
print(f"  example: item_ids {sorted(dup[0])[:6]} share one cell-A row verbatim.")
print(f"  distinct cell-A questions: {len({norm(byitem[i]['A']['question']) for i in byitem})}")
print(f"  distinct cell-A contexts : {len({norm(byitem[i]['A']['context']) for i in byitem})}")
out["d_source_qa_groups"] = len(g); out["d_distinct_questions"] = len({norm(byitem[i]['A']['question']) for i in byitem})
json.dump(out, open("runs/cr_dupleak_audit.json", "w"), indent=2)
print("\nwrote runs/cr_dupleak_audit.json")
