"""Shared helpers for the camera-ready (cr_*) analyses.

Two things the original analysis scripts do that we deliberately change here:

1. MEMORY. The repo convention is `dict(np.load(path))`, which materialises every
   cached layer at once (~400 MB per model for 15 layers of float16). With four
   families that swaps on a 12 GB box. `LazyNpz` keeps the NpzFile handle and
   decompresses one member on demand, so only the layer under analysis is resident.

2. FOLD DETERMINISM. sklearn's GroupKFold assigns folds via np.argsort over group
   sizes. When group sizes are tied -- 1193/1200 CtrlPairs items have exactly 3 rows,
   and every RAGTruth item is a singleton -- the assignment is decided entirely by
   argsort tie-breaking, which differs between numpy versions (numpy <=2.2 and >=2.3
   give different folds, and hence different AUROCs: e.g. the Llama in-domain probe
   moves 0.8152 <-> 0.8242). `seeded_group_folds` shuffles the unique groups under an
   explicit seed instead, so results are reproducible across numpy versions and can be
   reported as mean +- sd over seeds.
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import roc_auc_score


class LazyNpz:
    """dict-like view over an .npz that decompresses members on first access."""

    def __init__(self, path):
        self._z = np.load(path, allow_pickle=True)
        self._cache = {}

    def __getitem__(self, k):
        if k not in self._cache:
            self._cache[k] = self._z[k]
        return self._cache[k]

    def __contains__(self, k):
        return k in self._z.files

    def get(self, k, default=None):
        return self[k] if k in self else default

    @property
    def files(self):
        return self._z.files

    def drop(self, k):
        """Free one cached member (call after finishing a layer)."""
        self._cache.pop(k, None)

    def layer(self, L, pooling="last", dtype=np.float64):
        return self[f"{pooling}_{L}"].astype(dtype)


def load(tag, root="runs"):
    """Return (LazyNpz, meta) for runs/<tag>/."""
    return (LazyNpz(f"{root}/{tag}/features.npz"),
            json.load(open(f"{root}/{tag}/features.meta.json")))


def apriori_layer(meta):
    """The paper's a-priori analysis layer: nearest cached layer to relative depth 0.5."""
    return min(meta["layers"], key=lambda x: abs(x - 0.5 * meta["n_layers"]))


def band_layers(meta, lo=0.4, hi=0.6):
    """Cached layers inside the paper's mid-layer band (0.4-0.6 relative depth)."""
    return [L for L in meta["layers"] if lo <= L / meta["n_layers"] <= hi]


def auroc(s, y):
    """Polarity-corrected AUROC -- the max(AUC, 1-AUC) convention used in the paper."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, s)
    return float(max(a, 1.0 - a))


def auroc_signed(s, y):
    """Plain AUROC, no polarity flip."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def seeded_group_folds(groups, n_splits=5, seed=0):
    """Deterministic grouped folds: seeded-shuffle the unique groups, deal round-robin."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = {uniq[perm[i]]: i % n_splits for i in range(len(uniq))}
    fold = np.array([fold_of_group[g] for g in groups])
    for k in range(n_splits):
        yield np.where(fold != k)[0], np.where(fold == k)[0]


def collision_map(path="data/ctrlpairs_v2.jsonl"):
    """item_id -> True when the cell-C donor question is string-identical to cell A's.

    Reproduces scripts/reviewer_analyses.py's donor-collision filter (278/1200 items).
    """
    byitem = {}
    for line in open(path):
        r = json.loads(line)
        byitem.setdefault(r["item_id"], {})[r["cell"]] = r
    norm = lambda x: " ".join(x.lower().split())
    return {i: (("A" in d and "C" in d) and norm(d["A"]["question"]) == norm(d["C"]["question"]))
            for i, d in byitem.items()}


def stat(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def source_qa_key(path="data/ctrlpairs_v2.jsonl"):
    """item_id -> normalised (question, context, answer) key of the underlying source QA.

    item_id is only the index of an NQ-Swap swap record, and several swap records share
    one original QA: the 1,200 cell-A rows cover just 555 distinct triples. Grouping by
    item_id therefore leaves byte-identical prompts on both sides of a fold boundary.
    Group by this key instead.
    """
    key = {}
    for line in open(path):
        r = json.loads(line)
        if r["cell"] == "A":
            n = lambda x: " ".join(str(x).lower().split())
            key[r["item_id"]] = f'{n(r["question"])}||{n(r["context"])}||{n(r["asserted_answer"])}'
    return key


def source_groups(item_ids, path="data/ctrlpairs_v2.jsonl"):
    """Vector of source-QA group labels aligned to `item_ids`."""
    key = source_qa_key(path)
    return np.array([key.get(int(i), f"__unmapped_{int(i)}") for i in np.asarray(item_ids)])


def passage_groups(path="data/ragtruth.jsonl"):
    """Normalised source passage per RAGTruth row.

    The released 2,000 rows cover only 450 distinct contexts (4-5 responses each), while
    item_id is a per-row counter, so grouping by item_id is row-level CV.
    """
    n = lambda x: " ".join(str(x).lower().split())
    return np.array([n(json.loads(l)["context"]) for l in open(path)])
