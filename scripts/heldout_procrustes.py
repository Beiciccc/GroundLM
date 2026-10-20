"""Point-5 control: is the C3 transported-to-native cosine an in-sample fitting
artifact? The reported 0.976 fits the Procrustes map AND both directions on the same
(calibration) items. Here we hold out items: the map is fit on the calibration split,
the TARGET native direction is re-estimated on a disjoint eval split, and we recompute
the cosine. If alignment is real (not leakage) the held-out cosine should stay near the
in-sample value and near the native self-consistency ceiling (cos of the target
direction estimated on the two disjoint splits), and far above the nulls.

Mirrors the paper pipeline exactly (same fit_map / transport_direction / fit_direction,
raw mid-layer activations, all cells, item-grouped split), so cos_insample reproduces
the reported 0.976 as a sanity check. Averages over SEEDS x 12 ordered pairs.
Writes runs/heldout_procrustes.json.
"""
from __future__ import annotations
import json, os, sys, itertools
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _cr_common import source_groups   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.transfer.procrustes import fit_map, transport_direction   # noqa: E402
from groundlm.probe.directions import fit_direction, _unit              # noqa: E402

MODELS = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
SEEDS = list(range(10))  # 12 ordered pairs x 10 seeds = 120 measurements/quantity
AXES = {"support": "S", "factuality": "F"}   # support = the claim; factuality = positive control


def load(m):
    npz = dict(np.load(f"runs/{m}_v2/features.npz", allow_pickle=True))
    meta = json.load(open(f"runs/{m}_v2/features.meta.json"))
    L = min(meta["layers"], key=lambda x: abs(x - 0.5 * meta["n_layers"]))
    return {"X": npz[f"last_{L}"].astype(np.float64), "S": npz["support"].astype(int),
            "F": npz["factuality"].astype(int), "item": source_groups(npz["item_id"])}


def split(item, seed):
    u = np.unique(item); r = np.random.default_rng(seed); r.shuffle(u)
    cal = set(u[: len(u) // 2].tolist())
    mask = np.array([i in cal for i in item])
    return mask, ~mask


def cos(a, b):
    return abs(float(np.dot(_unit(a), _unit(b))))


RUN = {m: load(m) for m in MODELS}
acc = {ax: {k: [] for k in
            ["insample", "heldout_tgt", "heldout_both", "native_selfcons", "null_rand", "null_shuf"]}
       for ax in AXES}
per_pair = {ax: {} for ax in AXES}

for s, t in itertools.permutations(MODELS, 2):
    Sd, Td = RUN[s], RUN[t]
    pair_vals = {ax: [] for ax in AXES}
    for seed in SEEDS:
        cal, ev = split(Sd["item"], seed)           # item-grouped, shared row order across models
        A = fit_map(Td["X"][cal], Sd["X"][cal], orthogonal=True)   # map: fit on CAL only
        rng = np.random.default_rng(1000 + seed)
        for ax, key in AXES.items():
            ds_cal = fit_direction(Sd["X"][cal], Sd[key][cal])
            ds_ev = fit_direction(Sd["X"][ev], Sd[key][ev])
            dt_cal = fit_direction(Td["X"][cal], Td[key][cal])
            dt_ev = fit_direction(Td["X"][ev], Td[key][ev])         # held-out target native
            tr_cal = transport_direction(A, ds_cal)                 # map+src from CAL
            tr_ev = transport_direction(A, ds_ev)                   # map CAL, src EV
            # nulls transported through the SAME cal-fit map, scored vs held-out native
            d_rand = _unit(rng.standard_normal(Sd["X"].shape[1]))
            y_shuf = Sd[key][cal].copy(); rng.shuffle(y_shuf)
            d_shuf = fit_direction(Sd["X"][cal], y_shuf)
            acc[ax]["insample"].append(cos(tr_cal, dt_cal))         # reproduces reported 0.976
            acc[ax]["heldout_tgt"].append(cos(tr_cal, dt_ev))       # PRIMARY: tgt native held out
            acc[ax]["heldout_both"].append(cos(tr_ev, dt_ev))       # STRONGER: both dirs held out from map
            acc[ax]["native_selfcons"].append(cos(dt_cal, dt_ev))   # ceiling: finite-sample noise
            acc[ax]["null_rand"].append(cos(transport_direction(A, d_rand), dt_ev))
            acc[ax]["null_shuf"].append(cos(transport_direction(A, d_shuf), dt_ev))
            pair_vals[ax].append(cos(tr_cal, dt_ev))
    for ax in AXES:
        per_pair[ax][f"{s}->{t}"] = float(np.mean(pair_vals[ax]))
    print(f"  {s}->{t}: support held-out cos = {np.mean(pair_vals['support']):.3f}", flush=True)


def summ(v):
    v = np.array(v)
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


out = {"seeds": SEEDS, "n_ordered_pairs": 12, "n_measurements": len(SEEDS) * 12,
       "axes": {ax: {k: summ(acc[ax][k]) for k in acc[ax]} for ax in AXES},
       "per_pair_support_heldout": per_pair["support"]}
os.makedirs("runs", exist_ok=True)
json.dump(out, open("runs/heldout_procrustes.json", "w"), indent=1)

print("\n==== held-out Procrustes cosine (mean +/- std over 12 pairs x %d seeds) ====" % len(SEEDS))
for ax in AXES:
    a = out["axes"][ax]
    print(f"[{ax}]")
    for k in ["insample", "heldout_tgt", "heldout_both", "native_selfcons", "null_rand", "null_shuf"]:
        s = a[k]
        print(f"   {k:16} {s['mean']:.3f} +/- {s['std']:.3f}   [min {s['min']:.3f}, max {s['max']:.3f}]")
print("\nwrote runs/heldout_procrustes.json")
