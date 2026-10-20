"""Camera-ready protocol audit, part 3: Figure 3 (fig_c3_heatmap) + the ACS diagonal.

Stage 1 (cheap, no SVD): panel (b) ACS matrix as published, plus the honest
  within-family ACS diagonal that panel (b)'s caption actually claims.
Stage 2 (SVD): panel (a) Procrustes matrix as published.
Stage 3: both panels under an item-grouped calibration/evaluation split.
Every cell is reported as signed AUROC and as the published max(AUC,1-AUC) value.

Writes runs/cr_protocol_audit_fig3.json incrementally.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from _cr_common import load, apriori_layer, auroc_maxflip as auroc, auroc_signed       # noqa: E402
from groundlm.analyze import build_confounds                          # noqa: E402
from groundlm.probe.directions import fit_direction                   # noqa: E402
from groundlm.probe.confidence import purge                           # noqa: E402
from groundlm.transfer.procrustes import (fit_map, transport_direction,  # noqa: E402
                                          acs_features)

MODELS = [("qwen25_7b", "Qwen2.5-7B"), ("mistral7b_v03", "Mistral-7B"),
          ("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B")]
OUT = "runs/cr_protocol_audit_fig3.json"
t0 = time.time()

runs = []
for d, name in MODELS:
    z, meta = load(f"{d}_v2")
    L = apriori_layer(meta)
    X = z.layer(L)
    runs.append({"name": name, "key": d, "layer": int(L), "X": X,
                 "Xp": purge(X, build_confounds(z, X, "confidence")),
                 "y": z["support"].astype(int), "item": z["item_id"]})
    del z
n = len(runs)
out = {"names": [r["name"] for r in runs], "layers": [r["layer"] for r in runs]}


def save():
    json.dump(out, open(OUT, "w"), indent=2, default=float)


def row_split(nrows, seed=0):
    """Exactly transfer_auroc's split: rng.permutation, first 50% = calibration."""
    idx = np.random.default_rng(seed).permutation(nrows)
    nc = int(0.5 * nrows)
    return idx[:nc], idx[nc:]


def grp_split(item, seed=0):
    u = np.unique(item); r = np.random.default_rng(seed); r.shuffle(u)
    cal = set(u[: len(u) // 2].tolist())
    mask = np.array([i in cal for i in item])
    return np.where(mask)[0], np.where(~mask)[0]


def acs_cell(s, t, cal, ev, purged):
    Xs = s["Xp"] if purged else s["X"]
    Xt = t["Xp"] if purged else t["X"]
    a = cal[: min(256, len(cal))]
    Zs = acs_features(Xs, Xs[a]); Zt = acs_features(Xt, Xt[a])
    d = fit_direction(Zs[cal], s["y"][cal])
    sc = Zt[ev] @ d
    return auroc_signed(sc, t["y"][ev]), auroc(sc, t["y"][ev])


def within_cell(t, cal, ev, purged):
    Xt = t["Xp"] if purged else t["X"]
    d = fit_direction(Xt[cal], t["y"][cal])
    sc = Xt[ev] @ d
    return auroc_signed(sc, t["y"][ev]), auroc(sc, t["y"][ev])


def proc_cell(s, t, cal, ev, purged):
    Xs = s["Xp"] if purged else s["X"]
    Xt = t["Xp"] if purged else t["X"]
    d_src = fit_direction(Xs[cal], s["y"][cal])
    A = fit_map(Xt[cal], Xs[cal], orthogonal=True)
    d_tgt = transport_direction(A, d_src)
    sc = Xt[ev] @ d_tgt
    return auroc_signed(sc, t["y"][ev]), auroc(sc, t["y"][ev])


# ---------- Stage 1: panel (b) as published + honest ACS diagonal ----------
Mb_f = np.zeros((n, n)); Mb_s = np.zeros((n, n))
honest_b_f = np.zeros((n, n)); honest_b_s = np.zeros((n, n))
for i, s in enumerate(runs):
    cal, ev = row_split(len(s["y"]), 0)
    for j, t in enumerate(runs):
        if i == j:      # what make_figures actually plots: within_target, UNPURGED
            Mb_s[i, j], Mb_f[i, j] = within_cell(t, cal, ev, purged=False)
        else:
            Mb_s[i, j], Mb_f[i, j] = acs_cell(s, t, cal, ev, purged=True)
        # honest panel (b): ACS everywhere, including the diagonal
        honest_b_s[i, j], honest_b_f[i, j] = acs_cell(s, t, cal, ev, purged=True)
out["published_b_acs_flipped"] = Mb_f.tolist()
out["published_b_acs_signed"] = Mb_s.tolist()
out["honest_b_acs_everywhere_flipped"] = honest_b_f.tolist()
out["honest_b_acs_everywhere_signed"] = honest_b_s.tolist()
print(f"[{time.time()-t0:.0f}s] published panel b (diag=within_target):\n{np.round(Mb_f,3)}", flush=True)
print(f"honest panel b (ACS on the diagonal too):\n{np.round(honest_b_f,3)}", flush=True)
print(f"published diag {np.round(np.diag(Mb_f),3)} -> honest ACS diag {np.round(np.diag(honest_b_f),3)}", flush=True)
save()

# ---------- Stage 2: panel (a) as published ----------
Ma_f = np.zeros((n, n)); Ma_s = np.zeros((n, n))
for i, s in enumerate(runs):
    cal, ev = row_split(len(s["y"]), 0)
    for j, t in enumerate(runs):
        if i == j:
            Ma_s[i, j], Ma_f[i, j] = within_cell(t, cal, ev, purged=False)
        else:
            Ma_s[i, j], Ma_f[i, j] = proc_cell(s, t, cal, ev, purged=False)
        print(f"  [{time.time()-t0:.0f}s] a[{i},{j}]={Ma_f[i,j]:.3f} (signed {Ma_s[i,j]:.3f})", flush=True)
out["published_a_procrustes_flipped"] = Ma_f.tolist()
out["published_a_procrustes_signed"] = Ma_s.tolist()
print(f"published panel a:\n{np.round(Ma_f,3)}", flush=True)
save()

# ---------- Stage 3: item-grouped calibration/evaluation split ----------
Ga_f = np.zeros((n, n)); Gb_f = np.zeros((n, n))
Ga_s = np.zeros((n, n)); Gb_s = np.zeros((n, n))
for i, s in enumerate(runs):
    cal, ev = grp_split(s["item"], 0)
    for j, t in enumerate(runs):
        if i == j:
            Ga_s[i, j], Ga_f[i, j] = within_cell(t, cal, ev, purged=False)
            Gb_s[i, j], Gb_f[i, j] = within_cell(t, cal, ev, purged=False)
        else:
            Ga_s[i, j], Ga_f[i, j] = proc_cell(s, t, cal, ev, purged=False)
            Gb_s[i, j], Gb_f[i, j] = acs_cell(s, t, cal, ev, purged=True)
        print(f"  [{time.time()-t0:.0f}s] grouped a[{i},{j}]={Ga_f[i,j]:.3f} b[{i},{j}]={Gb_f[i,j]:.3f}", flush=True)
out["grouped_a_procrustes_flipped"] = Ga_f.tolist()
out["grouped_a_procrustes_signed"] = Ga_s.tolist()
out["grouped_b_acs_flipped"] = Gb_f.tolist()
out["grouped_b_acs_signed"] = Gb_s.tolist()
print(f"item-grouped panel a:\n{np.round(Ga_f,3)}\nitem-grouped panel b:\n{np.round(Gb_f,3)}", flush=True)
save()

# ---------- Stage 4: ACS with the SAME (raw) preprocessing as panel (a) ----------
# The published comparison is acs_PURGED (0.67) vs procrustes_RAW (0.906): different
# preprocessing. This is the like-for-like ACS number.
Ar_f = np.zeros((n, n))
for i, s in enumerate(runs):
    cal, ev = grp_split(s["item"], 0)
    for j, t in enumerate(runs):
        _, Ar_f[i, j] = acs_cell(s, t, cal, ev, purged=False) if i != j else within_cell(t, cal, ev, False)
out["grouped_acs_raw_flipped"] = Ar_f.tolist()
od = lambda M: float(np.mean([M[i][j] for i in range(n) for j in range(n) if i != j]))
out["offdiag_means"] = {
    "published_a_procrustes_raw": od(out["published_a_procrustes_flipped"]),
    "published_b_acs_purged": od(out["published_b_acs_flipped"]),
    "grouped_a_procrustes_raw": od(out["grouped_a_procrustes_flipped"]),
    "grouped_b_acs_purged": od(out["grouped_b_acs_flipped"]),
    "grouped_acs_raw": od(Ar_f.tolist()),
}
print("off-diagonal means:", json.dumps(out["offdiag_means"], indent=1), flush=True)
save()
print("wrote", OUT)
