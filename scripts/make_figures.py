"""Publication figures for the GroundLM paper (ACL workshop sizing, vector PDF)."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import cv_auroc, fit_direction          # noqa: E402
from groundlm.probe.confidence import purge                            # noqa: E402
from groundlm.analyze import build_confounds                           # noqa: E402
from groundlm.transfer import transfer_auroc                           # noqa: E402
from _cr_common import source_groups, seeded_group_folds                # noqa: E402
from groundlm.probe.directions import project                           # noqa: E402


def _purge_fold(Xtr, Xte, Ctr, Cte):
    """Residualise with the regression fitted on the training fold only."""
    mu, sd = Ctr.mean(0), Ctr.std(0) + 1e-8
    Atr = np.concatenate([np.ones((len(Ctr), 1)), (Ctr - mu) / sd], 1)
    beta, *_ = np.linalg.lstsq(Atr, Xtr, rcond=None)
    Ate = np.concatenate([np.ones((len(Cte), 1)), (Cte - mu) / sd], 1)
    return Xtr - Atr @ beta, Xte - Ate @ beta


def gcv_auroc(X, y, groups, folds=5, seed=0, C=None):
    """Grouped, signed cross-validated AUROC.

    Replaces cv_auroc, which shuffles rows and np.array_splits them: that is a row-level
    split, so byte-identical statements of one source QA land on both sides of the fold
    boundary, and it contradicted the grouped protocol the paper states.
    """
    y = np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for tr, te in seeded_group_folds(groups, folds, seed):
        Xtr, Xte = (X[tr], X[te]) if C is None else _purge_fold(X[tr], X[te], C[tr], C[te])
        oof[te] = project(Xte, fit_direction(Xtr, y[tr]))
    return auroc_(oof, y)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})
# Okabe-Ito colorblind-safe
BLUE, SKY, ORANGE, GREEN, VERM, GRAY, PURPLE = (
    "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#9A9A9A", "#CC79A7")
FIGS = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
os.makedirs(FIGS, exist_ok=True)
MODELS = [("qwen25_7b", "Qwen2.5-7B"), ("mistral7b_v03", "Mistral-7B"),
          ("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B")]


from sklearn.metrics import roc_auc_score                              # noqa: E402
def auroc_(s, y):
    """Signed AUROC. Polarity is fixed by the training fold, never by the test fold."""
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, s))


def load(d):
    data = np.load(f"runs/{d}/features.npz", allow_pickle=True)
    meta = json.load(open(f"runs/{d}/features.meta.json"))
    return data, meta


def save(fig, name):
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIGS}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def fig_decoupling():
    # panel a: corr(support,overlap); panel b: NLI entailment by cell
    rows = [json.loads(l) for l in open("data/ctrlpairs_v2.jsonl")]
    S = np.array([r["support"] for r in rows]); O = np.array([r["overlap_measured"] for r in rows])
    corr_v2 = abs(np.corrcoef(S, O)[0, 1])
    v1 = json.load(open("runs/v1_corr.json"))   # committed; produced by scripts/v1_corr.py
    corr_v1 = abs(v1["corr_support_overlap_exactstring"])
    corrs = [corr_v1, corr_v2, 0.0]
    clabs = ["extractive\n(NQ-Swap)", "cells\nA/C/D", "$+$cell B\n(projected)"]
    # NLI by cell — exclude donor-collision (q'==q) cell-C items
    byitem = {}
    for r in rows: byitem.setdefault(r["item_id"], {})[r["cell"]] = r
    nrm = lambda x: " ".join(x.lower().split())
    coll = {i: (("A" in d and "C" in d) and nrm(d["A"]["question"]) == nrm(d["C"]["question"])) for i, d in byitem.items()}
    nli = {c: [] for c in "ACD"}
    for r in rows:
        if r.get("nli_entail_prob", -1) < 0: continue
        if r["cell"] == "C" and coll.get(r["item_id"]): continue   # drop contaminated C
        if r["cell"] in nli: nli[r["cell"]].append(r["nli_entail_prob"])
    cellmeans = {c: float(np.mean(v)) for c, v in nli.items()}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.4, 2.0))
    bars = axa.bar(range(3), corrs, color=[VERM, BLUE, SKY], width=0.62)
    bars[2].set_hatch("//"); bars[2].set_edgecolor("white")     # cell B is zero-yield (projected)
    axa.set_xticks(range(3)); axa.set_xticklabels(clabs)
    axa.set_ylabel(r"$|\mathrm{corr}(\mathrm{support},\,\mathrm{overlap})|$")
    axa.set_ylim(0, 1.15); axa.set_title("(a) decoupling the confound", loc="left", fontweight="bold")
    for b, v in zip(bars, corrs):
        axa.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    cells = ["A\nsup,pres", "C\nunsup,pres", "D\nunsup,abs"]
    vals = [cellmeans["A"], cellmeans["C"], cellmeans["D"]]
    cols = [GREEN, VERM, GRAY]
    bars = axb.bar(range(3), vals, color=cols, width=0.62)
    axb.axhline(0.5, ls="--", lw=0.7, color="k")
    axb.text(2.45, 0.52, "decision\nthreshold", fontsize=7.5, ha="right", va="bottom")
    axb.set_xticks(range(3)); axb.set_xticklabels(cells)
    axb.set_ylabel("NLI entailment prob."); axb.set_ylim(0, 1.0)
    axb.set_title("(b) NLI is fooled by overlap", loc="left", fontweight="bold")
    for b, v in zip(bars, vals):
        axb.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axb.annotate("answer present\nbut unsupported", xy=(1.27, vals[1]), xytext=(1.05, 0.82),
                 fontsize=7.5, ha="center", color=VERM,
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=0.6))
    save(fig, "fig_decoupling")


FIG2_DUMP = {}


def fig_c1_layers():
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.3), sharex=False)
    handles_labels = None
    for ax, (d, title) in zip(axes.ravel(), MODELS):
        data, meta = load(f"{d}_v2")
        S = data["support"].astype(int); O = data["overlap_measured"].astype(int)
        m1 = O == 1
        g1 = source_groups(data["item_id"])[m1]
        conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"], data["first_maxsoftmax"]], 1)[m1]
        logp = auroc_(data["mean_logprob"].astype(float)[m1], S[m1])   # layer-independent
        FIG2_DUMP[d] = {"layers": list(map(int, meta["layers"])), "n_layers": int(meta["n_layers"])}
        layers = meta["layers"]; nL = meta["n_layers"]; xs = [L / nL for L in layers]
        raw, pur = [], []
        for L in layers:
            X = data[f"last_{L}"].astype(np.float64)[m1]
            raw.append(gcv_auroc(X, S[m1], g1)); pur.append(gcv_auroc(X, S[m1], g1, C=conf))
        FIG2_DUMP[d].update({"raw": [float(v) for v in raw], "purged": [float(v) for v in pur],
                             "logprob_baseline": float(logp)})
        ax.plot(xs, raw, "-o", color=BLUE, ms=2, lw=1.1, label=r"support$|_{O=1}$ (with context)")
        ax.plot(xs, pur, "--s", color=VERM, ms=2, lw=1.0, label="$+$confidence-purge")
        ax.axhline(logp, ls="-", lw=0.7, color=GRAY, label="log-prob baseline")
        ax.axhline(0.5, ls=":", lw=0.6, color="k")
        # no-context overlay: the SAME all-cells CV-AUROC metric as the raw curve,
        # at the a-priori mid layer (the only layer cached for the ablation run).
        La = int(list(json.load(open(f"runs/{d}_v2_nc/features.meta.json"))["layers"])[0])
        nc = dict(np.load(f"runs/{d}_v2_nc/features.npz", allow_pickle=True))
        raw_nc = gcv_auroc(nc[f"last_{La}"].astype(np.float64)[m1], S[m1], g1)
        xa = La / nL; raw_ctx_a = raw[layers.index(La)]
        ax.plot([xa, xa], [raw_ctx_a, raw_nc], color=GREEN, lw=0.8, zorder=4)
        ax.plot([xa], [raw_nc], marker="*", ms=8, color=GREEN, mec="k", mew=0.4,
                ls="none", zorder=5, label="context removed (mid-layer)")
        FIG2_DUMP[d].update({"apriori_layer": int(La), "raw_no_context": float(raw_nc),
                             "raw_with_context_at_apriori": float(raw_ctx_a)})
        ax.annotate(f"$-{raw_ctx_a - raw_nc:.2f}$", xy=(xa, raw_nc),
                    xytext=(xa + 0.015, raw_nc - 0.006), fontsize=7, color=GREEN, va="top")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(0.45, 0.92); ax.set_xlim(min(xs) - .02, max(xs) + .02)
        ax.set_xlabel("relative layer depth"); ax.set_ylabel("AUROC")
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.21)
    fig.legend(*handles_labels, loc="lower center", ncol=4, fontsize=7.5,
               handlelength=1.5, columnspacing=1.1, bbox_to_anchor=(0.5, -0.004))
    json.dump(FIG2_DUMP, open("runs/fig_c1_layers_values.json", "w"), indent=2, default=float)
    save(fig, "fig_c1_layers")


def _transfer_mat(mode_off):
    runs = []
    for d, name in MODELS:
        data, meta = load(f"{d}_v2")
        L = min(meta["layers"], key=lambda L: abs(L - 0.5 * meta["n_layers"]))
        X = data[f"last_{L}"].astype(np.float64)
        runs.append({"name": name, "X": X, "y": data["support"].astype(int),
                     "conf": build_confounds(data, X, "confidence"),
                     "groups": source_groups(data["item_id"])})
    n = len(runs); M = np.zeros((n, n))
    purged = "purged" in mode_off
    for i, s in enumerate(runs):
        for j, t in enumerate(runs):
            # Every cell, diagonal included, uses THIS panel's estimator. Previously the
            # diagonal silently fell back to `within_target` -- an ordinary within-family
            # probe scored raw -- so panel (b)'s diagonal was not ACS at all and was
            # preprocessed differently from its own off-diagonal.
            M[i, j] = transfer_auroc(s["X"], s["y"], s["conf"], t["X"], t["y"], t["conf"],
                                     mode=mode_off, purged=purged,
                                     groups=s["groups"])["auroc"]
    return M, [r["name"] for r in runs]


def fig_c3_heatmap():
    Mp, names = _transfer_mat("procrustes_raw")
    Ma, _ = _transfer_mat("acs")   # raw, matching panel (a); mixing raw vs purged is not a fair comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.6, 2.0), gridspec_kw={"wspace": 0.18})
    for ax, M, ttl in [(a1, Mp, "(a) Procrustes"), (a2, Ma, "(b) anchor (ACS)")]:
        im = ax.imshow(M, vmin=0.5, vmax=0.92, cmap="viridis", aspect="equal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        short = [n.split("-")[0] for n in names]
        ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_yticklabels(short if ax is a1 else [])
        ax.set_title(ttl, loc="left", fontsize=9, fontweight="bold")
        ax.set_xlabel("target");
        if ax is a1: ax.set_ylabel("source")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.78 else "black")
    cb = fig.colorbar(im, ax=[a1, a2], fraction=0.025, pad=0.02)
    cb.set_label("transfer AUROC (support axis)", fontsize=8)
    json.dump({"models": names, "procrustes_raw": Mp.tolist(), "acs_raw": Ma.tolist()},
              open("runs/fig_c3_heatmap_values.json", "w"), indent=2)
    save(fig, "fig_c3_heatmap")


def fig_c4_ragtruth():
    # all values read from the committed gate JSONs (no hardcoded literals)
    fams = [("qwen25_7b", "Qwen2.5-7B"), ("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B")]
    # Read the SAME source as Table 3 so the bars and the table cannot diverge.
    _T3 = json.load(open("runs/cr_final_tables.json"))["table3"]
    _KEY = {"support_axis_raw": "synth_dS", "conf": "confidence",
            "nli": "nli", "lexical_overlap": "lexical_overlap"}
    def g(m, k):
        if k == "in_domain":
            return _T3[m]["in_domain"]["mean"]
        return _T3[m][_KEY[k]]
    synth = [g(m, "support_axis_raw") for m, _ in fams]
    nli = [g(m, "nli") for m, _ in fams]
    overlap = [g(m, "lexical_overlap") for m, _ in fams]
    indom = [g(m, "in_domain") for m, _ in fams]
    fams = [n for _, n in fams]
    methods = [("synth $d_S$ (label-free)", synth, BLUE, None),
               ("NLI", nli, GRAY, None), ("lexical overlap", overlap, VERM, None),
               ("in-domain (labeled)", indom, GREEN, "//")]
    fig, ax = plt.subplots(figsize=(2.9, 2.25))
    x = np.arange(len(fams)); w = 0.2
    for k, (lab, vals, col, hatch) in enumerate(methods):
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab, color=col, hatch=hatch, edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, ls="--", lw=0.7, color="k")
    ax.text(2.5, 0.506, "chance", fontsize=8, ha="left", va="bottom", color="0.3")
    ax.set_xticks(x); ax.set_xticklabels([f.split("-")[0] for f in fams])
    ax.set_ylabel("hallucination-detection AUROC"); ax.set_ylim(0.5, 0.85); ax.set_xlim(-0.55, 3.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2, fontsize=8.5, handlelength=1.2, columnspacing=1.0)
    save(fig, "fig_c4_ragtruth")


if __name__ == "__main__":
    print("fig 1 decoupling"); fig_decoupling()
    print("fig 2 C1 layers (computing per-layer AUROC...)"); fig_c1_layers()
    print("fig 3 C3 heatmap (computing 4x4 transfer x2...)"); fig_c3_heatmap()
    print("fig 4 C4 ragtruth"); fig_c4_ragtruth()
    print("done ->", FIGS)
