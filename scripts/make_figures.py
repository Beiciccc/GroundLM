"""Publication figures for the GroundLM paper (ACL workshop sizing, vector PDF)."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.probe.directions import cv_auroc, fit_direction          # noqa: E402
from groundlm.probe.confidence import purge                            # noqa: E402
from groundlm.analyze import build_confounds                           # noqa: E402
from groundlm.transfer import transfer_auroc                           # noqa: E402

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
    y = np.asarray(y).astype(int); a = roc_auc_score(y, s); return float(max(a, 1 - a))


def load(d):
    data = dict(np.load(f"runs/{d}/features.npz", allow_pickle=True))
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
    clabs = ["v1\n(NQ-Swap)", "v2 cells\nA/C/D", "$+$cell B\n(projected)"]
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
    axb.annotate("answer present\nbut unsupported", xy=(1.24, 0.18), xytext=(1.05, 0.82),
                 fontsize=7.5, ha="center", color=VERM,
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=0.6))
    save(fig, "fig_decoupling")


def fig_c1_layers():
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.3), sharex=False)
    handles_labels = None
    for ax, (d, title) in zip(axes.ravel(), MODELS):
        data, meta = load(f"{d}_v2")
        S = data["support"].astype(int); O = data["overlap_measured"].astype(int)
        m1 = O == 1
        conf = np.stack([data["mean_logprob"], data["mean_maxsoftmax"], data["first_maxsoftmax"]], 1)[m1]
        logp = auroc_(data["mean_logprob"].astype(float)[m1], S[m1])   # layer-independent
        layers = meta["layers"]; nL = meta["n_layers"]; xs = [L / nL for L in layers]
        raw, pur = [], []
        for L in layers:
            X = data[f"last_{L}"].astype(np.float64)[m1]
            raw.append(cv_auroc(X, S[m1])); pur.append(cv_auroc(purge(X, conf), S[m1]))
        ax.plot(xs, raw, "-o", color=BLUE, ms=2, lw=1.1, label=r"support$|_{O=1}$ (with context)")
        ax.plot(xs, pur, "--s", color=VERM, ms=2, lw=1.0, label="$+$confidence-purge")
        ax.axhline(logp, ls="-", lw=0.7, color=GRAY, label="log-prob baseline")
        ax.axhline(0.5, ls=":", lw=0.6, color="k")
        # no-context overlay: the SAME all-cells CV-AUROC metric as the raw curve,
        # at the a-priori mid layer (the only layer cached for the ablation run).
        La = int(list(json.load(open(f"runs/{d}_v2_nc/features.meta.json"))["layers"])[0])
        nc = dict(np.load(f"runs/{d}_v2_nc/features.npz", allow_pickle=True))
        raw_nc = cv_auroc(nc[f"last_{La}"].astype(np.float64)[m1], S[m1])
        xa = La / nL; raw_ctx_a = raw[layers.index(La)]
        ax.plot([xa, xa], [raw_ctx_a, raw_nc], color=GREEN, lw=0.8, zorder=4)
        ax.plot([xa], [raw_nc], marker="*", ms=8, color=GREEN, mec="k", mew=0.4,
                ls="none", zorder=5, label="context removed (mid-layer)")
        ax.annotate(f"$-{raw_ctx_a - raw_nc:.2f}$", xy=(xa, raw_nc),
                    xytext=(xa + 0.015, raw_nc - 0.006), fontsize=7, color=GREEN, va="top")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(0.45, 0.92); ax.set_xlim(min(xs) - .02, max(xs) + .02)
        ax.set_xlabel("relative layer depth"); ax.set_ylabel("AUROC")
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(bottom=0.14)
    fig.legend(*handles_labels, loc="lower center", ncol=4, fontsize=7.5,
               handlelength=1.5, columnspacing=1.1, bbox_to_anchor=(0.5, 0.01))
    save(fig, "fig_c1_layers")


def _transfer_mat(mode_off):
    runs = []
    for d, name in MODELS:
        data, meta = load(f"{d}_v2")
        L = min(meta["layers"], key=lambda L: abs(L - 0.5 * meta["n_layers"]))
        X = data[f"last_{L}"].astype(np.float64)
        runs.append({"name": name, "X": X, "y": data["support"].astype(int),
                     "conf": build_confounds(data, X, "confidence")})
    n = len(runs); M = np.zeros((n, n))
    for i, s in enumerate(runs):
        for j, t in enumerate(runs):
            mode = "within_target" if i == j else mode_off
            M[i, j] = transfer_auroc(s["X"], s["y"], s["conf"], t["X"], t["y"], t["conf"], mode=mode)["auroc"]
    return M, [r["name"] for r in runs]


def fig_c3_heatmap():
    Mp, names = _transfer_mat("procrustes_raw")
    Ma, _ = _transfer_mat("acs_purged")
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
    save(fig, "fig_c3_heatmap")


def fig_c4_ragtruth():
    # all values read from the committed gate JSONs (no hardcoded literals)
    fams = [("qwen25_7b", "Qwen2.5-7B"), ("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B")]
    def g(m, k): return json.load(open(f"runs/{m}_rt/report_gate_ragtruth.json"))["metrics"][k]["auroc"]
    synth = [g(m, "support_axis_raw") for m, _ in fams]
    nli = [g(m, "nli") for m, _ in fams]
    overlap = [g(m, "lexical_overlap") for m, _ in fams]
    indom = [g(m, "in_domain") for m, _ in fams]
    fams = [n for _, n in fams]
    methods = [("synth $d_S$ (ours, label-free)", synth, BLUE, None),
               ("NLI", nli, GRAY, None), ("lexical overlap", overlap, VERM, None),
               ("in-domain probe (uses labels)", indom, GREEN, "//")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(fams)); w = 0.2
    for k, (lab, vals, col, hatch) in enumerate(methods):
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab, color=col, hatch=hatch, edgecolor="white", linewidth=0.3)
    ax.axhline(0.5, ls="--", lw=0.7, color="k"); ax.text(2.4, 0.51, "chance", fontsize=7.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f.split("-")[0] for f in fams])
    ax.set_ylabel("hallucination-detection AUROC"); ax.set_ylim(0.5, 0.83)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=2, fontsize=7, handlelength=1.3)
    save(fig, "fig_c4_ragtruth")


if __name__ == "__main__":
    print("fig 1 decoupling"); fig_decoupling()
    print("fig 2 C1 layers (computing per-layer AUROC...)"); fig_c1_layers()
    print("fig 3 C3 heatmap (computing 4x4 transfer x2...)"); fig_c3_heatmap()
    print("fig 4 C4 ragtruth"); fig_c4_ragtruth()
    print("done ->", FIGS)
