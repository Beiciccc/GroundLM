"""C4 — two-axis conformal grounding gate on v2.

Decision score = signed projection onto the confidence-PURGED support axis (and a
two-axis combination with the factuality axis). "Safe to answer" = grounded
(support=1). We report risk-coverage / AURC vs baselines (raw confidence, the open
NLI entailment score) and the conformal coverage at a target risk.

HEADLINE: the overlap=1 subset (cells A supported vs C unsupported-but-present),
where the answer string is present in BOTH, so lexical cues are useless and NLI is
known to be fooled ~40%. If the internal support gate beats NLI there, that is the
C4 win: a one-pass internal signal that detects ungrounded-yet-present answers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.analyze import build_confounds                                   # noqa: E402
from groundlm.probe.directions import fit_direction, project                  # noqa: E402
from groundlm.probe.confidence import purge                                    # noqa: E402
from groundlm.gate import evaluate_gate, aurc, coverage_at_risk, split_conformal_threshold  # noqa: E402


def pick_layer(meta, depth):
    return min(meta["layers"], key=lambda L: abs(L - depth * meta["n_layers"]))


def gate_scores(data, layer, pooling):
    X = data[f"{pooling}_{layer}"].astype(np.float64)
    conf = build_confounds(data, X, "confidence")
    Xp = purge(X, conf)                                  # confidence-purged features
    d_sup = fit_direction(Xp, data["support"].astype(int))
    d_fac = fit_direction(X, data["factuality"].astype(int))
    s_support = project(Xp, d_sup)
    s_fact = project(X, d_fac)
    # orient so higher = safer (more grounded)
    if np.corrcoef(s_support, data["support"])[0, 1] < 0:
        s_support = -s_support
    return s_support, s_fact


def evaluate(tag, s_support, s_fact, correct, baselines, target_risk):
    res = evaluate_gate(s_support, s_fact, correct, baselines=baselines)
    cov = coverage_at_risk(s_support, correct, target_risk)
    print(f"\n  [{tag}] n={len(correct)} grounded_rate={correct.mean():.2f}  (AURC, lower=better)")
    for k in sorted(res):
        print(f"     {k:<24} {res[k]:.3f}")
    print(f"     coverage@risk<= {target_risk:.2f} (support gate): {cov:.2f}")
    return {**res, "coverage_at_risk": cov, "n": int(len(correct))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--target-risk", type=float, default=0.1)
    args = ap.parse_args()

    data = dict(np.load(os.path.join(args.dir, "features.npz"), allow_pickle=True))
    meta = json.load(open(os.path.join(args.dir, "features.meta.json")))
    L = pick_layer(meta, args.depth)
    print(f"model={meta['model_id']} layer L{L}/{meta['n_layers']}")

    s_support, s_fact = gate_scores(data, L, args.pooling)
    correct = data["support"].astype(int)                 # grounded = safe to answer
    O = data["overlap_measured"].astype(int)

    baselines = {"max_softmax": data["mean_maxsoftmax"].astype(np.float64)}
    nli = data.get("nli_entail_prob")
    if nli is not None and np.nanmax(nli) > 0:            # stored at build time
        baselines["nli"] = nli.astype(np.float64)

    report = {"layer": int(L), "model": meta["model_id"]}
    report["all"] = evaluate("ALL items", s_support, s_fact, correct, baselines, args.target_risk)

    m = O == 1                                            # HARD subset: answer present in both
    if m.sum() > 30 and len(np.unique(correct[m])) == 2:
        bl = {k: v[m] for k, v in baselines.items()}
        hard = report["overlap1_hard"] = evaluate(
            "OVERLAP=1 hard subset (A vs C)",
            s_support[m], s_fact[m], correct[m], bl, args.target_risk)
        if "aurc_nli" in hard:
            win = hard["aurc_faith_axis"] < hard["aurc_nli"]
            print(f"\n  C4 headline: support gate {'BEATS' if win else 'does NOT beat'} NLI on the "
                  f"overlap=1 hard subset (AURC {hard['aurc_faith_axis']:.3f} vs NLI {hard['aurc_nli']:.3f}).")
        else:
            print("\n  C4 headline: NLI baseline not stored (build with NLI, i.e. without --skip-nli) "
                  "to enable the probe-vs-NLI comparison on the hard subset.")

    json.dump(report, open(os.path.join(args.dir, "report_gate_v2.json"), "w"), indent=2, default=float)
    print(f"\n  wrote {os.path.join(args.dir, 'report_gate_v2.json')}")


if __name__ == "__main__":
    main()
