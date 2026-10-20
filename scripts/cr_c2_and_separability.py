"""Camera-ready: put numbers behind the two analyses the submission describes but never reports.

(1) C2 (Sec. 5.2) is currently a null stated in prose with no effect size, no CI, no
    p-value, and no per-family breakdown; Gemma-2-9B has no stored measurement at all.
    This recomputes the confidence-asymmetry
        conf_asym = absorbed(factuality) - absorbed(faithfulness)
    for ALL four families at EVERY cached layer, with the bootstrap CI and the
    one-sided p-value that probe/confidence.py already produces, and summarises
    per family (mean +- sd over layers, sign counts, how many layers are
    significantly positive).

(2) The "Null-calibrated separability" paragraph in Sec. 4 fully specifies a test
    (acute angle between d_S and d_F against a within-notion half-split null) whose
    result appears nowhere in the paper. This computes it per family at the a-priori
    depth-0.5 layer so the paragraph can either be backed by numbers or deleted.
    NOTE the test is a within-notion resampling null, NOT a label-permutation test,
    and the paper's n_null=10,000 does not match any committed caller (analyze_v2.py
    uses 150; the library default is 300). N_NULL below is explicit and reported.

Usage:  PYTHONPATH=src python scripts/cr_c2_and_separability.py
Writes: runs/cr_c2_and_separability.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from groundlm.analyze import build_confounds                       # noqa: E402
from groundlm.probe.confidence import asymmetry_test               # noqa: E402
from groundlm.probe.identification import separability_test        # noqa: E402
from _cr_common import load, apriori_layer                          # noqa: E402

MODELS4 = ["qwen25_7b", "mistral7b_v03", "llama31_8b", "gemma2_9b"]
N_BOOT = int(os.environ.get("N_BOOT", "200"))
N_NULL = int(os.environ.get("N_NULL", "2000"))
OUTP = os.environ.get("OUT", "runs/cr_c2_and_separability.json")


out = {"n_boot": N_BOOT, "n_null": N_NULL, "numpy": np.__version__,
       "c2_per_layer": {}, "c2_summary": {}, "separability": {}}
t0 = time.time()

for m in MODELS4:
    data, meta = load(f"{m}_v2")
    S = data["support"].astype(int)
    F = data["factuality"].astype(int)
    L0 = apriori_layer(meta)
    rows = []
    print(f"=== {m} (layers {meta['layers']}, a-priori L{L0}) ===", flush=True)
    for L in meta["layers"]:
        X = data.layer(L)
        conf = build_confounds(data, X, "confidence")
        a = asymmetry_test(X, S, F, conf, n_boot=N_BOOT)
        rows.append({"layer": int(L),
                     "depth": round(L / meta["n_layers"], 3),
                     "conf_asym": a.asymmetry,
                     "ci": list(a.asymmetry_ci),
                     "p": a.p_value,
                     "absorbed_faith": a.faith["absorbed"],
                     "absorbed_fact": a.fact["absorbed"]})
        data.drop(f"last_{L}")
        print(f"  L{L:>2} (d={L/meta['n_layers']:.2f}) conf_asym={a.asymmetry:+.4f} "
              f"CI[{a.asymmetry_ci[0]:+.3f},{a.asymmetry_ci[1]:+.3f}] p={a.p_value:.3f}", flush=True)
    out["c2_per_layer"][m] = rows

    v = np.array([r["conf_asym"] for r in rows])
    ps = np.array([r["p"] for r in rows])
    out["c2_summary"][m] = {
        "layer_a_priori": int(L0),
        "conf_asym_at_a_priori": next(r["conf_asym"] for r in rows if r["layer"] == L0),
        "p_at_a_priori": next(r["p"] for r in rows if r["layer"] == L0),
        "n_layers": len(rows),
        "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
        "min": float(v.min()), "max": float(v.max()),
        "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum()),
        "n_sig_favouring_factuality": int((ps < 0.05).sum()),   # one-sided P(asym<=0)
    }
    s = out["c2_summary"][m]
    print(f"  -> mean={s['mean']:+.4f}+-{s['sd']:.4f} range[{s['min']:+.4f},{s['max']:+.4f}] "
          f"pos/neg={s['n_positive']}/{s['n_negative']} sig+={s['n_sig_favouring_factuality']}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---- separability at the a-priori layer ----
    X0 = data.layer(L0)
    sep = separability_test(X0, S, F, n_null=N_NULL, n_boot=200)
    out["separability"][m] = {
        "layer": int(L0),
        "cross_angle_deg": float(sep.cross_angle),
        "p_value": float(sep.p_value),
        "n_null": N_NULL,
        "p_resolution_bound": 1.0 / N_NULL,
        "SF_identical": bool(np.array_equal(S, F)),
        "SF_corr": float(np.corrcoef(S, F)[0, 1]),
    }
    r = out["separability"][m]
    print(f"  separability L{L0}: angle={r['cross_angle_deg']:.1f}deg p={r['p_value']:.4g} "
          f"(n_null={N_NULL}, resolution {1.0/N_NULL:.0e}) S==F={r['SF_identical']} "
          f"corr(S,F)={r['SF_corr']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    json.dump(out, open(OUTP, "w"), indent=2, default=float)

print(f"\nwrote {OUTP} [{time.time()-t0:.0f}s]")
