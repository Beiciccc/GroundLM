"""Validate Stage-5 transfer and Stage-6 gate math on synthetic data.

Stage 5: two "models" with DIFFERENT hidden dims share a latent grounding signal
but carry family-specific confidence components. Checks:
  * within-target AUROC is high (probe works),
  * random-direction transfer ~ 0.5 (lower-bound null),
  * Procrustes transport of the purged faithfulness probe transfers well above
    random (the C3 mechanism), and ACS baseline also transfers.

Stage 6: checks the split-conformal selective-risk guarantee holds out-of-sample
and that the faithfulness axis is a useful selective score (AURC < random).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.transfer import transfer_auroc                       # noqa: E402
from groundlm.gate import (aurc, split_conformal_threshold)        # noqa: E402


def make_two_models(n=1400, dA=96, dB=64, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice([-1.0, 1.0], n)      # grounding (faithfulness) sign
    f = rng.choice([-1.0, 1.0], n)      # factuality sign
    lat = np.stack([g, f], axis=1)      # shared latent

    def model(dim, conf_dir_seed):
        r = np.random.default_rng(conf_dir_seed)
        W = r.standard_normal((2, dim))                       # latent -> hidden
        conf = np.clip(0.5 + 0.45 * f + 0.08 * r.standard_normal(n), 0, 1)  # tracks factuality
        e = r.standard_normal(dim); e /= np.linalg.norm(e)    # family-specific conf direction
        X = lat @ W + 1.4 * r.standard_normal((n, dim)) + 2.0 * conf[:, None] * e[None, :]
        confounds = np.stack([conf, np.linalg.norm(X, axis=1), r.random(n)], axis=1)
        return X, confounds

    XA, cA = model(dA, 1)
    XB, cB = model(dB, 2)
    return (XA, cA), (XB, cB), (g > 0).astype(int), (f > 0).astype(int)


def stage5():
    (XA, cA), (XB, cB), gy, fy = make_two_models()
    print("=== Stage 5: cross-family transfer (dims 96 -> 64) ===")
    checks = {}
    for mode in ["within_target", "random", "procrustes_purged", "procrustes_raw", "acs_purged"]:
        r = transfer_auroc(XA, gy, cA, XB, gy, cB, mode=mode, seed=1)
        checks[mode] = r["auroc"]
        print(f"  grounding/faithfulness transfer [{mode:>16}] AUROC = {r['auroc']:.3f}")
    ok = (checks["within_target"] > 0.75 and checks["random"] < 0.62
          and checks["procrustes_purged"] > 0.65 and checks["acs_purged"] > 0.58)
    print(f"  CHECK transfer math: {'PASS' if ok else 'FAIL'}")
    return ok


def stage6(alpha=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n = 4000
    s_true = rng.standard_normal(n)
    correct = (rng.random(n) < 1 / (1 + np.exp(-2.0 * s_true))).astype(int)
    s_faith = s_true + 0.7 * rng.standard_normal(n)     # noisy observation
    s_fact = rng.standard_normal(n)

    idx = rng.permutation(n); cal, te = idx[:2000], idx[2000:]
    tau = split_conformal_threshold(s_faith[cal], correct[cal], alpha, higher_is_safer=True)
    acc = s_faith[te] >= tau
    test_risk = float((1 - correct[te][acc]).mean()) if acc.any() else 0.0
    cov = float(acc.mean())

    a_faith = aurc(s_faith, correct)
    a_rand = aurc(rng.standard_normal(n), correct)
    print("\n=== Stage 6: conformal gate ===")
    print(f"  target risk alpha={alpha}; out-of-sample answered risk={test_risk:.3f} "
          f"at coverage={cov:.2f}")
    print(f"  AURC faith-axis={a_faith:.3f} vs random={a_rand:.3f}")
    guarantee = (test_risk <= alpha + 0.05) and (cov > 0.10)   # controls risk AND answers a meaningful fraction
    useful = a_faith < a_rand - 0.05
    print(f"  CHECK conformal risk guarantee (+nonzero coverage): {'PASS' if guarantee else 'FAIL'}")
    print(f"  CHECK faith-axis is a useful selective score: {'PASS' if useful else 'FAIL'}")
    return guarantee and useful


def main():
    ok5 = stage5()
    ok6 = stage6()
    print("\n================ SUMMARY ================")
    print(f"  Stage 5 transfer math: {'PASS' if ok5 else 'FAIL'}")
    print(f"  Stage 6 gate math:     {'PASS' if ok6 else 'FAIL'}")
    print(f"  OVERALL: {'ALL PASS' if (ok5 and ok6) else 'SOME FAILED'}")
    sys.exit(0 if (ok5 and ok6) else 1)


if __name__ == "__main__":
    main()
