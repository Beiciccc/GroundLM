"""Method-validation on synthetic data with KNOWN ground-truth geometry.

Runs entirely on CPU, no model downloads. Purpose: prove the C1 separability
test and the C2 confidence-asymmetry test are correct *before* spending GPU
hours — and that the C1 test does NOT false-positive when the two notions share
a single axis. This doubles as a figure-bearing sanity experiment in the paper.

Scenarios:
  POSITIVE : faithfulness and factuality planted at a true 60 deg axis angle;
             confidence engineered to load on the factuality axis >> faithfulness.
             Expect: cross-angle ~ 60, p < 0.05 (rejects collinearity),
             asymmetry > 0 (factuality more absorbed).
  NULL     : a single shared axis (true angle ~ 0). Expect: cross-angle small,
             p NOT significant (no false positive).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groundlm.probe import separability_test, asymmetry_test  # noqa: E402


def make_dataset(true_angle_deg: float, n: int = 1000, dim: int = 256,
                 a_faith: float = 1.0, a_fact: float = 1.0, noise: float = 1.0,
                 seed: int = 0):
    """Plant two signed axes at a known angle, plus confounds that load on the
    factuality axis far more than the faithfulness axis."""
    rng = np.random.default_rng(seed)
    th = np.radians(true_angle_deg)
    u_fact = np.zeros(dim); u_fact[0] = 1.0
    u_faith = np.zeros(dim); u_faith[0] = np.cos(th); u_faith[1] = np.sin(th)

    faith_sign = rng.choice([-1.0, 1.0], size=n)   # 2x2 balanced & independent
    fact_sign = rng.choice([-1.0, 1.0], size=n)

    X = (a_fact * fact_sign[:, None] * u_fact[None, :]
         + a_faith * faith_sign[:, None] * u_faith[None, :]
         + noise * rng.standard_normal((n, dim)))

    # confound: a softmax-confidence proxy that tracks the FACTUALITY sign
    # strongly and the faithfulness sign negligibly (the asymmetry we hunt for).
    conf = 0.5 + 0.45 * fact_sign + 0.00 * faith_sign + 0.08 * rng.standard_normal(n)
    conf = np.clip(conf, 0.0, 1.0)
    logprob = -1.0 + 0.8 * (conf - 0.5) + 0.1 * rng.standard_normal(n)
    norm = np.linalg.norm(X, axis=1)
    length = rng.integers(3, 12, size=n).astype(float)
    overlap = rng.random(n)
    confounds = np.stack([conf, logprob, norm, length, overlap], axis=1)

    faith_y = (faith_sign > 0).astype(int)
    fact_y = (fact_sign > 0).astype(int)
    return X, faith_y, fact_y, confounds


def run(label: str, true_angle: float, expect_reject: bool):
    X, fy, gy, conf = make_dataset(true_angle, seed=0)
    sep = separability_test(X, fy, gy, n_null=200, n_boot=200, seed=1)
    asym = asymmetry_test(X, fy, gy, conf, n_boot=120, seed=2)

    print(f"\n=== {label}  (planted axis angle = {true_angle:.0f} deg) ===")
    print(f"  cross-notion angle      : {sep.cross_angle:6.2f} deg "
          f"(95% CI {sep.cross_angle_boot_ci[0]:.2f}..{sep.cross_angle_boot_ci[1]:.2f})")
    print(f"  within-faith null mean  : {sep.within_faith_null_mean:6.2f} deg "
          f"(q95 {sep.within_faith_null_q95:.2f})")
    print(f"  within-fact  null mean  : {sep.within_fact_null_mean:6.2f} deg "
          f"(q95 {sep.within_fact_null_q95:.2f})")
    print(f"  shared-axis-removed     : {sep.cross_angle_shared_removed:6.2f} deg")
    print(f"  separability p-value    : {sep.p_value:.4f}  "
          f"-> reject collinear={sep.reject_collinear()}")
    print(f"  absorbed(faith)={asym.faith['absorbed']:.3f} "
          f"(auc {asym.faith['auroc_raw']:.3f}->{asym.faith['auroc_purged']:.3f})")
    print(f"  absorbed(fact) ={asym.fact['absorbed']:.3f} "
          f"(auc {asym.fact['auroc_raw']:.3f}->{asym.fact['auroc_purged']:.3f})")
    print(f"  asymmetry (fact-faith) ={asym.asymmetry:.3f} "
          f"(CI {asym.asymmetry_ci[0]:.3f}..{asym.asymmetry_ci[1]:.3f}, p={asym.p_value:.3f})")

    ok = (sep.reject_collinear() == expect_reject)
    print(f"  CHECK separability outcome == expected({expect_reject}): {'PASS' if ok else 'FAIL'}")
    return ok, sep, asym


def main():
    pos_ok, pos_sep, pos_asym = run("POSITIVE", true_angle=60.0, expect_reject=True)
    null_ok, _, _ = run("NULL (shared axis)", true_angle=2.0, expect_reject=False)

    asym_ok = pos_asym.supports_hypothesis()
    print("\n================ SUMMARY ================")
    print(f"  POSITIVE rejects collinearity : {'PASS' if pos_ok else 'FAIL'}")
    print(f"  NULL does NOT false-positive   : {'PASS' if null_ok else 'FAIL'}")
    print(f"  POSITIVE confidence asymmetry  : {'PASS' if asym_ok else 'FAIL'} "
          f"(fact more absorbed than faith, significant)")
    all_ok = pos_ok and null_ok and asym_ok
    print(f"  OVERALL: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
