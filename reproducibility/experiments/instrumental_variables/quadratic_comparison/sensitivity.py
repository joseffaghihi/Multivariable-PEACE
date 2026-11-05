"""Explain the observed accuracy gap using local coefficient sensitivity.

This analyzes saved fits; no simulation or model selection is repeated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

import comparison as study

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def analyze():
    a, b, lo, hi = 1.0, 0.75, -2.5, 2.5
    crossing = -a / (2 * b)
    normalizer = norm.cdf(hi) - norm.cdf(lo)
    gradient_a = 1 - 2 * (norm.cdf(crossing) - norm.cdf(lo)) / normalizer
    gradient_b = 4 * (norm.pdf(crossing) - norm.pdf(hi)) / normalizer
    truth = float(study.functionals(study.true_theta())["peace"][2])

    def independent_peace(aa, bb):
        root = -aa / (2 * bb)
        return quad(
            lambda x: abs(aa + 2 * bb * x) * norm.pdf(x) / normalizer,
            lo,
            hi,
            points=[root],
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]

    step = 1e-5
    finite_difference = np.array(
        [
            (independent_peace(a + step, b) - independent_peace(a - step, b))
            / (2 * step),
            (independent_peace(a, b + step) - independent_peace(a, b - step))
            / (2 * step),
        ]
    )
    gradient = np.array([gradient_a, gradient_b])
    assert np.max(abs(finite_difference - gradient)) < 1e-8
    assert abs(a * gradient_a + b * gradient_b - truth) < 1e-11
    coefficients = pd.read_csv(OUT / "large_sample_coefficients.csv.gz")
    group = coefficients[
        (coefficients.n == 64000)
        & (coefficients.rho == 0.2)
        & (coefficients.method == "IV degree 2")
    ]
    ab = np.column_stack([group.theta_1, group.theta_2 / np.sqrt(2)])
    covariance = np.cov(ab, rowvar=False)
    summaries = pd.read_csv(OUT / "large_sample_summary.csv")
    selected = summaries[
        (summaries.n == 64000)
        & (summaries.rho == 0.2)
        & (summaries.method == "IV degree 2")
        & (summaries.d == 0.5)
    ].set_index("metric")
    rows = []
    for metric, grad, target in [
        ("peace", gradient, truth),
        ("signed_derivative", np.array([1.0, 0.0]), 1.0),
        ("contrast", np.array([1.0, 1.0]), 1.75),
    ]:
        sd = np.sqrt(grad @ covariance @ grad)
        rows.append(
            {
                "metric": metric,
                "truth": target,
                "gradient_a": grad[0],
                "gradient_b": grad[1],
                "relative_gradient_a": grad[0] / target,
                "relative_gradient_b": grad[1] / target,
                "linearized_empirical_sd": sd,
                "actual_empirical_sd": selected.loc[metric, "sd"],
                "linearized_relative_sd": sd / target,
                "actual_relative_sd": selected.loc[metric, "sd"] / target,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "coefficient_sensitivity.csv", index=False)
    main = pd.read_csv(OUT / "comparison_summary.csv")
    all_summaries = pd.concat([main, summaries], ignore_index=True)
    sample = all_summaries[
        (all_summaries.scenario == "quadratic")
        & (all_summaries.rho == 0.2)
        & (all_summaries.method == "IV degree 2")
        & (all_summaries.d == 0.5)
    ]
    sample = sample[
        [
            "n",
            "metric",
            "truth",
            "median",
            "q05",
            "q95",
            "within_10_percent",
            "within_10_percent_wilson_lower",
            "within_10_percent_wilson_upper",
        ]
    ].sort_values(["n", "metric"])
    assert len(sample) == 15
    sample.to_csv(OUT / "sample_size_comparison.csv", index=False)
    checks = {
        "setting": {"n": 64000, "rho": 0.2, "d": 0.5},
        "true_a": a,
        "true_b": b,
        "mean_peace": truth,
        "mean_peace_gradient_a": gradient_a,
        "mean_peace_gradient_b": gradient_b,
        "coefficient_covariance": covariance.tolist(),
        "coefficient_sd_a": np.sqrt(covariance[0, 0]),
        "coefficient_sd_b": np.sqrt(covariance[1, 1]),
        "max_gradient_finite_difference_error": float(
            np.max(abs(finite_difference - gradient))
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "First-order error propagation at the true quadratic coefficients. Descriptive finite-run covariance; not a universal weak-IV variance or confidence theorem.",
        "all_checks_passed": True,
    }
    (OUT / "sensitivity_analysis.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
