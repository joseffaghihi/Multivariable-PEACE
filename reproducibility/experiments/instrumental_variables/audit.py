"""Independent mathematical and numerical checks for the weak-IV study."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import roots_hermitenorm
from scipy.stats import t
from threadpoolctl import threadpool_limits

import experiment as study

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def independent_derivative(theta, x):
    a = np.pad(np.asarray(theta), (0, 6 - len(theta)))
    return (
        a[1]
        + np.sqrt(2) * a[2] * x
        + a[3] * (3 * x * x - 3) / np.sqrt(6)
        + a[4] * (4 * x**3 - 12 * x) / np.sqrt(24)
        + a[5] * (5 * x**4 - 30 * x * x + 15) / np.sqrt(120)
    )


def independent_score(theta, d, mu=0, sigma=1):
    # Derivative polynomial constructed directly; no study coefficient converter.
    a = np.pad(np.asarray(theta), (0, 6 - len(theta)))
    coefficients = np.array(
        [
            a[1] - 3 * a[3] / np.sqrt(6) + 15 * a[5] / np.sqrt(120),
            np.sqrt(2) * a[2] - 12 * a[4] / np.sqrt(24),
            3 * a[3] / np.sqrt(6) - 30 * a[5] / np.sqrt(120),
            4 * a[4] / np.sqrt(24),
            5 * a[5] / np.sqrt(120),
        ]
    )
    p = np.poly1d(np.trim_zeros(coefficients, "b")[::-1])
    roots = [z.real for z in np.roots(p) if abs(z.imag) < 1e-8 and -2.5 < z.real < 2.5]
    breaks = [-2.5, *sorted(roots), 2.5]
    numerator = sum(
        quad(
            lambda x: (
                abs(independent_derivative(theta, x))
                * np.exp(-d * ((x - mu) / sigma) ** 2)
            ),
            l,
            r,
            epsabs=1e-10,
            epsrel=2e-12,
        )[0]
        for l, r in itertools.pairwise(breaks)
    )
    denominator = quad(
        lambda x: np.exp(-d * ((x - mu) / sigma) ** 2),
        -2.5,
        2.5,
        epsabs=1e-12,
        epsrel=2e-12,
    )[0]
    return numerator / denominator


def hermite_checks():
    nodes, weights = roots_hermitenorm(40)
    weights /= np.sqrt(2 * np.pi)
    max_error = 0
    for rho in study.CONFIG["instrument_correlations"]:
        for w in [-2.2, -0.3, 0, 0.8, 2.4]:
            conditional_x = rho * w + np.sqrt(1 - rho * rho) * nodes
            actual = weights @ study.basis(conditional_x, 5)
            expected = study.basis(np.array([w]), 5)[0] * rho ** np.arange(6)
            max_error = max(max_error, max(abs(actual - expected)))
    assert max_error < 1e-11
    # Independent true coefficients and derivative roots.
    expected_theta = np.array([0, 1, 0.5 * np.sqrt(2), 0.4 * np.sqrt(6)])
    assert np.allclose(study.true_theta(), expected_theta, rtol=0, atol=1e-14)
    assert np.allclose(
        study.real_breaks(np.array([-0.2, 1.0, 1.2])), [-2.5, -1, 1 / 6, 2.5]
    )
    return max_error


def integration_checks(coefficients, raw):
    selected = coefficients.sample(180, random_state=20261104)
    # Include the largest unregularized estimates rather than checking only typical fits.
    extreme_keys = raw.nlargest(20, "estimate")[
        ["repetition", "n", "rho", "method", "penalty"]
    ].drop_duplicates()
    selected = pd.concat([selected, coefficients.merge(extreme_keys)]).drop_duplicates()
    maximum_relative = 0
    doubled_relative = 0
    for row in selected.itertuples():
        theta = np.array([getattr(row, f"theta_{j}") for j in range(6)])
        scores = study.target_values(theta)[0]
        doubled = study.target_values(theta, order=64)[0]
        doubled_relative = max(
            doubled_relative, max(abs(scores - doubled) / (1 + abs(doubled)))
        )
        for d, value in zip(study.DEGREES, scores):
            independent = independent_score(theta, d)
            maximum_relative = max(
                maximum_relative, abs(value - independent) / (1 + abs(independent))
            )
    assert maximum_relative < 1e-9
    assert doubled_relative < 1e-9
    for d, truth in zip(study.DEGREES, study.target_values(study.true_theta())[0]):
        assert abs(independent_score(study.true_theta(), d) - truth) < 1e-11
    for row in selected.head(20).itertuples():
        theta = np.array([getattr(row, f"theta_{j}") for j in range(6)])
        mu, sigma = 0.03, 0.97
        values = study.target_values(theta, mu, sigma)[0]
        for d, value in zip(study.DEGREES, values):
            assert (
                abs(value - independent_score(theta, d, mu, sigma)) / (1 + abs(value))
                < 1e-9
            )
    return {
        "checked_fits": len(selected),
        "max_relative_adaptive_error": maximum_relative,
        "max_relative_doubled_rule_error": doubled_relative,
    }


def coefficient_checks(saved):
    maximum = 0
    cases = [(0, 4000, rho) for rho in [0.8, 0.2, 0.05, 0.0]]
    cases += [(17, 1000, 0.5), (34, 16000, 0.1)]
    for repetition, n, rho in cases:
        w, x, y = study.generate_observed(study.generate_latent(16000, repetition), rho)
        w, x, y = w[:n], x[:n], y[:n]
        rows = saved[
            (saved.repetition == repetition) & (saved.n == n) & (saved.rho == rho)
        ]
        for row in rows.itertuples():
            degree = 5 if row.method.endswith("5") else 3

            # Explicit polynomial columns independently implement the basis.
            def independent_basis(v, degree=degree):
                columns = [
                    np.ones(len(v)),
                    v,
                    (v * v - 1) / np.sqrt(2),
                    (v**3 - 3 * v) / np.sqrt(6),
                ]
                if degree == 5:
                    columns += [
                        (v**4 - 6 * v * v + 3) / np.sqrt(24),
                        (v**5 - 10 * v**3 + 15 * v) / np.sqrt(120),
                    ]
                return np.column_stack(columns)

            p, q = independent_basis(x), independent_basis(w)
            if row.method.startswith("OLS"):
                result = np.linalg.lstsq(p, y, rcond=None)[0]
            elif row.method.startswith("IV"):
                result = np.linalg.solve(q.T @ p, q.T @ y)
            else:
                projected_p = q @ np.linalg.lstsq(q, p, rcond=None)[0]
                penalty = np.diag([0] + [row.penalty] * degree)
                result = np.linalg.solve(
                    p.T @ projected_p / n + penalty, projected_p.T @ y / n
                )
            actual = np.array([getattr(row, f"theta_{j}") for j in range(degree + 1)])
            error = max(abs(result - actual) / (1 + abs(actual)))
            maximum = max(maximum, error)
    assert maximum < 2e-8
    return {"cases": len(cases), "maximum_relative_coefficient_error": maximum}


def linear_checks(raw):
    sampled = raw.sample(5000, random_state=6172026)
    for magnitude in [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
        direct = (
            sampled.ar_a * magnitude**2 + sampled.ar_b * magnitude + sampled.ar_c <= 0
        ) | (sampled.ar_a * magnitude**2 - sampled.ar_b * magnitude + sampled.ar_c <= 0)
        stored = (sampled.ar_lower <= magnitude) & (magnitude <= sampled.ar_upper)
        assert (direct == stored).all(), (
            "Incorrect absolute-value confidence-set projection"
        )
    max_stat_error = 0
    for rho in [0.8, 0.05, 0.0]:
        w, u, r, epsilon = study.generate_latent(1000, int(rho * 1000) + 84)
        x = rho * w + 0.5 * u + np.sqrt(0.75 - rho * rho) * r
        y = x + u + 0.5 * epsilon
        data = np.column_stack([w, x, y])
        s = (data - data.mean(axis=0)).T @ (data - data.mean(axis=0))
        critical = t.ppf(0.975, 998)
        a, b, c = study.ar_quadratic(s, 1000, critical)
        for beta in [-5, -1, 0, 0.8, 1, 1.4, 5]:
            residual = y - beta * x
            design = np.column_stack([np.ones(len(w)), w])
            coef = np.linalg.lstsq(design, residual, rcond=None)[0]
            sigma2 = np.sum((residual - design @ coef) ** 2) / 998
            se = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
            statistic = coef[1] / se
            quadratic = a * beta**2 + b * beta + c
            assert (abs(statistic) <= critical) == (quadratic <= 0)
            # Independent decomposition of the same statistic.
            wc = w - w.mean()
            rc = residual - residual.mean()
            direct = (
                np.sqrt(998)
                * np.dot(wc, rc)
                / np.sqrt(np.dot(wc, wc) * np.dot(rc, rc) - np.dot(wc, rc) ** 2)
            )
            max_stat_error = max(
                max_stat_error, abs(statistic - direct) / (1 + abs(direct))
            )
    assert max_stat_error < 1e-11
    summaries = pd.read_csv(OUT / "linear_summary.csv")
    # Exact coverage should agree with 95% within five Monte Carlo standard errors.
    max_z = max(
        abs(summaries.ar_coefficient_coverage - 0.95) / np.sqrt(0.95 * 0.05 / 5000)
    )
    assert max_z < 5
    return {
        "confidence_set_projection_rows": len(sampled),
        "max_relative_direct_regression_statistic_error": max_stat_error,
        "maximum_AR_coverage_MC_z": max_z,
    }


def data_checks(raw, coefficients, linear):
    expected = study.CONFIG["nonlinear_repetitions"] * 18 * 7
    assert len(coefficients) == expected
    assert len(raw) == expected * len(study.DEGREES)
    assert len(linear) == study.CONFIG["linear_repetitions"] * 18
    assert np.isfinite(raw.select_dtypes("number").to_numpy()).all()
    assert np.isfinite(coefficients.select_dtypes("number").to_numpy()).all()
    assert (raw.estimate >= 0).all()
    counts = raw.groupby(["n", "rho", "method", "penalty", "d"]).size()
    assert (counts == study.CONFIG["nonlinear_repetitions"]).all()
    # Zero-degree normalized weighting cannot depend on an estimated density.
    d0 = raw[raw.d == 0]
    assert np.max(abs(d0.estimate - d0.estimated_density_estimate)) < 1e-8
    # Independently recompute all finite-run bias/SD/RMSE values.
    saved = pd.read_csv(OUT / "nonlinear_summary.csv")
    recomputed = []
    for key, g in raw.groupby(["n", "rho", "method", "penalty", "d"]):
        estimates = np.asarray(g.estimate)
        errors = estimates - g.truth.iloc[0]
        recomputed.append(
            dict(
                zip(["n", "rho", "method", "penalty", "d"], key),
                mean=np.mean(estimates),
                bias=np.mean(errors),
                sd=np.std(estimates, ddof=1),
                rmse=np.sqrt(np.mean(errors**2)),
            )
        )
    checked = saved.merge(
        pd.DataFrame(recomputed),
        on=["n", "rho", "method", "penalty", "d"],
        suffixes=("_saved", "_check"),
    )
    for name in ["mean", "bias", "sd", "rmse"]:
        assert np.allclose(
            checked[name + "_saved"], checked[name + "_check"], rtol=1e-12, atol=1e-12
        )
    return {
        "nonlinear_estimates": len(raw),
        "fitted_responses": len(coefficients),
        "linear_repetitions": len(linear),
        "all_results_retained": True,
        "summaries_independently_recomputed": True,
    }


def main():
    metadata = json.loads((OUT / "run_metadata.json").read_text())
    assert metadata["manuscript_unchanged"]
    assert metadata["manuscript_before"] == study.manuscript_hashes()
    assert json.loads((OUT / "frozen_config.json").read_text()) == study.CONFIG
    assert (
        metadata["source_sha256"]
        == hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest()
    )
    raw = pd.read_csv(OUT / "nonlinear_estimates.csv")
    coefficients = pd.read_csv(OUT / "response_coefficients.csv")
    linear = pd.read_csv(OUT / "linear_estimates.csv")
    checks = {
        "hermite_conditional_expectation_max_error": hermite_checks(),
        "integration": integration_checks(coefficients, raw),
        "estimation": coefficient_checks(coefficients),
        "inference": linear_checks(linear),
        "results": data_checks(raw, coefficients, linear),
        "manuscript_unchanged": True,
        "all_passed": True,
    }
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
