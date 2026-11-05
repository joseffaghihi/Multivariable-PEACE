"""Independent checks of the quadratic follow-up and scalar comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

import comparison as study

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
KEY = study.KEY


def independent_basis(x, degree):
    columns = [np.ones(len(x)), x, (x * x - 1) / np.sqrt(2)]
    if degree >= 3:
        columns.append((x**3 - 3 * x) / np.sqrt(6))
    return np.column_stack(columns)


def independent_polynomial(theta):
    a = np.pad(np.asarray(theta), (0, 6 - len(theta)))
    return np.array(
        [
            a[0] - a[2] / np.sqrt(2) + 3 * a[4] / np.sqrt(24),
            a[1] - 3 * a[3] / np.sqrt(6) + 15 * a[5] / np.sqrt(120),
            a[2] / np.sqrt(2) - 6 * a[4] / np.sqrt(24),
            a[3] / np.sqrt(6) - 10 * a[5] / np.sqrt(120),
            a[4] / np.sqrt(24),
            a[5] / np.sqrt(120),
        ]
    )


def independent_metrics(theta, d, mean=0.0, sd=1.0):
    p = independent_polynomial(theta)
    derivative = np.arange(1, len(p)) * p[1:]
    roots = (
        np.polynomial.polynomial.polyroots(np.trim_zeros(derivative, "b"))
        if np.any(derivative)
        else []
    )
    breaks = [
        -2.5,
        *sorted(z.real for z in roots if abs(z.imag) < 1e-8 and -2.5 < z.real < 2.5),
        2.5,
    ]
    law = lambda x: np.exp(-d * ((x - mean) / sd) ** 2)
    deriv = lambda x: sum(derivative[k] * x**k for k in range(len(derivative)))
    norm = quad(law, -2.5, 2.5, epsabs=1e-12, epsrel=1e-12)[0]
    signed = (
        quad(lambda x: deriv(x) * law(x), -2.5, 2.5, epsabs=1e-10, epsrel=1e-12)[0]
        / norm
    )
    absolute = 0.0
    for k in range(len(breaks) - 1):
        absolute += quad(
            lambda x: abs(deriv(x)) * law(x),
            breaks[k],
            breaks[k + 1],
            epsabs=1e-10,
            epsrel=1e-12,
        )[0]
    return {
        "peace": absolute / norm,
        "signed_derivative": signed,
        "contrast": np.sum(p[1:]),
    }


def analytic_truth_checks():
    truth = pd.read_csv(OUT / "targets.csv")
    maximum = 0.0
    for scenario, theta in [
        ("quadratic", np.array([0, 1, 0.75 * np.sqrt(2)])),
        ("cubic", np.array([0, 1, 0.5 * np.sqrt(2), 0.4 * np.sqrt(6)])),
    ]:
        for d in study.DEGREES:
            independent = independent_metrics(theta, d)
            for metric, value in independent.items():
                saved = truth[
                    (truth.scenario == scenario)
                    & (truth.d == d)
                    & (truth.metric == metric)
                ].truth.iloc[0]
                maximum = max(maximum, abs(value - saved))
    assert maximum < 1e-11
    q = truth[truth.scenario == "quadratic"]
    assert np.allclose(q[q.metric == "signed_derivative"].truth, 1, rtol=0, atol=1e-13)
    assert np.allclose(q[q.metric == "contrast"].truth, 1.75, rtol=0, atol=1e-13)
    return maximum


def numerical_checks(qcoef, cubic_coef):
    maximum = 0.0
    doubled = 0.0
    for coefficients in [
        qcoef.sample(80, random_state=20261104),
        cubic_coef.sample(80, random_state=20261104),
    ]:
        for row in coefficients.itertuples():
            theta = np.array([getattr(row, f"theta_{k}") for k in range(6)])
            primary = study.functionals(theta)
            refined = study.functionals(theta, order=64)
            for j, d in enumerate(study.DEGREES):
                independent = independent_metrics(theta, d)
                for metric in study.METRICS:
                    maximum = max(
                        maximum,
                        abs(primary[metric][j] - independent[metric])
                        / (1 + abs(independent[metric])),
                    )
                    doubled = max(
                        doubled,
                        abs(primary[metric][j] - refined[metric][j])
                        / (1 + abs(refined[metric][j])),
                    )
    assert maximum < 1e-9
    assert doubled < 1e-9
    return {
        "checked_responses": 160,
        "max_relative_adaptive_error": maximum,
        "max_relative_doubled_rule_error": doubled,
    }


def estimator_checks(saved, scalar):
    maximum = 0.0
    pairing_error = 0.0
    for rep, n, rho in [
        (0, 4000, 0.8),
        (0, 4000, 0.2),
        (0, 4000, 0.05),
        (0, 4000, 0.0),
        (49, 1000, 0.5),
        (17, 16000, 0.1),
    ]:
        latent = study.core.generate_latent(16000, rep)
        w, x, y = study.observed(latent, rho)
        old_w, old_x, old_y = study.core.generate_observed(latent, rho)
        assert np.array_equal(w, old_w) and np.array_equal(x, old_x)
        u, e = latent[1], latent[3]
        expected = x + 0.75 * (x * x - 1) + u + 0.5 * e
        pairing_error = max(pairing_error, max(abs(y - expected)))
        assert (
            max(
                abs(
                    (y - old_y)
                    - (
                        x
                        + 0.75 * (x * x - 1)
                        - (0.4 * x**3 + 0.5 * x * x - 0.2 * x - 0.5)
                    )
                )
            )
            < 1e-12
        )
        w, x, y = w[:n], x[:n], y[:n]
        data = saved[(saved.repetition == rep) & (saved.n == n) & (saved.rho == rho)]
        for row in data.itertuples():
            degree = int(row.method[-1])
            p, q = independent_basis(x, degree), independent_basis(w, degree)
            if row.method.startswith("OLS"):
                theta = np.linalg.lstsq(p, y, rcond=None)[0]
            elif row.method.startswith("IV"):
                theta = np.linalg.solve(q.T @ p, q.T @ y)
            else:
                projected = q @ np.linalg.lstsq(q, p, rcond=None)[0]
                penalty = np.diag([0] + [row.penalty] * degree)
                theta = np.linalg.solve(
                    p.T @ projected / n + penalty, projected.T @ y / n
                )
            actual = np.array([getattr(row, f"theta_{j}") for j in range(degree + 1)])
            maximum = max(maximum, max(abs(theta - actual) / (1 + abs(actual))))
        direct = np.polyfit(w, y, 1)[0] / np.polyfit(w, x, 1)[0]
        actual = scalar[
            (scalar.repetition == rep) & (scalar.n == n) & (scalar.rho == rho)
        ].estimate.iloc[0]
        assert abs(direct - actual) / (1 + abs(actual)) < 1e-11
    assert maximum < 1e-8
    return {
        "max_relative_independent_coefficient_error": maximum,
        "common_random_pairing_verified": True,
        "max_explicit_quadratic_generator_error": pairing_error,
        "direct_scalar_ratio_verified": True,
    }


def data_checks(q, c, qcoef, scalar):
    assert (
        len(q) == 315000
        and len(c) == 315000
        and len(qcoef) == 63000
        and len(scalar) == 9000
    )
    for frame in [q, c]:
        assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
        assert (
            frame.groupby(["n", "rho", "method", "penalty", "d"]).size() == 500
        ).all()
        for suffix in ["", "_fitted_density"]:
            gap = frame["peace" + suffix] - abs(frame["signed_derivative" + suffix])
            assert (gap >= -1e-9 * (1 + frame["peace" + suffix])).all()
        d0 = frame[frame.d == 0]
        assert np.allclose(d0.peace, d0.peace_fitted_density, rtol=0, atol=1e-9)
        assert np.allclose(
            d0.signed_derivative, d0.signed_derivative_fitted_density, rtol=0, atol=1e-9
        )
        assert np.array_equal(frame.contrast, frame.contrast_fitted_density)
    parent = pd.read_csv(study.PARENT / "results/nonlinear_estimates.csv")
    joined = c.merge(parent, on=KEY + ["d"], validate="one_to_one")
    assert np.array_equal(joined.peace.to_numpy(), joined.estimate.to_numpy())
    assert np.array_equal(
        joined.peace_fitted_density.to_numpy(),
        joined.estimated_density_estimate.to_numpy(),
    )
    # Independently verify all ordinary averages and error summaries.
    summary = pd.read_csv(OUT / "comparison_summary.csv")
    largest_summary_error = 0.0
    for scenario, frame in [("quadratic", q), ("cubic", c)]:
        lookup = summary[
            (summary.scenario == scenario) & (summary.method != "Direct scalar IV")
        ].set_index(["n", "rho", "method", "penalty", "d", "metric"])
        for key, g in frame.groupby(["n", "rho", "method", "penalty", "d"]):
            for metric in study.METRICS:
                if metric == "contrast" and key[-1] != 0.5:
                    continue
                row = lookup.loc[(*key, metric)]
                values = g[metric].to_numpy()
                truth = g[metric + "_truth"].iloc[0]
                err = values - truth
                expected = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "bias": np.mean(err),
                    "sd": np.std(values, ddof=1),
                    "rmse": np.sqrt(np.mean(err**2)),
                    "within_10_percent": np.mean(abs(err) <= 0.1 * abs(truth)),
                }
                for name, value in expected.items():
                    relative = abs(value - row[name]) / (1 + abs(value))
                    largest_summary_error = max(largest_summary_error, relative)
    assert largest_summary_error < 1e-11
    # A quadratic derivative's signed symmetric average equals its linear coefficient.
    quadratic_class = q[q.method.str.endswith("2")]
    signed = quadratic_class.merge(
        qcoef[KEY + ["theta_1"]], on=KEY, validate="many_to_one"
    )
    assert np.allclose(signed.signed_derivative, signed.theta_1, rtol=1e-10, atol=1e-9)
    return {
        "quadratic_degree_rows": len(q),
        "cubic_degree_rows": len(c),
        "quadratic_fits": len(qcoef),
        "scalar_estimates": len(scalar),
        "all_original_PEACE_estimates_preserved_exactly": True,
        "max_relative_recomputed_summary_error": largest_summary_error,
        "no_extreme_estimates_removed": True,
        "PEACE_dominates_absolute_signed_average": True,
    }


def main():
    metadata = json.loads((OUT / "metadata.json").read_text())
    assert metadata["protected_files_unchanged"]
    assert metadata["protected_before"] == study.protected_hashes()
    assert (
        metadata["source_sha256"]
        == hashlib.sha256((ROOT / "comparison.py").read_bytes()).hexdigest()
    )
    assert study.CONFIG == json.loads((OUT / "frozen_config.json").read_text())
    q = pd.read_csv(OUT / "quadratic_estimates.csv.gz")
    c = pd.read_csv(OUT / "cubic_comparison_estimates.csv.gz")
    coef = pd.read_csv(OUT / "quadratic_coefficients.csv.gz")
    scalar = pd.read_csv(OUT / "scalar_signed_estimates.csv.gz")
    cubic_coef = pd.read_csv(study.PARENT / "results/response_coefficients.csv")
    checks = {
        "truth_max_absolute_error": analytic_truth_checks(),
        "quadrature": numerical_checks(coef, cubic_coef),
        "estimation": estimator_checks(coef, scalar),
        "results": data_checks(q, c, coef, scalar),
        "large_sample_check": large_sample_checks(),
        "original_study_unchanged": True,
        "manuscript_unchanged": True,
        "all_passed": True,
    }
    (OUT / "audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def large_sample_checks():
    metadata = json.loads((OUT / "large_sample_metadata.json").read_text())
    assert metadata["protected_before"] == study.protected_hashes()
    assert (
        metadata["source_sha256"]
        == hashlib.sha256((ROOT / "large_sample.py").read_bytes()).hexdigest()
    )
    assert (
        metadata["comparison_source_sha256"]
        == hashlib.sha256((ROOT / "comparison.py").read_bytes()).hexdigest()
    )
    raw = pd.read_csv(OUT / "large_sample_estimates.csv.gz")
    coef = pd.read_csv(OUT / "large_sample_coefficients.csv.gz")
    scalar = pd.read_csv(OUT / "large_sample_scalar.csv.gz")
    summary = pd.read_csv(OUT / "large_sample_summary.csv")
    assert len(raw) == 45000 and len(coef) == 9000 and len(scalar) == 3000
    assert np.isfinite(raw.select_dtypes("number").to_numpy()).all()
    assert (raw.groupby(["n", "rho", "method", "penalty", "d"]).size() == 500).all()
    for metric in study.METRICS:
        assert np.isfinite(raw[metric]).all()
    assert (raw.peace - abs(raw.signed_derivative) >= -1e-9 * (1 + raw.peace)).all()
    for row in coef.sample(40, random_state=5417).itertuples():
        theta = np.array([getattr(row, f"theta_{j}") for j in range(6)])
        group = raw[
            (raw.repetition == row.repetition)
            & (raw.n == row.n)
            & (raw.rho == row.rho)
            & (raw.method == row.method)
            & (raw.penalty == row.penalty)
        ].set_index("d")
        for d in study.DEGREES:
            independently = independent_metrics(theta, d)
            for metric, value in independently.items():
                assert abs(value - group.loc[d, metric]) / (1 + abs(value)) < 1e-9
    lookup = summary[summary.method != "Direct scalar IV"].set_index(
        ["n", "rho", "method", "penalty", "d", "metric"]
    )
    for key, g in raw.groupby(["n", "rho", "method", "penalty", "d"]):
        for metric in study.METRICS:
            if metric == "contrast" and key[-1] != 0.5:
                continue
            row = lookup.loc[(*key, metric)]
            values = g[metric].to_numpy()
            truth = g[metric + "_truth"].iloc[0]
            assert np.isclose(row["median"], np.median(values), rtol=1e-12, atol=1e-12)
            assert np.isclose(
                row.rmse,
                np.sqrt(np.mean((values - truth) ** 2)),
                rtol=1e-12,
                atol=1e-12,
            )
            assert np.isclose(
                row.within_10_percent,
                np.mean(abs(values - truth) <= 0.1 * abs(truth)),
                rtol=0,
                atol=1e-12,
            )
    large_latent = study.core.generate_latent(256000, 0)
    assert np.array_equal(large_latent[:, :16000], study.core.generate_latent(16000, 0))
    maximum = 0.0
    for n, rho in [(64000, 0.2), (256000, 0.05)]:
        w, x, y = study.observed(large_latent, rho)
        w, x, y = w[:n], x[:n], y[:n]
        p, q = independent_basis(x, 2), independent_basis(w, 2)
        independent = np.linalg.solve(q.T @ p, q.T @ y)
        row = coef[
            (coef.repetition == 0)
            & (coef.n == n)
            & (coef.rho == rho)
            & (coef.method == "IV degree 2")
        ].iloc[0]
        actual = row[["theta_0", "theta_1", "theta_2"]].to_numpy(dtype=float)
        maximum = max(maximum, max(abs(independent - actual) / (1 + abs(actual))))
    assert maximum < 1e-9
    return {
        "large_sample_datasets": len(scalar),
        "response_fits": len(coef),
        "degree_rows": len(raw),
        "independent_functional_checks": 40,
        "nested_samples_verified": True,
        "max_relative_independent_IV_coefficient_error": maximum,
        "all_six_cells_retained": True,
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
