"""Controlled weak-instrument benchmark for a causal mean-response PEACE target.

Run `python experiment.py run`, then `python audit.py` and `python report.py`.
The manuscript and the previously published experiments are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
import scipy
from numpy.polynomial import hermite_e as he
from numpy.polynomial import polynomial as poly
from scipy.special import ndtr, roots_legendre
from scipy.stats import t, wishart
from threadpoolctl import threadpool_limits

CONFIG = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "results"
DOMAIN = tuple(CONFIG["evaluation_domain"])
DEGREES = np.array(CONFIG["degrees"])
TRUE_HE = np.array(CONFIG["hermite_probabilists_coefficients"])
MANUSCRIPT = ROOT.parent / "manuscript"


def manuscript_hashes():
    return {
        name: hashlib.sha256((MANUSCRIPT / name).read_bytes()).hexdigest()
        for name in ["PEACE.tex", "Appendix.tex", "References.bib"]
        if (MANUSCRIPT / name).is_file()
    }


def basis(x, degree):
    """Orthonormal Hermite basis under N(0,1), including a constant."""
    return he.hermevander(x, degree) / np.sqrt(
        np.array([math.factorial(k) for k in range(degree + 1)])
    )


def power_coefficients(theta):
    scaling = np.sqrt([math.factorial(k) for k in range(len(theta))])
    return he.herme2poly(theta / scaling)


def true_theta(degree=3):
    result = np.zeros(degree + 1)
    result[:4] = TRUE_HE * np.sqrt([math.factorial(k) for k in range(4)])
    return result


def real_breaks(coefficients):
    roots = poly.polyroots(poly.polytrim(coefficients)) if len(coefficients) > 1 else []
    interior = sorted(
        float(z.real)
        for z in roots
        if abs(z.imag) < 1e-8 * (1 + abs(z.real)) and DOMAIN[0] < z.real < DOMAIN[1]
    )
    return np.array([DOMAIN[0], *interior, DOMAIN[1]])


def interval_rule(breaks, order=32):
    x, w = roots_legendre(order)
    left, right = breaks[:-1, None], breaks[1:, None]
    return (
        ((right + left) / 2 + (right - left) / 2 * x).ravel(),
        ((right - left) / 2 * w).ravel(),
    )


def target_values(theta, mean=0.0, sd=1.0, order=None):
    """Normalized and unnormalized variation under a restricted normal law."""
    if order is None:
        order = CONFIG["quadrature_order_per_sign_interval"]
    derivative = poly.polyder(power_coefficients(theta))
    x, w = interval_rule(real_breaks(derivative), order)
    # Split at all derivative roots before applying the smooth quadrature rule.
    z = np.exp(-DEGREES[:, None] * ((x[None, :] - mean) / sd) ** 2)
    integrals = z @ w
    numerators = z @ (w * abs(poly.polyval(x, derivative)))
    normal_probability = ndtr((DOMAIN[1] - mean) / sd) - ndtr((DOMAIN[0] - mean) / sd)
    density_scale = 1 / (sd * np.sqrt(2 * np.pi) * normal_probability)
    normalizers = integrals * density_scale ** (2 * DEGREES)
    return (
        numerators / integrals,
        numerators * density_scale ** (2 * DEGREES),
        normalizers,
    )


def invsqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    if min(values) <= 0:
        raise np.linalg.LinAlgError("Non-positive empirical Gram matrix")
    return (vectors / np.sqrt(values)) @ vectors.T


def centered_moments(p, q, y):
    pc = p[:, 1:] - p[:, 1:].mean(axis=0)
    qc = q[:, 1:] - q[:, 1:].mean(axis=0)
    yc = y - y.mean()
    n = len(y)
    gxx, gqq, gqx = pc.T @ pc / n, qc.T @ qc / n, qc.T @ pc / n
    gqy, gxy = qc.T @ yc / n, pc.T @ yc / n
    white_q = invsqrt(gqq)
    a, b = white_q @ gqx, white_q @ gqy
    smallest = np.linalg.svd(a @ invsqrt(gxx), compute_uv=False)[-1]
    return a, b, gxx, gxy, smallest


def add_intercept(nonconstant, p, y):
    return np.r_[y.mean() - p[:, 1:].mean(axis=0) @ nonconstant, nonconstant]


def estimate_models(x, w, y):
    models = []
    for degree in [3, 5]:
        p, q = basis(x, degree), basis(w, degree)
        a, b, gxx, gxy, smallest = centered_moments(p, q, y)
        if degree == 3:
            ols = np.linalg.solve(gxx, gxy)
            models.append(("OLS cubic", 0.0, add_intercept(ols, p, y), smallest))
        # Exactly identified series 2SLS: do not silently truncate weak directions.
        fitted = np.linalg.solve(a, b)
        models.append(
            (f"IV degree {degree}", 0.0, add_intercept(fitted, p, y), smallest)
        )
        penalties = (
            CONFIG["ridge_penalties"]
            if degree == 3
            else [CONFIG["primary_ridge_penalty"]]
        )
        for penalty in penalties:
            fitted = np.linalg.solve(a.T @ a + penalty * np.eye(degree), a.T @ b)
            models.append(
                (
                    f"Ridge IV degree {degree}",
                    penalty,
                    add_intercept(fitted, p, y),
                    smallest,
                )
            )
    return models


def generate_latent(n, repetition):
    return (
        np.random.default_rng(np.random.SeedSequence([CONFIG["seed"], repetition, 1]))
        .standard_normal((n, 4))
        .T
    )


def generate_observed(latent, rho):
    w, u, r, e = latent
    x = rho * w + 0.5 * u + np.sqrt(0.75 - rho**2) * r
    y = he.hermeval(x, TRUE_HE) + u + 0.5 * e
    return w, x, y


def linear_first_stage(w, x):
    wc, xc = w - w.mean(), x - x.mean()
    correlation = np.dot(wc, xc) / np.sqrt(np.dot(wc, wc) * np.dot(xc, xc))
    return (len(x) - 2) * correlation**2 / (1 - correlation**2)


def nonlinear_run():
    truth, unnormalized, normalizers = target_values(true_theta())
    observational, observational_un, _ = target_values(
        true_theta() + np.array([0, 0.5, 0, 0])
    )
    pd.DataFrame(
        {
            "d": DEGREES,
            "truth": truth,
            "unnormalized_truth": unnormalized,
            "normalizer": normalizers,
            "population_OLS_target": observational,
            "population_OLS_unnormalized": observational_un,
        }
    ).to_csv(OUT / "targets.csv", index=False)
    node, weight = interval_rule(np.array(DOMAIN), 96)
    natural_weight = weight * np.exp(-(node**2) / 2)
    natural_weight /= natural_weight.sum()
    true_response = he.hermeval(node, TRUE_HE)
    true_derivative = he.hermeval(node, he.hermeder(TRUE_HE))
    metric_rows, coefficient_rows, diagnostic_rows = [], [], []
    for repetition in range(CONFIG["nonlinear_repetitions"]):
        latent = generate_latent(max(CONFIG["sample_sizes"]), repetition)
        for rho in CONFIG["instrument_correlations"]:
            wf, xf, yf = generate_observed(latent, rho)
            for n in CONFIG["sample_sizes"]:
                w, x, y = wf[:n], xf[:n], yf[:n]
                density_mean, density_sd = x.mean(), x.std(ddof=0)
                first_stage = linear_first_stage(w, x)
                for method, penalty, theta, smallest in estimate_models(x, w, y):
                    score, unnorm, _ = target_values(theta)
                    density_score, _, _ = target_values(theta, density_mean, density_sd)
                    pcoef = power_coefficients(theta)
                    response_error = np.sqrt(
                        np.dot(
                            natural_weight,
                            (poly.polyval(node, pcoef) - true_response) ** 2,
                        )
                    )
                    derivative_error = np.sqrt(
                        np.dot(
                            natural_weight,
                            (poly.polyval(node, poly.polyder(pcoef)) - true_derivative)
                            ** 2,
                        )
                    )
                    key = {
                        "repetition": repetition,
                        "n": n,
                        "rho": rho,
                        "method": method,
                        "penalty": penalty,
                    }
                    coefficients = {
                        f"theta_{j}": float(theta[j]) if j < len(theta) else 0.0
                        for j in range(6)
                    }
                    coefficient_rows.append(key | coefficients)
                    diagnostic_rows.append(
                        key
                        | {
                            "first_stage_F": first_stage,
                            "empirical_smallest_canonical": smallest,
                            "population_smallest_canonical": rho ** (len(theta) - 1),
                            "response_L2_error": response_error,
                            "derivative_L2_error": derivative_error,
                            "density_mean": density_mean,
                            "density_sd": density_sd,
                        }
                    )
                    for i, d in enumerate(DEGREES):
                        metric_rows.append(
                            key
                            | {
                                "d": d,
                                "truth": truth[i],
                                "estimate": score[i],
                                "unnormalized_truth": unnormalized[i],
                                "unnormalized_estimate": unnorm[i],
                                "estimated_density_estimate": density_score[i],
                            }
                        )
        if (repetition + 1) % 25 == 0:
            print(
                f"Nonlinear repetitions: {repetition + 1}/{CONFIG['nonlinear_repetitions']}",
                flush=True,
            )
    raw = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    raw.to_csv(OUT / "nonlinear_estimates.csv", index=False)
    coefficients.to_csv(OUT / "response_coefficients.csv", index=False)
    diagnostics.to_csv(OUT / "response_diagnostics.csv", index=False)
    summary = []
    for key, group in raw.groupby(["n", "rho", "method", "penalty", "d"], sort=True):
        errors = group.estimate.to_numpy() - group.truth.iloc[0]
        estimates = group.estimate.to_numpy()
        r = len(group)
        rmse = np.sqrt(np.mean(errors**2))
        summary.append(
            dict(
                zip(["n", "rho", "method", "penalty", "d"], key),
                repetitions=r,
                truth=group.truth.iloc[0],
                mean=estimates.mean(),
                bias=errors.mean(),
                sd=estimates.std(ddof=1),
                rmse=rmse,
                mean_mcse=estimates.std(ddof=1) / np.sqrt(r),
                rmse_mcse=(np.std(errors**2, ddof=1) / np.sqrt(r) / (2 * rmse))
                if rmse
                else 0,
                median=np.median(estimates),
                q05=np.quantile(estimates, 0.05),
                q95=np.quantile(estimates, 0.95),
                q99=np.quantile(estimates, 0.99),
                within_10_percent=np.mean(abs(errors) <= 0.1 * group.truth.iloc[0]),
                relative_error_over_100_percent=np.mean(
                    abs(errors) > group.truth.iloc[0]
                ),
                density_shift_median=np.median(
                    group.estimated_density_estimate - group.estimate
                ),
                density_shift_abs_q95=np.quantile(
                    abs(group.estimated_density_estimate - group.estimate), 0.95
                ),
                estimated_density_rmse=np.sqrt(
                    np.mean((group.estimated_density_estimate - group.truth) ** 2)
                ),
            )
        )
    pd.DataFrame(summary).to_csv(OUT / "nonlinear_summary.csv", index=False)
    diagnostics.groupby(["n", "rho", "method", "penalty"]).agg(
        median_first_stage_F=("first_stage_F", "median"),
        median_empirical_smallest_canonical=("empirical_smallest_canonical", "median"),
        median_response_error=("response_L2_error", "median"),
        median_derivative_error=("derivative_L2_error", "median"),
        q95_derivative_error=("derivative_L2_error", lambda v: v.quantile(0.95)),
    ).reset_index().to_csv(OUT / "response_summary.csv", index=False)


def ar_quadratic(s, n, critical):
    sww, sxx, syy = s[..., 0, 0], s[..., 1, 1], s[..., 2, 2]
    swx, swy, sxy = s[..., 0, 1], s[..., 0, 2], s[..., 1, 2]
    multiplier = n - 2 + critical**2
    a = multiplier * swx**2 - critical**2 * sww * sxx
    b = -2 * multiplier * swx * swy + 2 * critical**2 * sww * sxy
    c = multiplier * swy**2 - critical**2 * sww * syy
    return a, b, c


def ar_set(a, b, c):
    """Return coefficient-set type and bounds of its absolute-value image.

    The original coefficient set is {beta: a*beta^2+b*beta+c <= 0}.
    Its coefficients are saved, including when the set is disconnected.
    """
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return ("all real", 0.0, np.inf) if a < 0 else ("empty", np.nan, np.nan)
    if a == 0:
        if b == 0:
            return ("all real", 0.0, np.inf) if c <= 0 else ("empty", np.nan, np.nan)
        bound = -c / b
        lower = max(0.0, -bound) if b > 0 else max(0.0, bound)
        return "half line", lower, np.inf
    # Stable quadratic roots, including a small root far from the other root.
    root = np.sqrt(discriminant)
    q = -0.5 * (b + np.copysign(root, b))
    roots = sorted([q / a, c / q]) if q != 0 else [0.0, 0.0]
    left, right = roots
    if a < 0:
        lower = 0.0 if left >= 0 or right <= 0 else min(-left, right)
        return "two rays", lower, np.inf
    lower = 0.0 if left <= 0 <= right else min(abs(left), abs(right))
    return "bounded", lower, max(abs(left), abs(right))


def linear_run():
    rows = []
    count = CONFIG["linear_repetitions"]
    beta = CONFIG["linear_beta"]
    for n in CONFIG["sample_sizes"]:
        critical = t.ppf(0.975, n - 2)
        for ri, rho in enumerate(CONFIG["instrument_correlations"]):
            # Exact centered cross-product law for independent Gaussian rows.
            covariance = np.array(
                [
                    [1, rho, beta * rho],
                    [rho, 1, beta + 0.5],
                    [beta * rho, beta + 0.5, beta**2 + beta + 1.25],
                ]
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([CONFIG["seed"], n, ri, 2])
            )
            s = wishart.rvs(df=n - 1, scale=covariance, size=count, random_state=rng)
            sww, sxx, syy = s[:, 0, 0], s[:, 1, 1], s[:, 2, 2]
            swx, swy, sxy = s[:, 0, 1], s[:, 0, 2], s[:, 1, 2]
            estimate = swy / swx
            residual_ss = syy - 2 * estimate * sxy + estimate**2 * sxx
            se = np.sqrt(residual_ss / (n - 2) * sww / swx**2)
            wald_lower = np.maximum(0, abs(estimate) - critical * se)
            wald_upper = abs(estimate) + critical * se
            a, b, c = ar_quadratic(s, n, critical)
            first_stage = (n - 2) * swx**2 / (sww * sxx - swx**2)
            for rep in range(count):
                kind, lower, upper = ar_set(a[rep], b[rep], c[rep])
                rows.append(
                    {
                        "repetition": rep,
                        "n": n,
                        "rho": rho,
                        "truth": abs(beta),
                        "coefficient_estimate": estimate[rep],
                        "estimate": abs(estimate[rep]),
                        "conventional_se": se[rep],
                        "first_stage_F": first_stage[rep],
                        "wald_lower": wald_lower[rep],
                        "wald_upper": wald_upper[rep],
                        "wald_coverage": wald_lower[rep]
                        <= abs(beta)
                        <= wald_upper[rep],
                        "wald_coefficient_coverage": abs(estimate[rep] - beta)
                        <= critical * se[rep],
                        "ar_a": a[rep],
                        "ar_b": b[rep],
                        "ar_c": c[rep],
                        "ar_type": kind,
                        "ar_lower": lower,
                        "ar_upper": upper,
                        "ar_coverage": lower <= abs(beta) <= upper,
                        "ar_coefficient_coverage": (
                            a[rep] * beta**2 + b[rep] * beta + c[rep] <= 0
                        ),
                        "ar_unbounded": not np.isfinite(upper),
                    }
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "linear_estimates.csv", index=False)
    summaries = []
    for (n, rho), g in raw.groupby(["n", "rho"], sort=True):
        record = {
            "n": n,
            "rho": rho,
            "repetitions": len(g),
            "truth": abs(beta),
            "median_estimate": g.estimate.median(),
            "q05": g.estimate.quantile(0.05),
            "q95": g.estimate.quantile(0.95),
            "median_first_stage_F": g.first_stage_F.median(),
            "median_wald_length": (g.wald_upper - g.wald_lower).median(),
        }
        for name in [
            "wald_coverage",
            "wald_coefficient_coverage",
            "ar_coverage",
            "ar_coefficient_coverage",
            "ar_unbounded",
        ]:
            p = g[name].mean()
            record[name] = p
            record[name + "_mcse"] = np.sqrt(p * (1 - p) / len(g))
        finite = g[np.isfinite(g.ar_upper)]
        record["bounded_ar_length_median"] = (
            (finite.ar_upper - finite.ar_lower).median() if len(finite) else np.nan
        )
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(OUT / "linear_summary.csv", index=False)
    print("Linear inference: 5,000 repetitions per cell completed", flush=True)


def run():
    OUT.mkdir(exist_ok=True)
    before = manuscript_hashes()
    start = time.monotonic()
    (OUT / "frozen_config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    nonlinear_run()
    linear_run()
    after = manuscript_hashes()
    metadata = {
        "config": CONFIG,
        "design_sha256": hashlib.sha256((ROOT / "DESIGN.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(
            (ROOT / "config.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "runtime_seconds": time.monotonic() - start,
        "manuscript_before": before,
        "manuscript_after": after,
        "manuscript_unchanged": before == after,
        "manuscript_check_available": len(before) == 3,
        "nonlinear_dataset_cells": len(CONFIG["sample_sizes"])
        * len(CONFIG["instrument_correlations"])
        * CONFIG["nonlinear_repetitions"],
        "linear_dataset_cells": len(CONFIG["sample_sizes"])
        * len(CONFIG["instrument_correlations"])
        * CONFIG["linear_repetitions"],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert before == after, "The manuscript changed during this run"
    print(f"Completed in {metadata['runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run()
