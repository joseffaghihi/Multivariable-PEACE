"""Quadratic weak-IV follow-up and comparisons with causal scalar summaries.

The parent study is a local dependency; its sources and results are read only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import pandas as pd
from numpy.polynomial import hermite_e as he
from numpy.polynomial import polynomial as poly
from scipy.special import roots_legendre
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location("weak_iv_core", PARENT / "experiment.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
CONFIG = json.loads((ROOT / "config.json").read_text())
DEGREES = np.array(CONFIG["degrees"])
TRUE_HE = np.array(CONFIG["hermite_probabilists_coefficients"])
KEY = ["repetition", "n", "rho", "method", "penalty"]
METRICS = ["peace", "signed_derivative", "contrast"]
COMPRESS = {"method": "gzip", "mtime": 0}


def hashes(paths, relative_to):
    return {
        str(p.relative_to(relative_to)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def protected_hashes():
    files = [
        p
        for p in PARENT.iterdir()
        if p.is_file() and p.suffix in {".py", ".md", ".html", ".json", ".txt"}
    ]
    files += [p for p in (PARENT / "results").iterdir() if p.is_file()]
    manuscript = core.manuscript_hashes()
    return {"original_study": hashes(files, PARENT), "manuscript": manuscript}


def true_theta():
    return TRUE_HE * np.sqrt([math.factorial(k) for k in range(3)])


def functionals(theta, mean=0.0, sd=1.0, order=32):
    p = core.power_coefficients(theta)
    derivative = poly.polyder(p)
    x, w = core.interval_rule(core.real_breaks(derivative), order)
    weights = w[None, :] * np.exp(-DEGREES[:, None] * ((x[None, :] - mean) / sd) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    values = poly.polyval(x, derivative)
    lo, hi = CONFIG["intervention_contrast"]
    return {
        "peace": weights @ abs(values),
        "signed_derivative": weights @ values,
        "contrast": np.repeat(poly.polyval(hi, p) - poly.polyval(lo, p), len(DEGREES)),
    }


def fit_models(x, w, y):
    models = []
    for degree in [2, 3]:
        p, q = core.basis(x, degree), core.basis(w, degree)
        a, b, gxx, gxy, smallest = core.centered_moments(p, q, y)
        if degree == 2:
            models.append(
                (
                    "OLS degree 2",
                    0.0,
                    core.add_intercept(np.linalg.solve(gxx, gxy), p, y),
                    smallest,
                )
            )
        nonconstant = np.linalg.solve(a, b)
        models.append(
            (
                f"IV degree {degree}",
                0.0,
                core.add_intercept(nonconstant, p, y),
                smallest,
            )
        )
        penalties = (
            CONFIG["ridge_penalties"]
            if degree == 2
            else [CONFIG["primary_ridge_penalty"]]
        )
        for penalty in penalties:
            nonconstant = np.linalg.solve(a.T @ a + penalty * np.eye(degree), a.T @ b)
            models.append(
                (
                    f"Ridge IV degree {degree}",
                    penalty,
                    core.add_intercept(nonconstant, p, y),
                    smallest,
                )
            )
    return models


def observed(latent, rho):
    w, u, r, e = latent
    x = rho * w + 0.5 * u + np.sqrt(0.75 - rho * rho) * r
    return w, x, he.hermeval(x, TRUE_HE) + u + 0.5 * e


def wide_rows(key, values, plugin, truth):
    for j, d in enumerate(DEGREES):
        row = key | {"d": d}
        for name in METRICS:
            row[name] = values[name][j]
            row[name + "_truth"] = truth[name][j]
            row[name + "_fitted_density"] = plugin[name][j]
        yield row


def run_quadratic():
    truth = functionals(true_theta())
    rows, coefficients, diagnostics, scalar = [], [], [], []
    nodes, weights = core.interval_rule(np.array(CONFIG["evaluation_domain"]), 80)
    weights *= np.exp(-(nodes**2) / 2)
    weights /= weights.sum()
    ht = he.hermeval(nodes, TRUE_HE)
    dt = he.hermeval(nodes, he.hermeder(TRUE_HE))
    for rep in range(CONFIG["repetitions"]):
        latent = core.generate_latent(max(CONFIG["sample_sizes"]), rep)
        for rho in CONFIG["instrument_correlations"]:
            wf, xf, yf = observed(latent, rho)
            for n in CONFIG["sample_sizes"]:
                w, x, y = wf[:n], xf[:n], yf[:n]
                mu, sigma = x.mean(), x.std(ddof=0)
                f = core.linear_first_stage(w, x)
                wc, xc, yc = w - w.mean(), x - x.mean(), y - y.mean()
                scalar.append(
                    {
                        "repetition": rep,
                        "n": n,
                        "rho": rho,
                        "estimate": np.dot(wc, yc) / np.dot(wc, xc),
                        "truth": 1.0,
                        "first_stage_F": f,
                    }
                )
                for method, penalty, theta, smallest in fit_models(x, w, y):
                    key = {
                        "repetition": rep,
                        "n": n,
                        "rho": rho,
                        "method": method,
                        "penalty": penalty,
                    }
                    values = functionals(theta)
                    plugin = functionals(theta, mu, sigma)
                    rows.extend(wide_rows(key, values, plugin, truth))
                    coefficients.append(
                        key
                        | {
                            f"theta_{j}": theta[j] if j < len(theta) else 0.0
                            for j in range(6)
                        }
                    )
                    p = core.power_coefficients(theta)
                    diagnostics.append(
                        key
                        | {
                            "first_stage_F": f,
                            "model_degree": len(theta) - 1,
                            "empirical_smallest_canonical": smallest,
                            "population_smallest_canonical": rho ** (len(theta) - 1),
                            "mode_signal_scale": n * rho ** (2 * (len(theta) - 1)),
                            "response_L2_error": np.sqrt(
                                np.dot(weights, (poly.polyval(nodes, p) - ht) ** 2)
                            ),
                            "derivative_L2_error": np.sqrt(
                                np.dot(
                                    weights,
                                    (poly.polyval(nodes, poly.polyder(p)) - dt) ** 2,
                                )
                            ),
                            "density_mean": mu,
                            "density_sd": sigma,
                        }
                    )
        if (rep + 1) % 50 == 0:
            print(
                f"Quadratic repetitions: {rep + 1}/{CONFIG['repetitions']}", flush=True
            )
    frames = {}
    for name, data in [
        ("quadratic_estimates", rows),
        ("quadratic_coefficients", coefficients),
        ("quadratic_diagnostics", diagnostics),
        ("scalar_signed_estimates", scalar),
    ]:
        frame = pd.DataFrame(data)
        frame.to_csv(OUT / (name + ".csv.gz"), index=False, compression=COMPRESS)
        frames[name] = frame
    diagnostic = frames["quadratic_diagnostics"]
    diagnostic.groupby(["n", "rho", "method", "penalty"]).agg(
        median_first_stage_F=("first_stage_F", "median"),
        median_smallest_canonical=("empirical_smallest_canonical", "median"),
        median_response_error=("response_L2_error", "median"),
        median_derivative_error=("derivative_L2_error", "median"),
        q95_derivative_error=("derivative_L2_error", lambda v: v.quantile(0.95)),
    ).reset_index().to_csv(OUT / "quadratic_response_summary.csv", index=False)
    return frames


def signed_batch(coefficients, means, sds):
    """Smooth derivative quadrature; weights may differ across fitted densities."""
    nodes, weights = roots_legendre(48)
    lo, hi = CONFIG["evaluation_domain"]
    nodes = (hi + lo) / 2 + (hi - lo) / 2 * nodes
    weights *= (hi - lo) / 2
    derivative_columns = np.column_stack(
        [np.zeros(len(nodes))]
        + [np.sqrt(k) * core.basis(nodes, 4)[:, k - 1] for k in range(1, 6)]
    )
    answer = np.empty((len(coefficients), len(DEGREES)))
    for start in range(0, len(coefficients), 1000):
        end = min(start + 1000, len(coefficients))
        derivatives = coefficients[start:end] @ derivative_columns.T
        z2 = ((nodes[None, :] - means[start:end, None]) / sds[start:end, None]) ** 2
        for j, d in enumerate(DEGREES):
            law = weights[None, :] * np.exp(-d * z2)
            answer[start:end, j] = np.sum(derivatives * law, axis=1) / law.sum(axis=1)
    return answer


def augment_cubic():
    coefficients = pd.read_csv(PARENT / "results/response_coefficients.csv")
    diagnostics = pd.read_csv(PARENT / "results/response_diagnostics.csv")
    old = pd.read_csv(PARENT / "results/nonlinear_estimates.csv")
    merged = coefficients.merge(
        diagnostics[KEY + ["density_mean", "density_sd"]], on=KEY, validate="one_to_one"
    )
    theta = merged[[f"theta_{j}" for j in range(6)]].to_numpy()
    signed = signed_batch(theta, np.zeros(len(theta)), np.ones(len(theta)))
    plugin_signed = signed_batch(
        theta, merged.density_mean.to_numpy(), merged.density_sd.to_numpy()
    )
    lo, hi = CONFIG["intervention_contrast"]
    contrast = theta @ (
        core.basis(np.array([hi]), 5)[0] - core.basis(np.array([lo]), 5)[0]
    )
    values = []
    for j, d in enumerate(DEGREES):
        v = merged[KEY].copy()
        v["d"] = d
        v["signed_derivative"] = signed[:, j]
        v["signed_derivative_fitted_density"] = plugin_signed[:, j]
        v["contrast"] = contrast
        v["contrast_fitted_density"] = contrast
        values.append(v)
    all_values = pd.concat(values, ignore_index=True)
    old = old.rename(
        columns={
            "estimate": "peace",
            "truth": "peace_truth",
            "estimated_density_estimate": "peace_fitted_density",
        }
    )
    frame = old[KEY + ["d", "peace", "peace_truth", "peace_fitted_density"]].merge(
        all_values, on=KEY + ["d"], validate="one_to_one"
    )
    truth = functionals(core.true_theta())
    truth_map = dict(zip(DEGREES, truth["signed_derivative"]))
    frame["signed_derivative_truth"] = frame.d.map(truth_map)
    frame["contrast_truth"] = truth["contrast"][0]
    frame.to_csv(
        OUT / "cubic_comparison_estimates.csv.gz", index=False, compression=COMPRESS
    )
    return frame


def wilson(p, n):
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def summary_stats(values, truth):
    values = np.asarray(values)
    error = values - truth
    n = len(values)
    rmse = np.sqrt(np.mean(error**2))
    within = np.mean(abs(error) <= 0.1 * abs(truth))
    lower, upper = wilson(within, n)
    return {
        "repetitions": n,
        "truth": truth,
        "mean": values.mean(),
        "median": np.median(values),
        "q05": np.quantile(values, 0.05),
        "q95": np.quantile(values, 0.95),
        "q99": np.quantile(values, 0.99),
        "bias": error.mean(),
        "sd": values.std(ddof=1),
        "rmse": rmse,
        "relative_rmse": rmse / abs(truth),
        "relative_bias": error.mean() / abs(truth),
        "median_absolute_relative_error": np.median(abs(error)) / abs(truth),
        "within_10_percent": within,
        "within_10_percent_mcse": np.sqrt(within * (1 - within) / n),
        "within_10_percent_wilson_lower": lower,
        "within_10_percent_wilson_upper": upper,
        "sign_reversal_frequency": np.mean(values * truth < 0),
        "mean_plugin_mcse": values.std(ddof=1) / np.sqrt(n),
    }


def summarize(quadratic, cubic, scalar):
    records = []
    group_keys = ["n", "rho", "method", "penalty", "d"]
    for scenario, frame in [("quadratic", quadratic), ("cubic", cubic)]:
        for key, group in frame.groupby(group_keys):
            base = {"scenario": scenario} | dict(zip(group_keys, key))
            for metric in METRICS:
                if metric == "contrast" and base["d"] != 0.5:
                    continue
                result = (
                    base
                    | {"metric": metric}
                    | summary_stats(group[metric], group[metric + "_truth"].iloc[0])
                )
                result["fitted_density_rmse"] = np.sqrt(
                    np.mean((group[metric + "_fitted_density"] - result["truth"]) ** 2)
                )
                result["density_shift_abs_q95"] = np.quantile(
                    abs(group[metric + "_fitted_density"] - group[metric]), 0.95
                )
                records.append(result)
    for (n, rho), group in scalar.groupby(["n", "rho"]):
        records.append(
            {
                "scenario": "quadratic",
                "n": n,
                "rho": rho,
                "method": "Direct scalar IV",
                "penalty": 0.0,
                "d": 0.5,
                "metric": "signed_derivative",
            }
            | summary_stats(group.estimate, 1.0)
        )
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "comparison_summary.csv", index=False)
    pairs = []
    q = quadratic[(quadratic.method == "IV degree 2") & (quadratic.d == 0.5)]
    c = cubic[(cubic.method == "IV degree 3") & (cubic.d == 0.5)]
    paired = q.merge(
        c,
        on=["repetition", "n", "rho", "d"],
        suffixes=("_quadratic", "_cubic"),
        validate="one_to_one",
    )
    for (n, rho), group in paired.groupby(["n", "rho"]):
        for metric in METRICS:
            qerr = abs(
                group[metric + "_quadratic"] - group[metric + "_truth_quadratic"]
            ) / abs(group[metric + "_truth_quadratic"])
            cerr = abs(group[metric + "_cubic"] - group[metric + "_truth_cubic"]) / abs(
                group[metric + "_truth_cubic"]
            )
            difference = (qerr <= 0.1).astype(float) - (cerr <= 0.1).astype(float)
            pairs.append(
                {
                    "n": n,
                    "rho": rho,
                    "metric": metric,
                    "quadratic_within_10_percent": np.mean(qerr <= 0.1),
                    "cubic_within_10_percent": np.mean(cerr <= 0.1),
                    "paired_accuracy_difference": difference.mean(),
                    "paired_difference_mcse": difference.std(ddof=1)
                    / np.sqrt(len(group)),
                    "quadratic_median_relative_error": np.median(qerr),
                    "cubic_median_relative_error": np.median(cerr),
                }
            )
    pd.DataFrame(pairs).to_csv(OUT / "paired_mechanism_comparison.csv", index=False)


def run():
    OUT.mkdir(exist_ok=True)
    assert CONFIG["seed"] == core.CONFIG["seed"]
    assert np.array_equal(DEGREES, core.DEGREES)
    assert CONFIG["sample_sizes"] == core.CONFIG["sample_sizes"]
    assert CONFIG["instrument_correlations"] == core.CONFIG["instrument_correlations"]
    before = protected_hashes()
    start = time.monotonic()
    (OUT / "frozen_config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    q = run_quadratic()
    c = augment_cubic()
    summarize(q["quadratic_estimates"], c, q["scalar_signed_estimates"])
    truths = []
    for scenario, theta in [("quadratic", true_theta()), ("cubic", core.true_theta())]:
        causal = functionals(theta)
        observed_theta = theta.copy()
        observed_theta[1] += 0.5
        observational = functionals(observed_theta)
        for i, d in enumerate(DEGREES):
            for metric in METRICS:
                truths.append(
                    {
                        "scenario": scenario,
                        "d": d,
                        "metric": metric,
                        "truth": causal[metric][i],
                        "population_OLS_target": observational[metric][i],
                    }
                )
    pd.DataFrame(truths).to_csv(OUT / "targets.csv", index=False)
    after = protected_hashes()
    metadata = {
        "config": CONFIG,
        "protected_before": before,
        "protected_after": after,
        "protected_files_unchanged": before == after,
        "runtime_seconds": time.monotonic() - start,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design_sha256": hashlib.sha256((ROOT / "DESIGN.md").read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256(
            (PARENT / "experiment.py").read_bytes()
        ).hexdigest(),
        "quadratic_datasets": CONFIG["repetitions"] * 18,
        "quadratic_response_fits": len(q["quadratic_coefficients"]),
        "cubic_response_fits_reused": len(c) // len(DEGREES),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert before == after, "Protected original study or manuscript changed"
    print(f"Completed in {metadata['runtime_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
