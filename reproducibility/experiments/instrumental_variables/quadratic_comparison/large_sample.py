"""Additional precision check, fixed after the initial quadratic results."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import comparison as study

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CONFIG = json.loads((ROOT / "large_sample_config.json").read_text())


def run():
    before = study.protected_hashes()
    start = time.monotonic()
    truth = study.functionals(study.true_theta())
    rows, coefficient_rows, scalar_rows, diagnostic_rows = [], [], [], []
    (OUT / "frozen_large_sample_config.json").write_text(
        json.dumps(CONFIG, indent=2) + "\n"
    )
    for rep in range(CONFIG["repetitions"]):
        latent = study.core.generate_latent(max(CONFIG["sample_sizes"]), rep)
        for rho in CONFIG["instrument_correlations"]:
            wf, xf, yf = study.observed(latent, rho)
            for n in CONFIG["sample_sizes"]:
                w, x, y = wf[:n], xf[:n], yf[:n]
                p, q = study.core.basis(x, 2), study.core.basis(w, 2)
                a, b, gxx, gxy, smallest = study.core.centered_moments(p, q, y)
                penalty = CONFIG["primary_ridge_penalty"]
                fits = [
                    ("OLS degree 2", 0.0, np.linalg.solve(gxx, gxy)),
                    ("IV degree 2", 0.0, np.linalg.solve(a, b)),
                    (
                        "Ridge IV degree 2",
                        penalty,
                        np.linalg.solve(a.T @ a + penalty * np.eye(2), a.T @ b),
                    ),
                ]
                for method, penalty, nonconstant in fits:
                    theta = study.core.add_intercept(nonconstant, p, y)
                    key = {
                        "repetition": rep,
                        "n": n,
                        "rho": rho,
                        "method": method,
                        "penalty": penalty,
                    }
                    values = study.functionals(theta)
                    plugin = study.functionals(theta, x.mean(), x.std(ddof=0))
                    rows.extend(study.wide_rows(key, values, plugin, truth))
                    coefficient_rows.append(
                        key
                        | {f"theta_{j}": theta[j] if j < 3 else 0.0 for j in range(6)}
                    )
                wc, xc, yc = w - w.mean(), x - x.mean(), y - y.mean()
                scalar_rows.append(
                    {
                        "repetition": rep,
                        "n": n,
                        "rho": rho,
                        "estimate": np.dot(wc, yc) / np.dot(wc, xc),
                        "truth": 1.0,
                    }
                )
                diagnostic_rows.append(
                    {
                        "repetition": rep,
                        "n": n,
                        "rho": rho,
                        "first_stage_F": study.core.linear_first_stage(w, x),
                        "empirical_smallest_canonical": smallest,
                        "population_smallest_canonical": rho * rho,
                        "mode_signal_scale": n * rho**4,
                    }
                )
        if (rep + 1) % 50 == 0:
            print(
                f"Large-sample repetitions: {rep + 1}/{CONFIG['repetitions']}",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    scalars = pd.DataFrame(scalar_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    for name, frame in [
        ("large_sample_estimates", raw),
        ("large_sample_coefficients", pd.DataFrame(coefficient_rows)),
        ("large_sample_scalar", scalars),
        ("large_sample_diagnostics", diagnostics),
    ]:
        frame.to_csv(OUT / (name + ".csv.gz"), index=False, compression=study.COMPRESS)
    summary = []
    for key, g in raw.groupby(["n", "rho", "method", "penalty", "d"]):
        base = {"scenario": "quadratic"} | dict(
            zip(["n", "rho", "method", "penalty", "d"], key)
        )
        for metric in study.METRICS:
            if metric == "contrast" and base["d"] != 0.5:
                continue
            record = (
                base
                | {"metric": metric}
                | study.summary_stats(g[metric], g[metric + "_truth"].iloc[0])
            )
            record["fitted_density_rmse"] = np.sqrt(
                np.mean((g[metric + "_fitted_density"] - record["truth"]) ** 2)
            )
            summary.append(record)
    for (n, rho), g in scalars.groupby(["n", "rho"]):
        summary.append(
            {
                "scenario": "quadratic",
                "n": n,
                "rho": rho,
                "method": "Direct scalar IV",
                "penalty": 0.0,
                "d": 0.5,
                "metric": "signed_derivative",
            }
            | study.summary_stats(g.estimate, 1.0)
        )
    pd.DataFrame(summary).to_csv(OUT / "large_sample_summary.csv", index=False)
    diagnostics.groupby(["n", "rho"]).median(numeric_only=True).reset_index().to_csv(
        OUT / "large_sample_diagnostic_summary.csv", index=False
    )
    after = study.protected_hashes()
    metadata = {
        "config": CONFIG,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design_sha256": hashlib.sha256(
            (ROOT / "LARGE_SAMPLE_DESIGN.md").read_bytes()
        ).hexdigest(),
        "comparison_source_sha256": hashlib.sha256(
            (ROOT / "comparison.py").read_bytes()
        ).hexdigest(),
        "protected_before": before,
        "protected_after": after,
        "protected_files_unchanged": before == after,
        "runtime_seconds": time.monotonic() - start,
        "datasets": len(scalars),
        "response_fits": len(coefficient_rows),
    }
    (OUT / "large_sample_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    assert before == after
    print(
        f"Large-sample check completed in {metadata['runtime_seconds']:.1f} seconds",
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
