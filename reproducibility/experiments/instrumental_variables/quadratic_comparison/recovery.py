"""Paired accuracy of a finite intervention contrast and Mean PEACE.

Read all saved fits. The conditional frequencies are Monte Carlo diagnostics,
not a test of accuracy that can be evaluated from real data without truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import comparison as study

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
GROUP = ["scenario", "n", "rho", "method", "penalty"]


def analyze():
    columns = study.KEY + ["d", "peace", "peace_truth", "contrast", "contrast_truth"]
    frames = []
    for scenario, filename in [
        ("quadratic", "quadratic_estimates.csv.gz"),
        ("quadratic", "large_sample_estimates.csv.gz"),
        ("cubic", "cubic_comparison_estimates.csv.gz"),
    ]:
        frame = pd.read_csv(OUT / filename, usecols=columns)
        frame = frame[frame.d == 0.5].copy()
        frame["scenario"] = scenario
        frame["peace_relative_error"] = (frame.peace - frame.peace_truth) / abs(
            frame.peace_truth
        )
        frame["contrast_relative_error"] = (
            frame.contrast - frame.contrast_truth
        ) / abs(frame.contrast_truth)
        frame["peace_accurate"] = abs(frame.peace_relative_error) <= 0.1
        frame["contrast_accurate"] = abs(frame.contrast_relative_error) <= 0.1
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    assert not raw.duplicated(GROUP + ["repetition"]).any()
    raw.to_csv(OUT / "paired_recovery.csv.gz", index=False, compression=study.COMPRESS)
    records = []
    for key, g in raw.groupby(GROUP):
        p, c = g.peace_accurate, g.contrast_accurate
        n = len(g)
        both = int((p & c).sum())
        p_count = int(p.sum())
        c_count = int(c.sum())
        p_direct = abs(g.peace - g.peace_truth) <= 0.1 * abs(g.peace_truth)
        c_direct = abs(g.contrast - g.contrast_truth) <= 0.1 * abs(g.contrast_truth)
        assert np.array_equal(p, p_direct) and np.array_equal(c, c_direct)
        assert both == int(np.count_nonzero(p_direct.to_numpy() & c_direct.to_numpy()))
        conditional = both / c_count if c_count else np.nan
        lo, hi = study.wilson(conditional, c_count) if c_count else (np.nan, np.nan)
        records.append(
            dict(
                zip(GROUP, key),
                repetitions=n,
                peace_truth=g.peace_truth.iloc[0],
                contrast_truth=g.contrast_truth.iloc[0],
                peace_accurate_count=p_count,
                contrast_accurate_count=c_count,
                both_accurate_count=both,
                contrast_only_count=int((c & ~p).sum()),
                peace_only_count=int((p & ~c).sum()),
                neither_accurate_count=int((~p & ~c).sum()),
                peace_accuracy=p_count / n,
                contrast_accuracy=c_count / n,
                peace_accuracy_given_contrast_accuracy=conditional,
                conditional_wilson_lower=lo,
                conditional_wilson_upper=hi,
            )
        )
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "paired_recovery_summary.csv", index=False)
    # Independently verify every count and the link with existing marginal tables.
    assert len(raw) == 135000 and (raw.groupby(GROUP).size() == 500).all()
    for row in summary.itertuples():
        assert (
            row.both_accurate_count + row.contrast_only_count
            == row.contrast_accurate_count
        )
        assert (
            row.both_accurate_count + row.peace_only_count == row.peace_accurate_count
        )
        assert (
            row.both_accurate_count
            + row.contrast_only_count
            + row.peace_only_count
            + row.neither_accurate_count
            == 500
        )
    old = pd.concat(
        [
            pd.read_csv(OUT / "comparison_summary.csv"),
            pd.read_csv(OUT / "large_sample_summary.csv"),
        ]
    )
    old = old[(old.d == 0.5) & old.metric.isin(["peace", "contrast"])]
    for metric, col in [("peace", "peace_accuracy"), ("contrast", "contrast_accuracy")]:
        checked = summary.merge(
            old[old.metric == metric], on=GROUP, validate="one_to_one"
        )
        assert len(checked) == len(summary)
        assert np.allclose(checked[col], checked.within_10_percent, rtol=0, atol=1e-14)
    protected = json.loads((OUT / "metadata.json").read_text())["protected_before"]
    assert protected == study.protected_hashes()
    audit = {
        "all_checks_passed": True,
        "paired_response_fits": len(raw),
        "simulation_cells": len(summary),
        "all_saved_fits_retained": True,
        "all_four_outcome_counts_checked": True,
        "marginal_rates_equal_original_simulation_summaries": True,
        "zero_conditional_denominator_is_missing_not_zero": True,
        "manuscript_and_original_cubic_study_unchanged": True,
        "scope": "Descriptive conditional recovery frequency over simulated repeated datasets with known truth. No inference that one accurate finite contrast guarantees accurate Mean PEACE.",
    }
    (OUT / "paired_recovery_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return raw, summary


if __name__ == "__main__":
    _, summary = analyze()
    print(
        summary[
            (summary.scenario == "quadratic")
            & (summary.rho == 0.2)
            & (summary.method == "IV degree 2")
        ].to_string(index=False)
    )
