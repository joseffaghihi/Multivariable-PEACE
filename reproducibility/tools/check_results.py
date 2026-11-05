"""Verify the reference snapshot and selected manuscript values without refitting."""

import csv
import hashlib
import json
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resource(path):
    local = ROOT / path
    if local.exists():
        return local.read_bytes()
    if path.startswith("experiments/instrumental_variables/"):
        index = json.loads((ROOT / "iv_reference_index.json").read_text())
        archive_name = index[path]
    else:
        archive_name = "reference_outputs.zip"
    with zipfile.ZipFile(ROOT / archive_name) as archive:
        return archive.read(path)


def rows(path):
    return list(csv.DictReader(io.StringIO(resource(path).decode("utf-8"))))


def close(actual, expected, tolerance=5e-6):
    if abs(float(actual) - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, found {actual}")


def main():
    manifest = json.loads((ROOT / "reference_manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256(resource(name)).hexdigest() != expected:
            raise AssertionError(f"Reference checksum mismatch: {name}")
    sodium = rows("experiments/sodium/results/simulation_summary.csv")
    expected = {
        500: (4.01683, 0.51275),
        2000: (3.85419, 0.24188),
        8000: (3.84298, 0.10066),
    }
    for n, (mean, rmse) in expected.items():
        selected = [
            r
            for r in sodium
            if r["scenario"] == "baseline"
            and int(r["n"]) == n
            and r["method"] == "spline"
            and r["density"] == "estimated_gaussian"
            and r["exposure"] == "sodium"
            and float(r["d"]) == 0.5
            and float(r["r"]) == 1
        ]
        assert len(selected) == 1
        close(selected[0]["mean"], mean)
        close(selected[0]["rmse"], rmse)
    customer = rows(
        "experiments/customer_satisfaction/flexible_estimation/results/global_summary.csv"
    )
    expected = {
        "Sales calls": (1.01619, 0.01600),
        "Product need": (5.16202, 0.01080),
        "Last upgrade": (0.72699, 0.01764),
        "Monthly usage": (3.00151, 0.00685),
    }
    for variable, (mean, rmse) in expected.items():
        selected = [
            r
            for r in customer
            if r["scenario"] == "original"
            and int(r["n"]) == 5000
            and r["method"] == "additive"
            and r["variable"] == variable
            and float(r["d"]) == 0.5
        ]
        assert len(selected) == 1
        close(selected[0]["mean"], mean)
        close(selected[0]["rmse"], rmse)
    matched = rows("experiments/matched_mechanisms/results/matched_baselines.csv")
    assert len(matched) == 2
    for row in matched:
        close(row["natural_absolute_derivative"], 15.6, 1e-12)
        close(row["signed_ADE"], 0, 1e-12)
    iv_root = "experiments/instrumental_variables/quadratic_comparison/results/"
    recovery = rows(iv_root + "paired_recovery_summary.csv")
    expected_iv = {
        (0.05, 1000): (60, 45, 23),
        (0.8, 4000): (500, 500, 500),
        (0.2, 4000): (190, 146, 138),
        (0.2, 16000): (348, 263, 262),
        (0.2, 64000): (487, 436, 436),
        (0.2, 256000): (500, 499, 499),
    }
    for (rho, n), (contrast, peace, both) in expected_iv.items():
        selected = [
            row
            for row in recovery
            if row["scenario"] == "quadratic"
            and row["method"] == "IV degree 2"
            and float(row["penalty"]) == 0
            and float(row["rho"]) == rho
            and int(row["n"]) == n
        ]
        assert len(selected) == 1
        row = selected[0]
        assert int(row["repetitions"]) == 500
        assert int(row["contrast_accurate_count"]) == contrast
        assert int(row["peace_accurate_count"]) == peace
        assert int(row["both_accurate_count"]) == both
        close(row["peace_accuracy_given_contrast_accuracy"], both / contrast, 1e-14)
        close(row["contrast_truth"], 1.75, 1e-13)
        close(row["peace_truth"], 1.41838949, 5e-9)
    weak = [
        row
        for row in rows(iv_root + "comparison_summary.csv")
        if row["scenario"] == "quadratic"
        and row["method"] == "IV degree 2"
        and float(row["rho"]) == 0.05
        and int(row["n"]) == 1000
        and float(row["d"]) == 0.5
        and row["metric"] == "peace"
    ]
    assert len(weak) == 1
    close(weak[0]["median"], 2.0729, 5e-5)
    first_stage = [
        row
        for row in rows(iv_root + "quadratic_response_summary.csv")
        if row["method"] == "IV degree 2"
        and float(row["penalty"]) == 0
        and float(row["rho"]) == 0.05
        and int(row["n"]) == 1000
    ]
    assert len(first_stage) == 1
    close(first_stage[0]["median_first_stage_F"], 2.68, 5e-3)
    large = rows(iv_root + "large_sample_summary.csv")
    for metric, expected_median in [("contrast", 1.7503), ("peace", 1.4169)]:
        selected = [
            row
            for row in large
            if row["scenario"] == "quadratic"
            and row["method"] == "IV degree 2"
            and float(row["penalty"]) == 0
            and float(row["rho"]) == 0.2
            and int(row["n"]) == 64000
            and float(row["d"]) == 0.5
            and row["metric"] == metric
        ]
        assert len(selected) == 1
        close(selected[0]["median"], expected_median, 5e-5)
    print(
        f"Verified {len(manifest)} reference files and all selected manuscript values."
    )


if __name__ == "__main__":
    main()
