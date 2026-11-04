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
    with zipfile.ZipFile(ROOT / "reference_outputs.zip") as archive:
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
    print(
        f"Verified {len(manifest)} reference files and all selected manuscript values."
    )


if __name__ == "__main__":
    main()
