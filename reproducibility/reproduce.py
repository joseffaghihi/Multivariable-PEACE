"""Run the PEACE studies in isolated working directories."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
STUDIES = {
    "sodium": "sodium",
    "customer": "customer_satisfaction",
    "matched": "matched_mechanisms",
}


def environment(work):
    """Limit numerical threads and locate the wheel-provided OpenMP runtime."""
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    cache = work / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache / "matplotlib")
    env["XDG_CACHE_HOME"] = str(cache)
    if sys.platform == "darwin":
        spec = importlib.util.find_spec("sklearn")
        if spec and spec.origin:
            library = Path(spec.origin).parent / ".dylibs"
            if (library / "libomp.dylib").exists():
                env["DYLD_LIBRARY_PATH"] = str(library) + (
                    ":" + env["DYLD_LIBRARY_PATH"]
                    if env.get("DYLD_LIBRARY_PATH")
                    else ""
                )
    return env


def stages(study, mode):
    if mode == "smoke":
        return [(".", ["tools/smoke.py", study])]
    if study == "sodium":
        checks = ["experiment.py --stage oracle", "validate.py", "validate_kde.py"]
        figures = [
            "experiment.py --stage plots",
            "interpretation_plots.py",
            "make_report.py",
        ]
        full = [
            "experiment.py --stage core",
            "experiment.py --stage comparators --reps 50",
            "experiment.py --stage uncertainty --reps 100 --boots 200",
            "experiment.py --stage density --reps 30",
            "extra_analysis.py",
            "separability_analysis.py",
        ]
        commands = (
            checks
            if mode == "validate"
            else figures
            if mode == "figures"
            else full + checks + figures
        )
        return [("experiments/sodium", x.split()) for x in commands]
    if study == "customer":
        main = "experiments/customer_satisfaction"
        flex = main + "/flexible_estimation"
        learned = main + "/learned_mechanisms"
        checks = [
            (main, ["extra_analysis.py", "validate"]),
            (flex, ["study.py", "validate"]),
            (flex, ["audit.py"]),
            (flex, ["selected_audit.py"]),
            (learned, ["study.py", "validate"]),
        ]
        figures = [
            (main, ["interpretation.py"]),
            (main, ["experiment.py", "sensitivity"]),
            (learned, ["report.py"]),
            (flex, ["report.py"]),
            (main, ["make_report.py"]),
        ]
        full = [
            (main, ["experiment.py", part])
            for part in ("oracle", "observational", "comparators", "prediction")
        ]
        full += [
            (main, ["extra_analysis.py", part])
            for part in ("joint", "bootstrap", "robustness", "null")
        ]
        full += [
            (learned, ["study.py", part, "--reps", "100"])
            for part in ("simulate", "stress", "uncertainty", "joint")
        ]
        full += [
            (flex, ["study.py", "run", "--reps", "40"]),
            (flex, ["selected_study.py"]),
            (flex, ["joint.py"]),
            (flex, ["shape_challenge.py"]),
        ]
        return (
            checks
            if mode == "validate"
            else figures
            if mode == "figures"
            else full + checks + figures
        )
    checks = [
        ("experiments/matched_mechanisms", x.split())
        for x in ("experiment.py validate", "density_audit.py", "independent_audit.py")
    ]
    figures = [("experiments/matched_mechanisms", ["report.py"])]
    if mode == "validate":
        return checks
    if mode == "figures":
        return figures
    return (
        [
            ("experiments/matched_mechanisms", ["experiment.py", "validate"]),
            (
                "experiments/matched_mechanisms",
                ["experiment.py", "run", "--reps", "40"],
            ),
        ]
        + checks[1:]
        + figures
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["all", *STUDIES], default="all")
    parser.add_argument(
        "--mode", choices=["smoke", "validate", "figures", "full"], default="smoke"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New working directory; existing directories are never overwritten.",
    )
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = (args.output or ROOT / "runs" / f"{args.mode}-{stamp}").resolve()
    if work.exists():
        parser.error(f"Output directory already exists: {work}")
    if ROOT == work or ROOT.is_relative_to(work):
        parser.error("The output directory must not contain the repository.")
    work.mkdir(parents=True)
    selected = list(STUDIES) if args.study == "all" else [args.study]
    for study in selected:

        def ignore(directory, names):
            omitted = {"__pycache__"}
            if args.mode == "full":
                omitted |= {"results", "figures"}
            return set(names) & omitted

        target = work / "experiments" / STUDIES[study]
        shutil.copytree(ROOT / "experiments" / STUDIES[study], target, ignore=ignore)
        for folder in [
            target,
            *[p for p in target.iterdir() if p.is_dir() and any(p.glob("*.py"))],
        ]:
            (folder / "results").mkdir(exist_ok=True)
            (folder / "figures").mkdir(exist_ok=True)
    shutil.copytree(ROOT / "tools", work / "tools")
    shutil.copytree(ROOT / "docs", work / "docs")
    if args.mode != "full":
        with zipfile.ZipFile(ROOT / "reference_outputs.zip") as archive:
            prefixes = tuple("experiments/" + STUDIES[name] + "/" for name in selected)
            for item in archive.infolist():
                if item.filename.startswith(prefixes):
                    if ".." in Path(item.filename).parts:
                        raise ValueError("Invalid reference archive path")
                    archive.extract(item, work)
    notebooks = (
        list(ROOT.glob("*.ipynb"))
        or list(ROOT.parent.glob("*.ipynb"))
        or list((ROOT.parent / "archive").glob("*.ipynb"))
    )
    for notebook in notebooks:
        shutil.copy2(notebook, work / notebook.name)
    logs = work / "logs"
    logs.mkdir()
    env = environment(work)
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mode": args.mode,
        "studies": selected,
        "stages": [],
    }
    for study in selected:
        for index, (relative, command) in enumerate(stages(study, args.mode)):
            cwd = work / relative
            logfile = logs / f"{study}-{index:02d}.log"
            start = time.monotonic()
            print(f"[{study}] {' '.join(command)}", flush=True)
            with logfile.open("w") as handle:
                result = subprocess.run(
                    [sys.executable, *command],
                    cwd=cwd,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            record["stages"].append(
                {
                    "study": study,
                    "command": command,
                    "seconds": round(time.monotonic() - start, 3),
                    "returncode": result.returncode,
                }
            )
            (work / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
            if result.returncode:
                print(logfile.read_text()[-5000:], file=sys.stderr)
                raise SystemExit(f"Stage failed. Complete log: {logfile}")
    print(f"Completed. Results and logs: {work}")


if __name__ == "__main__":
    main()
