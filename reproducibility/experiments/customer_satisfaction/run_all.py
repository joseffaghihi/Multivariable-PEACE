import tempfile

"""Rebuild all final analyses; initial audit files are never overwritten."""
from pathlib import Path
import os, sys, subprocess, importlib.util

root = Path(__file__).resolve().parent
env = os.environ.copy()
env["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "peace-customer-matplotlib")
env["XDG_CACHE_HOME"] = os.path.join(tempfile.gettempdir(), "peace-customer-cache")
for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    env[key] = "1"
spec = importlib.util.find_spec("sklearn")
if spec and spec.origin:
    omp = Path(spec.origin).parent / ".dylibs"
    if (omp / "libomp.dylib").exists():
        env["DYLD_LIBRARY_PATH"] = str(omp) + (
            ":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else ""
        )
phases = [
    ("experiment.py", "oracle"),
    ("experiment.py", "observational"),
    ("experiment.py", "comparators"),
    ("experiment.py", "prediction"),
    ("extra_analysis.py", "joint"),
    ("extra_analysis.py", "bootstrap"),
    ("extra_analysis.py", "robustness"),
    ("extra_analysis.py", "null"),
    ("extra_analysis.py", "validate"),
    ("interpretation.py", None),
    ("experiment.py", "sensitivity"),
    ("make_report.py", None),
]
for file, part in phases:
    print("Running", file, part or "", flush=True)
    subprocess.run(
        [sys.executable, str(root / file)] + ([part] if part else []),
        cwd=root,
        env=env,
        check=True,
    )
