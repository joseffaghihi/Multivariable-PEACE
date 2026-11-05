# Instrumental variables: a causal contrast and Mean PEACE

This fourth experiment accompanies **Probabilistic Easy Variational Causal Effect**. The [quadratic comparison](quadratic_comparison/README.md) supplies the manuscript's IV example and paired-recovery table. The parent cubic study and linear weak-instrument-robust inference check are included as supporting analyses and as the explicit local dependency of that comparison. All data are synthetic. No earlier project folders, manuscript files, credentials or external data are needed.

## Reproduce

Install the [package dependencies](../../README.md). From `reproducibility/`:

```sh
python tools/check_results.py
python reproduce.py --study iv --mode smoke
python reproduce.py --study iv --mode validate
python reproduce.py --study iv --mode figures
python reproduce.py --study iv --mode full
```

Each command creates a new directory under `runs/`; `--output /path/to/new-directory` selects another. Reference archives are never overwritten.

- **smoke** checks both target values against independent adaptive integration, fits fresh seeded quadratic IV samples and checks moment equations and numerical integration. It does not reproduce Monte Carlo frequencies.
- **validate** independently audits the supplied parent and quadratic results and reconstructs all paired-recovery counts.
- **figures** rebuilds both reports and their figures from complete reference results. Open `experiments/instrumental_variables/quadratic_comparison/REPORT.html` inside the run directory for the manuscript study.
- **full** starts without reference results and runs the parent simulation/audit/report, quadratic simulation, large-sample follow-up and their audit/report in dependency order. All published strengths, penalties, response classes, seeds and repetition counts are retained.

`iv_reference_outputs_*.zip` at the package root contains the complete IV CSV/JSON results, reports and figure assets. Its entries are covered by `reference_manifest.json`. The earlier three studies' archive is unchanged.

## Model and interpretation

For independent standard normal disturbances W,U,R,E, the quadratic model is `X=rho*W+0.5*U+sqrt(0.75-rho**2)*R` and `Y=X+0.75*(X**2-1)+U+0.5*E`. Only W,X,Y are observed. The interventional mean is `h(x)=x+0.75*(x**2-1)`. The contrast from 0 to 1 is **1.75**. Mean PEACE at d=1/2 is **1.41838949...** on (-2.5,2.5), under normalized restricted natural treatment weights.

Each cell retains all 500 repetitions. Accuracy means relative error at most 10% of the corresponding target's own truth. All four paired outcomes are retained: both accurate, contrast only, Mean PEACE only, and neither. At rho=0.2/n=64000 the conditional count is 436/487 (89.5%); at n=256000 it is 499/500 (99.8%). These are simulator diagnostics using known truth, not confidence intervals or real-data accuracy tests.

The declared polynomial classes and fixed penalties are modeling restrictions. Weak-information failures, extreme IV estimates, density sensitivity and the cubic stress analysis remain included. Zero strength is an explicitly unidentified endpoint. No favorable cases are removed. There are no covariates and no degree r.

## Files and provenance

`quadratic_comparison/` contains the manuscript experiment, design records, mathematical argument, configurations, simulations, independent audit, paired-recovery analysis and report generator. The parent `experiment.py`, `audit.py`, `report.py`, designs and mathematical argument supply the supporting cubic/linear study. The [manuscript result map](../../docs/MANUSCRIPT_RESULTS.md) gives exact output files and filters.

The parent study, quadratic comparison and later large-sample check are sequential exploratory analyses, not a single preregistered design. Publication edits update reproduction paths, status prose and execution wrappers. Numerical mechanisms, estimator equations, seeds and scoring rules are preserved. Manuscript hash checks apply only when manuscript sources are present; standalone numerical audits do not require them.
