# PEACE: reproducible simulation studies

Code and numerical results for **Probabilistic Easy Variational Causal Effect**, by Usef Faghihi and Amir Saki.

The November 2026 revision adds three self-contained synthetic studies. The two original notebooks are preserved unchanged in [archive/](../archive/README.md) as historical source material. Start with the scripts below for the revised results.

| Study | Question | Location |
| --- | --- | --- |
| Sodium intake and blood pressure | How do treatment/profile weights, overlap, density specification and hidden confounding affect a specified causal summary? | [Sodium](experiments/sodium/README.md) |
| Customer satisfaction | How do total responses differ from predictive attribution, and how accurately can their local variation be estimated? | [Customer satisfaction](experiments/customer_satisfaction/README.md) |
| Matched causal mechanisms | Can mechanisms with matched conventional scalar summaries have different probability-weighted local response variation? | [Matched mechanisms](experiments/matched_mechanisms/README.md) |

The first two studies support the revised manuscript. The third is a separate supporting study. All observations are simulated; no confidential or real participant data are included.

## Installation

The recorded environment is Python 3.13 on macOS ARM64. The exact numerical dependencies are pinned in requirements-lock.txt. Use a fresh virtual environment:

    python3.13 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-lock.txt

On Windows, activate with `.venv\Scripts\activate`. Availability of binary wheels and the OpenMP runtime can depend on platform. The runner sets one numerical thread and, on macOS, locates the OpenMP library supplied with scikit-learn before launching XGBoost.

## Reproduce

Run these commands from the repository root:

    python tools/check_results.py
    python reproduce.py --mode smoke
    python reproduce.py --mode validate
    python reproduce.py --mode figures
    python reproduce.py --mode full

- **check_results** verifies reference checksums and the selected values reported in the manuscript. It uses the Python standard library.
- **smoke** runs small seeded fits, analytic/numerical derivative checks, comparator imports/fits, and a one-replicate matched-mechanism calibration. It does not reproduce the full Monte Carlo precision tables.
- **validate** runs the numerical and integration checks using a working copy of the reference results.
- **figures** rebuilds reports and supplementary plots using the supplied result/figure snapshot; it does not refit the simulations. Full mode regenerates the complete analyses and their figures.
- **full** regenerates the simulations, checks, summaries and plots from scratch using the published seeds and replication counts. The bootstrap and flexible-estimation stages are substantial CPU workloads; expect this to take much longer than a smoke run.

Use `--study sodium`, `--study customer`, or `--study matched` to select one study. Supply `--output /path/to/new-directory` to choose a working directory. Existing output directories are rejected. By default each run creates a timestamped directory under runs/, with complete stage logs and execution.json. The reference_outputs.zip snapshot is never overwritten by the runner.

See [replication details](docs/REPRODUCIBILITY.md) for the distinction between reference-result verification, fresh simulation and platform-sensitive floating-point agreement.

## Interpretation

PEACE summarizes magnitudes of local changes in a specified interventional response under declared reference weights. It is not interchangeable with a signed ATE, a projection slope, a prediction score or SHAP. Comparator errors are assessed against each comparator's own target. Neither weighting nor flexible regression removes missing identification assumptions or unsupported interventions. Unfavorable estimation and uncertainty results are retained.

The studies use native treatment units and explicit domains. Larger PEACE is greater local responsiveness under the selected priorities; it is not automatically a better treatment, predictor or policy. [Provenance and scope](docs/PROVENANCE.md) record the simulator source, design choices and publication cleanup.
