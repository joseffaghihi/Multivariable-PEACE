# Multivariable PEACE

Research code for **Probabilistic Easy Variational Causal Effect**, by Usef Faghihi and Amir Saki.

## Revised simulation studies — November 2026

The [reproducibility package](reproducibility/README.md) contains the revised sodium-intake and customer-satisfaction experiments, together with the separate matched-mechanism study. It provides readable Python sources, fixed seeds, the recorded dependency environment, reference outputs, numerical checks and commands for fresh runs.

    cd reproducibility
    python3.13 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-lock.txt
    python tools/check_results.py
    python reproduce.py --mode smoke

Use `--mode validate`, `--mode figures`, or `--mode full` for numerical validation, report reconstruction, or complete Monte Carlo regeneration. See the [full instructions](reproducibility/docs/REPRODUCIBILITY.md) and [manuscript result map](reproducibility/docs/MANUSCRIPT_RESULTS.md).

The original two notebooks below are retained unchanged for provenance. Their exploratory cells are not the supported pipeline for the revised manuscript.
