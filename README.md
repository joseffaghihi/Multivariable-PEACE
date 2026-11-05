# PEACE

Code and reproducible experiments accompanying **Probabilistic Easy Variational Causal Effect**, by Usef Faghihi and Amir Saki.

## Experiments

- [Sodium intake and blood pressure](reproducibility/experiments/sodium/README.md)
- [Customer satisfaction](reproducibility/experiments/customer_satisfaction/README.md)
- [Matched causal mechanisms](reproducibility/experiments/matched_mechanisms/README.md)
- [Instrumental variables: a causal contrast and Mean PEACE](reproducibility/experiments/instrumental_variables/README.md)

## Reproducing the results

    cd reproducibility
    python3.13 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-lock.txt
    python tools/check_results.py
    python reproduce.py --mode smoke

Use `--mode validate` for numerical checks, `--mode figures` to rebuild reports and plots from the reference results, or `--mode full` to run the complete simulations.

See the [reproduction instructions](reproducibility/docs/REPRODUCIBILITY.md) and [manuscript result map](reproducibility/docs/MANUSCRIPT_RESULTS.md) for settings, outputs and computational requirements.
