# Replication details

## Reference snapshot

The CSV/JSON results and reference figure assets are stored in reference_outputs.zip, with their original experiments/*/results and figures paths. The runner extracts only the selected studies into its working directory. These are the numerical snapshot used for reporting. reference_manifest.json records their SHA-256 checksums. tools/check_results.py also checks the selected sodium and customer table entries and the matched-mechanism natural-weight identity.

A successful snapshot check establishes file integrity and consistency with those reported values. It is separate from regenerating the results from random draws.

## Fresh runs

reproduce.py copies the requested study into a new working directory and launches each phase in a fresh process. The full mode does not copy the reference results or figures into that working tree. Seeds, sample sizes, domains, quadrature settings and replication counts are declared in the study sources. Every stage writes a separate log. A nonzero stage exit stops the run.

Sodium uses 100 datasets per primary sample-size/scenario case, 50 comparator datasets per scenario, and 100 datasets with 200 bootstrap draws for the uncertainty experiment. Customer includes the original simulator benchmark, the stronger-assumption learned-mechanism benchmark, and the 40-dataset cross-fitted flexible estimation study. Matched mechanisms uses 40 paired calibration datasets at each of three sample sizes and a 20,000-observation natural reference sample.

The customer full run includes both follow-ups. The flexible estimator does not receive the true logistic or reciprocal response features. The learned-mechanism benchmark does receive the graph and functional families and learns their unknown parameters. These information sets must not be conflated.

## Numerical agreement

The lock file records the environment used for the supplied results. One-thread execution reduces variation from parallel reductions. Identical seeds do not promise byte-for-byte agreement across BLAS, architecture, solver or library changes. Compare numerical values with the stated integration tolerances and Monte Carlo precision; image metadata and font rendering need not match.

Reports are generated only after the expected full replication counts are present. A smoke run deliberately uses smaller calculations and cannot be passed off as a full replication. The run's execution.json records its mode.

## Statistical interpretation

The simulator supplies ground-truth intervention responses for scoring. Causal adjustment sets and support assumptions are declared inputs to observational estimators. The studies do not discover a causal graph from predictive accuracy. Prediction-based smoothing selection does not establish derivative consistency or valid confidence intervals. Reported replicate SD and Monte Carlo SE are distinct from confidence intervals for an individual causal estimate.

Density-only checks isolate reference-weight estimation; they are not an end-to-end guarantee for the full response estimator. Hidden-confounding scenarios deliberately violate the maintained identification assumptions. They illustrate sensitivity shared by observational causal procedures.

## Files and reports

Each working study contains results/, figures/, and its code. Figures mode writes REPORT.md and REPORT.html there. The customer follow-ups generate their own reports in flexible_estimation/ and learned_mechanisms/. Manuscript-facing figure ingredients are identified in docs/MANUSCRIPT_RESULTS.md.
