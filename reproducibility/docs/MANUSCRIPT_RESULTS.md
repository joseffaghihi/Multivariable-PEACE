# Manuscript result map

## Sodium

- Main table: experiments/sodium/results/simulation_summary.csv; baseline, estimated_gaussian density, d=0.5, r=1, sodium exposure; structural and spline methods at n=500, 2000 and 8000.
- Degree curves: oracle_curves.csv and joint_oracle.csv.
- Comparator targets/errors: comparators_summary.csv.
- Sensitivity: simulation_summary.csv by scenario; density_summary.csv for the density-only checks.

## Customer satisfaction

- Main table: experiments/customer_satisfaction/flexible_estimation/results/global_summary.csv; original scenario, n=5000, d=0.5; linear and additive methods. Fold estimates are first averaged within each dataset before Monte Carlo summaries.
- Conditional-support limitations: conditional_summary.csv, outcome_noise scenario; these are different targets from the global total response.
- Nonlinear-component selection: selected_global_summary.csv and selected_conditional_summary.csv.
- Sign-changing response: shape_challenge_summary.csv, shape_direction_summary.csv and shape_challenge_truth.json.
- Predictive attribution and causal comparators: the parent results/prediction_metrics.json, comparator_summary_mcse.csv and projection_truth.json.

## Separate matched-mechanism study

The study is not yet integrated into the manuscript. matched_baselines.csv contains the matched scalar summaries; oracle_scores.csv and oracle_surfaces.csv show the effects of d and r. calibration_results.csv retains paired simulation estimates. The weighted squared-derivative comparator is retained, and no exclusive advantage over an identically weighted absolute-derivative functional is claimed.
