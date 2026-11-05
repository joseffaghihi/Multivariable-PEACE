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

## Matched-mechanism study

The matched-mechanism example uses matched_baselines.csv for the matched scalar summaries; oracle_scores.csv and oracle_surfaces.csv show the effects of d and r. calibration_results.csv retains paired simulation estimates. The weighted squared-derivative comparator is retained, and no exclusive advantage over an identically weighted absolute-derivative functional is claimed.

## Instrumental-variable estimation

The paper uses the quadratic response in `experiments/instrumental_variables/quadratic_comparison/`, not the supporting parent's cubic response. Select `scenario=quadratic`, `method=IV degree 2`, `penalty=0`, and degree `d=0.5` where present. The treatment domain is `(-2.5,2.5)`, with normalized restricted natural treatment weights, and there are no remaining covariates.

The table is reconstructed from `results/paired_recovery_summary.csv`. Every row has 500 repetitions. Accuracy is absolute relative error at most 0.1 of each summary's own truth.

| rho | n | Contrast accurate | Mean PEACE accurate | Both / contrast accurate |
| --- | ---: | ---: | ---: | --- |
| 0.05 | 1000 | 60 | 45 | 23/60 (38.3%) |
| 0.80 | 4000 | 500 | 500 | 500/500 (100.0%) |
| 0.20 | 4000 | 190 | 146 | 138/190 (72.6%) |
| 0.20 | 16000 | 348 | 263 | 262/348 (75.3%) |
| 0.20 | 64000 | 487 | 436 | 436/487 (89.5%) |
| 0.20 | 256000 | 500 | 499 | 499/500 (99.8%) |

- Target values: `results/targets.csv`, quadratic scenario. The contrast is 1.75 and Mean PEACE at d=0.5 is 1.41838949....
- Weak-cell median Mean PEACE: `results/comparison_summary.csv`, rho=0.05, n=1000, metric=peace, the filters above: 2.0729 rounded.
- Weak-cell first-stage F: `results/quadratic_response_summary.csv`, rho=0.05, n=1000, method and penalty above: 2.68 rounded.
- Larger-sample median estimates: `results/large_sample_summary.csv`, rho=0.2, n=64000, metrics contrast and peace: 1.7503 and 1.4169 rounded.
- Every paired fitted response and all four accuracy states: `results/paired_recovery.csv.gz` and `paired_recovery_summary.csv`.

The parent cubic/linear simulations, all instrument strengths including zero, polynomial-class and fixed-penalty sensitivity, fitted-density variants and unsuccessful cases remain available. Conditional percentages are simulation recovery diagnostics using known truth, not confidence intervals or rankings of scientific estimands.
