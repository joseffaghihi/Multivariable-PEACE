# Estimation with supplied mechanism families

This benchmark supplies the graph and functional families, learns their unknown required coefficients and noise scales, and propagates the selected interventions through the fitted mechanisms. It is a stronger-information reference alongside the flexible estimation study.

The customer full pipeline runs simulate, stress, uncertainty, joint and validate stages of study.py with their recorded default counts, followed by report.py. Use reproduce.py from the repository root to run in an isolated output directory.

Graph-implied zeros are assumed exclusions, not null effects discovered from data. Global effects involving moving conditional supports require the stated parametric restrictions. Ordinary magnitude bootstrap intervals are not asserted to be valid for arbitrary flexible estimators or at all causal nulls.
