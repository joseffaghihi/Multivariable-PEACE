# Linear and flexible observational estimation

This is the 40-dataset matched comparison used for the revised customer experiment. The causal graph, valid unaffected adjustment sets, analysis domains and model-complexity choices are supplied. Exact logistic and reciprocal response transformations are not supplied to the learners.

Run the customer pipeline through reproduce.py at the repository root. For individual phases, run study.py validate, study.py run --reps 40, selected_study.py, joint.py, shape_challenge.py, audit.py, selected_audit.py and report.py in that order inside an isolated working copy.

Two folds within one dataset are averaged before calculating replicate SD, bias or RMSE. The selected additive model is an exploratory extension. Global discount/ad-spend estimates under additive extrapolation are distinguished from supported conditional-response estimates. Generic interaction-capable estimates are not reported outside support. Reported bugs is not fitted because its parent-adjusted conditional treatment variation is deterministic.
