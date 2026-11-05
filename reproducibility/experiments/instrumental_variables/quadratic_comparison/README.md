# Quadratic IV comparison used in the paper

This study estimates a finite intervention contrast and Mean PEACE from the same fitted causal mean. Its required parent numerical module and supporting cubic/linear analyses are included in the directory above. Follow the [IV reproduction instructions](../README.md) to run it from a clean download.

From `reproducibility/`:

```sh
python reproduce.py --study iv --mode validate
python reproduce.py --study iv --mode figures
python reproduce.py --study iv --mode full
```

After a run, open this subdirectory's `REPORT.html` for the full report and six figures, `REPORT.md` for Markdown, or `COMPACT_SUMMARY.md` for the short summary. [Mathematical_argument.md](Mathematical_argument.md), `DESIGN.md` and `LARGE_SAMPLE_DESIGN.md` retain the arguments and sequential design records.

The interventional mean is `h(x)=x+0.75*(x**2-1)`. The 0-to-1 contrast is 1.75; Mean PEACE at d=1/2 on (-2.5,2.5) is 1.41838949... under normalized restricted natural weights. Each cell retains all 500 repetitions and uses a 10% relative-error threshold against each target's own truth.

The main run has 9000 datasets and 63000 response fits. The large-sample follow-up adds 3000 datasets and 9000 fits. The parent cubic stress comparison contributes another 63000 fitted responses. Paired recovery covers all 135000 fits in 270 cells, retaining all four accuracy states and undefined conditional rates when qualifying counts are zero.

The manuscript table uses `results/paired_recovery_summary.csv` with `scenario=quadratic`, `method=IV degree 2`, `penalty=0`. The [manuscript result map](../../../docs/MANUSCRIPT_RESULTS.md) documents the six selected cells and exact counts. The complete functional estimates, coefficients and paired rows are supplied in the package's `iv_reference_outputs_*.zip` and extracted by the runner.

The fixed polynomial classes, penalties, domain and reference law are declared restrictions. Weak-information failures and extreme estimates remain reported. Conditional recovery frequencies are descriptive simulation diagnostics with known truth and do not rank the scientific meanings of the two quantities. Sampling percentiles are not causal-effect confidence intervals; Wilson intervals concern the Monte Carlo accuracy frequency.
