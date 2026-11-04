# Publication checks — 4 November 2026

The recorded dependencies were installed in a fresh Python 3.13 environment on macOS ARM64. Source formatting and undefined-name checks passed. All 128 reference data-file checksums and the selected manuscript table values passed verification.

The final portable package passed all three-study smoke tests, all numerical-validation stages and all report/figure reconstruction stages. The matched-mechanism full pipeline was also rerun from scratch at its complete 40-replicate design; all 4,800 calibration rows agree with the reference snapshot to absolute and relative tolerance 1e-10. verification.json records stage counts and outcomes.

The full sodium and customer Monte Carlo/bootstrap pipelines were not rerun in their entirety during publication packaging. Their complete sources, recorded seeds/settings and original final result snapshot are supplied, and their numerical validators and small fresh fits passed. Rebuilding reports from supplied outputs is distinct from a full simulation rerun.

The three core simulation modules retain the original function/class syntax trees after removal of formatting and comments: 22 sodium, 32 customer and 13 matched-mechanism definitions. Publication edits concern paths, explicit imports, execution wrappers and reporting documentation.
