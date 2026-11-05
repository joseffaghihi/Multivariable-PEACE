# Publication checks — 4 November 2026

The recorded dependencies were installed in a fresh Python 3.13 environment on macOS ARM64. Source formatting and undefined-name checks passed. All 128 reference data-file checksums and the selected manuscript table values passed verification.

The final portable package passed all three-study smoke tests, all numerical-validation stages and all report/figure reconstruction stages. The matched-mechanism full pipeline was also rerun from scratch at its complete 40-replicate design; all 4,800 calibration rows agree with the reference snapshot to absolute and relative tolerance 1e-10. verification.json records stage counts and outcomes.

The full sodium and customer Monte Carlo/bootstrap pipelines were not rerun in their entirety during publication packaging. Their complete sources, recorded seeds/settings and original final result snapshot are supplied, and their numerical validators and small fresh fits passed. Rebuilding reports from supplied outputs is distinct from a full simulation rerun.

The three core simulation modules retain the original function/class syntax trees after removal of formatting and comments: 22 sodium, 32 customer and 13 matched-mechanism definitions. Publication edits concern paths, explicit imports, execution wrappers and reporting documentation.

## Instrumental-variable addition — 5 November 2026

The complete parent cubic/linear run, quadratic comparison and large-sample follow-up were regenerated from the published sources in a fresh isolated output directory. All seven simulation/audit/report stages passed. All 28 resulting numerical CSV files are byte-identical to the original local outputs after decompressing gzip files. `IV_PUBLICATION_CHECKS.json` records the file list and scope.

The final multipart reference snapshot contains 69 IV entries. Together with the original 128 entries, all 197 reference-file checksums and every selected manuscript value pass `tools/check_results.py`. The earlier three studies' source directories and original reference archive are unchanged.

All four studies pass fresh-fit smoke checks. The final IV archive parts also pass numerical validation, complete paired-recovery reconstruction and report/figure reconstruction in new directories. The figures mode uses supplied results; the separately completed full mode genuinely starts without them. The archive parts preserve all data and each is under 20 MiB for browser-based publication. Source lint and formatting checks pass for all 62 Python files.
