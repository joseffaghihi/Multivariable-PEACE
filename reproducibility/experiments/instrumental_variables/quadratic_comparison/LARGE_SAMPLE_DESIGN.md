# Additional precision check

The first quadratic run improved recovery compared with the cubic benchmark, but at rho=0.2/n=4,000 only 29.2% of primary IV Mean PEACE estimates were within 10% of truth. At n=16,000 this increased to 52.6%. These results motivate an additional, explicitly exploratory sample-size check. The original results remain unchanged.

Before running this check, fix n=64,000 and 256,000, rho=0.2, 0.1 and 0.05, and 500 repetitions for every cell. Retain the identical quadratic response, treatment equation, noise/confounding, treatment window, degree grid, seeds and primary ridge penalty 1e-4. Compare quadratic OLS, quadratic unregularized IV, primary quadratic ridge IV and the direct scalar signed-average comparator. The expanded response class and full penalty grid remain available in the original quadratic run; this larger-sample check is explicitly conditional on the primary quadratic class.

All three metrics are computed from every fitted response. Known-density and fitted-normal-density variants are saved. No additional response coefficients or penalty values are searched. Report all six large-sample cells, including any that do not achieve good precision. No power/accuracy threshold is used to decide whether to retain a run.

For the quadratic direction the illustrative n*rho^4 scales are 102.4 and 409.6 at rho=0.2; 6.4 and 25.6 at rho=0.1; and 0.4 and 1.6 at rho=0.05. These anticipate a substantial range of effective information despite large n. They are not a sample-size theorem or a general weak-IV diagnostic cutoff. Good estimation at small correlation and large n must be described as recovery with adequate effective information, not evidence that genuinely weak identification has been overcome by PEACE.

The seed is shared with both prior studies, so the first observations remain nested and paired. This extension is additional evidence about precision versus sample size, not a new independent confirmatory study and not a replacement for the visible weak-IV cases. The original linear robust-confidence-set analysis remains separate.
