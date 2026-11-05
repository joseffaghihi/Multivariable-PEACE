# Estimating a finite contrast and aggregated local causal movement

This is an estimation study of two summaries of the same fitted causal mean response. The finite contrast supplies a reference for studying recovery of Mean PEACE. Their scientific meanings differ, so their accuracy percentages are not a ranking of causal metrics. The selected quadratic results support the manuscript IV example.

The SEM has W,U,R,E independent N(0,1), X=ρW+0.5U+√(0.75−ρ²)R, and Y=X+0.75(X²−1)+U+0.5E. Only W,X,Y are observed; U is hidden confounding. Intervening only on X gives h(x)=E[Y_x]=x+0.75(x²−1). Coefficients are learned in a declared polynomial response class.

The two targets are Δ=h(1)−h(0)=1.75 and Mean PEACE P=∫|h′(x)|w(x)dx=1.4184, with normalized restricted natural treatment weights on (−2.5,2.5), d=1/2. Mean PEACE aggregates infinitesimal local contrast magnitudes across the domain. It is not calculated from the single finite contrast Δ alone.

Actual estimates at ρ=0.2, n=64,000, across 500 simulated datasets (ranges are sampling-distribution percentiles, not confidence intervals):

| Quantity | Truth | Median | 5th–95th percentiles |
| --- | --- | --- | --- |
| Mean PEACE | 1.4184 | 1.4169 | 1.2707–1.5735 |
| Intervention contrast 0 → 1 | 1.7500 | 1.7503 | 1.6236–1.8799 |

Paired recovery at fixed ρ=0.2, d=1/2; 500 datasets per row. An estimate is accurate here when its relative error is at most 10% of its own known truth:

| n | Contrast within 10% | Mean PEACE within 10% | Mean PEACE accurate among accurate contrasts |
| --- | --- | --- | --- |
| 1,000 | 22.2% | 22.2% | 78/111 (70.3%) |
| 4,000 | 38.0% | 29.2% | 138/190 (72.6%) |
| 16,000 | 69.6% | 52.6% | 262/348 (75.3%) |
| 64,000 | 97.4% | 87.2% | 436/487 (89.5%) |
| 256,000 | 100.0% | 99.8% | 499/500 (99.8%) |

At n=64,000, 487 contrast estimates are accurate; 436 of these also have accurate Mean PEACE (89.5%). In 51 datasets the contrast is accurate but Mean PEACE is outside tolerance. At n=256,000 the conditional recovery frequency reaches 99.8%. This directly describes how reliably the local aggregation is recovered when the reference contrast is accurately estimated.

An accurate finite contrast does not guarantee accurate local aggregation: errors in different response components can cancel in h(1)−h(0) while affecting |h′| across the treatment domain. For a fitted quadratic c+a x+b(x²−1), Δ=a+b while P=∫|a+2bx|w(x)dx. These conditional percentages are simulation diagnostics using known truth, not an accuracy test available from real data alone.

Candidate explanatory remark for later discussion:

The IV simulation evaluates recovery of a specified causal response through its finite intervention contrast and Mean PEACE. Recovery of the aggregated local variation improves with sample size, while weak information about response curvature can leave Mean PEACE uncertain even when the finite contrast is accurately estimated.

The full report retains weak-information cases, the original cubic stress analysis, fixed-penalty and response-class sensitivity, density checks and all simulation data. [Full report and figures](REPORT.html).
