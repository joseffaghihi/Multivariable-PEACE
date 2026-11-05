# Weak-instrument study: design fixed before simulation

This supplementary numerical study addresses Reviewer 3, comment 3. It is kept separate from the manuscript and the three existing studies. The target is the mean-response IV construction in the manuscript's identification subsection, not a claim of weak-instrument robustness for PEACE.

## Data-generating mechanism

Let W, U, R and E be mutually independent standard normal variables. For each declared correlation rho,

    X = rho W + 0.5 U + sqrt(0.75 - rho^2) R,
    Y = h(X) + U + 0.5 E,
    h(x) = x + 0.5(x^2 - 1) + 0.4(x^3 - 3x).

Only (W, X, Y) are observed. U is an unobserved confounder; R is additional treatment noise. W has no direct effect on Y. Intervening on X replaces its equation and leaves the exogenous variables unchanged in this particular simulation. Thus E[Y_x] = h(x), and E[Y-h(X)|W] = 0. These are explicit model assumptions, not consequences of the definition of PEACE.

Across all rho, X ~ N(0,1), Cov(X,U) = 0.5, Var(Y-h(X)) = 1.25, and the true causal response and PEACE remain unchanged. The joint law of (X,Y) is also unchanged. Only the information supplied by W changes. At rho = 0 exclusion and mean exogeneity still hold, but relevance and completeness fail. This endpoint is reported separately as unidentified.

## Target and identification

Use the normalized Mean PEACE on the declared treatment window (-2.5,2.5), with the observational normal treatment law restricted to this window. The response is the population total mean h(x) = E[Y_x]. There are no covariates or mediators, so there is no r parameter or separate controlled/total distinction here. All observations, including observations outside the evaluation window, enter IV estimation.

The normalized target is the integral of |h'(x)| exp(-d x^2), divided by the integral of exp(-d x^2), over the window. This definition is finite at d = 0. The normalization of the truncated reference density cancels in the ratio. Both normalized and unnormalized numerical targets are saved, using the restricted, normalized reference density for the latter.

For orthonormal probabilists' Hermite polynomials H_k, E[H_k(X)|W] = rho^k H_k(W). Nonzero rho gives completeness on L2 of the normal treatment distribution. The finite cubic and quintic classes used for estimation therefore satisfy the paper's uniqueness requirement. Weakening rho attenuates higher-order response components especially severely. The usual linear first-stage F does not diagnose all these higher-order directions.

## Estimation comparisons

1. Cubic ordinary least squares, ignoring the latent confounder. Its population target is h(x)+0.5x, not h(x).
2. Cubic series 2SLS: empirical moments using the intercept and the first three Hermite polynomials of W as instruments for those of X. No true rho, latent variables, or population conditional-expectation operator is supplied to the estimator.
3. Cubic series IV with Tikhonov penalties lambda = 1e-6, 1e-4 and 1e-2 on the nonconstant orthonormal Hermite coefficients. All are reported. Lambda = 1e-4 is the prespecified primary regularized comparator. It is not selected from the results or by oracle error. A fixed penalty introduces shrinkage bias and is not asserted to give a consistent sequence as n increases.
4. Quintic series 2SLS and quintic regularized IV (lambda = 1e-4) as a response-class sensitivity analysis. The additional coefficients are zero in the true response but are estimated without this knowledge.

The basis class is a declared modeling restriction. This is a controlled series-IV benchmark, not unrestricted nonparametric discovery of an unknown response. No variable selection, oracle coefficient selection, winsorization, clipping, or removal of extreme IV estimates is used.

Primary results use the known marginal density to isolate response estimation. A paired secondary analysis plugs in a normal marginal density fitted to the observed X values. The latter is a correctly specified parametric density check, not evidence for arbitrary density estimation. The instrument-conditional law and instrument strength are estimated empirically in both cases.

## Repetitions, uncertainty, numerical integration

All seeds, strengths, n values, degrees, penalties and sample sizes are fixed in config.json before simulation. Use 500 nonlinear repetitions per cell, paired across strengths and nested sample sizes. Common random numbers improve comparisons; repetitions within each cell remain independent. Every estimate is retained. Report actual estimates, empirical bias, standard deviation, RMSE, Monte Carlo standard errors of finite-run summaries, median, 5th/95th/99th percentiles, relative-error frequencies, response error, derivative error and operator diagnostics. For just-identified IV, heavy tails can make population moments nonexistent: finite-run mean/SD/RMSE must not be read as guaranteed population moments, and quantiles are essential.

Integrate each fitted polynomial derivative separately between all its real roots inside the window. A fixed 32-point Gauss-Legendre rule per sign interval is independently checked against adaptive integration and a doubled rule. Quadrature tolerance is assessed separately from statistical error. Plotted response examples use repetition 0, fixed in advance, and repetition quantile bands are sampling-distribution bands, not confidence intervals for a single dataset.

## Linear inference check

Replace h(x) by beta*x, beta = 1, retaining the same treatment and disturbance equations. Mean PEACE = |beta| for every d. With 5,000 repetitions per cell, compare conventional homoskedastic IV Wald intervals with exact finite-sample Anderson-Rubin confidence sets, obtained by inverting the t test in regression of Y-bX on an intercept and W. Normality and independence of the structural disturbance and W make the test exact at the true coefficient, including zero strength. Map the complete coefficient confidence set through absolute value to obtain a PEACE confidence set. Retain unbounded and disconnected coefficient sets. No normal approximation to a weak-IV estimate is used by Anderson-Rubin inference.

For efficiency, linear sample cross-products are drawn from their exact Wishart distribution with n-1 degrees of freedom. This is distributionally equivalent to centering n independent draws of the Gaussian linear model. Direct-data regression and confidence-set inversion are checked independently. Coverage uncertainty is reported as Monte Carlo error; coverage does not imply a narrow or informative set.

## Interpretation fixed before seeing results

A strong valid instrument should recover this model-conditional causal response. Weak valid instruments are expected to produce unstable unregularized derivative estimates; regularization may reduce spread while distorting the response and PEACE. At zero strength there is no IV identification in the response class. Changing d changes the declared target, not instrument information; r is absent. The study should expose these limitations honestly and does not test predictive superiority, new identification assumptions, or universal robustness of PEACE.

## Primary methodological sources

- Newey and Powell (2003), Instrumental Variable Estimation of Nonparametric Models, Econometrica 71, 1565–1578. https://doi.org/10.1111/1468-0262.00459
- Darolles, Fan, Florens and Renault (2011), Nonparametric Instrumental Regression, Econometrica 79, 1541–1565. https://doi.org/10.3982/ECTA6539
- Andrews, Stock and Sun (2019), Weak Instruments in Instrumental Variables Regression: Theory and Practice, Annual Review of Economics 11, 727–753. https://doi.org/10.1146/annurev-economics-080218-025643

The Gaussian/Hermite calculations, target formulas and confidence-set inversion for this particular design are derived and independently checked in this study. They are not attributed as a new general IV method.
