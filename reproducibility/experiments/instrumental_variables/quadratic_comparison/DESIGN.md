# Quadratic-response follow-up and causal-summary comparisons

This extension was planned after examining the original cubic weak-IV benchmark. It is an exploratory follow-up with a mathematically motivated simpler response, not an independently preregistered validation. The configuration, comparisons and primary penalty below are fixed before running this extension. All declared strengths, sample sizes, penalties and expanded-class results are retained. The manuscript and the original cubic study are unchanged.

## Model and purpose

Retain independent W,U,R,E ~ N(0,1) and the original treatment/confounding mechanism:

    X = rho W + 0.5 U + sqrt(0.75-rho^2) R,
    Y = h(X) + U + 0.5 E,
    h(x) = x + 0.75(x^2-1).

Observe only (W,X,Y). Under the specified treatment intervention, E[Y_x] = h(x); causal exclusion and E[Y-h(X)|W]=0 hold. The joint law of (X,Y), treatment density N(0,1), and confounding are fixed across rho. The response derivative is 1+1.5x, so it has a genuinely negative region and a positive region. This avoids reducing the experiment to a constant-slope case where every degree gives the same PEACE.

There are no covariates or mediators: the response is the population total mean and r is absent. All target integrals use the same declared treatment window (-2.5,2.5) and normalized weights w_d(x) proportional to exp(-d x^2). All observations enter IV fitting, including observations outside this window. Known marginal weights isolate response estimation; paired fitted-normal weights quantify an additional, correctly specified density-estimation step.

The cubic study required recovery of a third Hermite component with multiplier rho^3. This quadratic response requires only multipliers rho and rho^2. At fixed n/rho it may therefore be easier to estimate, but good performance is a hypothesis, not a selection criterion. Low instrument correlation is not by itself a finite-sample weak-identification diagnosis. Report linear F, smallest response-class canonical correlation and the illustrative mode scale n*rho^(2K). Positive rho satisfies Gaussian completeness; rho=0 is an explicitly unidentified IV endpoint.

## Comparable functionals

For every fitted response, evaluate three distinct causal targets against their own exact truth:

1. Mean PEACE: P_d = integral |h'(x)| w_d(x) dx.
2. Weighted signed average derivative: S_d = integral h'(x) w_d(x) dx.
3. Mean intervention contrast: Delta = E[Y_1]-E[Y_0] = h(1)-h(0).

For this quadratic mechanism S_d=1 for every degree because the weights are symmetric; Delta=1.75. PEACE depends on curvature through the derivative's magnitude, with different values at different d. Under the original cubic response S_d also involves curvature. Recompute S_d and Delta for every already-saved cubic response fit; reuse its PEACE estimates exactly rather than replacing the original simulation. Do not interpret signed cancellations as lack of individual/local movement.

The functionals have different scientific meanings. Their numerical magnitudes, and raw RMSEs across outcome-versus-derivative units, are not a superiority ranking. Report actual estimates, bias, empirical SD, finite-run RMSE, relative RMSE, percentiles, within-10% rates and sign-reversal frequencies where meaningful. All truths in this declared grid are nonzero. Relative error can nevertheless be misleading for a near-zero signed target and is not proposed as a universal evaluation criterion. Estimates of the contrast are independent of d; identical contrast values may be repeated across the degree index in the wide data for convenient pairing, never pooled as independent observations.

## Estimation and modeling restrictions

Primary response class: all quadratic polynomials in X, represented by normalized Hermite columns, with the same-degree functions of W as instruments. Fit OLS, unregularized exactly identified series IV and ridge IV at the three fixed penalties 1e-6, 1e-4 and 1e-2. The primary ridge value is 1e-4, unchanged from the cubic study. Expanded-class sensitivity fits all cubic polynomials with cubic instrument columns, both unregularized and ridge IV at 1e-4. Extra true cubic coefficients are zero, but estimators are not told this.

The same empirical centered moment equations and unpenalized intercept as the original study are used. No true rho, confounder, true coefficients, true conditional operator or true outcome residual is supplied to the response estimators. The main quadratic class is correctly specified by design and is a material restriction; do not call this unrestricted nonparametric learning. Additional flexibility can worsen recovery.

A separate direct scalar-IV comparator estimates S_d=1 in the quadratic Gaussian model as Cov_n(W,Y)/Cov_n(W,X). Indeed E[W(X^2-1)]=0, so the ratio targets the linear coefficient a for any h(x)=a x+b(x^2-1), and S_d=a under the symmetric reference weights. It exploits the Gaussian/quadratic restrictions and is not asserted to estimate the restricted signed derivative under an arbitrary nonlinear response. This comparator illustrates the distinction between estimating a functional directly and first estimating a full response. It does not supply PEACE or the intervention contrast. Its density-plugin counterpart is not computed, because estimating a signed derivative under nonsymmetric fitted weights would also require curvature.

## Simulation and reporting

Use the original seed and common random draws: 500 repetitions at n=1,000, 4,000, 16,000 and rho=0.8,0.5,0.2,0.1,0.05,0. The quadratic and cubic response studies use identical observed X and W and identical latent U,E for a given repetition/n/rho. This improves comparison of structural difficulty. Rerun no alternative response coefficients to select the most favorable outcome. Keep all extreme estimates and the zero-strength endpoint.

Seven response fits per quadratic dataset give 63,000 fits. Evaluate the full degree grid for each; the original cubic study also has 63,000 fits. Save fitted coefficients, response/derivative errors, all three functionals, paired density estimates and diagnostics. Direct scalar IV has one estimate per quadratic dataset. Monte Carlo averages of just-identified IV can have nonexistent population moments; mean/SD/RMSE and their plug-in MC errors are finite-run diagnostics only. Percentiles and accuracy frequencies are necessary. Sampling-distribution percentile ranges are not confidence intervals.

Primary compact comparisons use d=1/2 and n=4,000, but all n, degrees and strengths must be available, including results that do not improve. The expanded response class is displayed. The existing cubic mechanism remains the more demanding stress analysis, not discarded evidence. No weak-instrument-robust confidence claim is added for nonlinear metrics; the original linear Anderson–Rubin analysis retains its stated limited scope.

## Numerical verification

Integrate polynomial derivatives between all real roots, using the original split-sign quadrature. Compute signed derivatives with the same weights and intervention contrasts at the prespecified endpoints. Independently check exact truths, quadrature, parameter/moment estimates, repeated summaries, common-random-number pairing, the inequality P_d >= |S_d|, equality of zero-degree known/fitted-density functionals, and the direct scalar-IV identity. Verify source hashes of the manuscript and original study remain unchanged. Repeat the full extension to check reproducibility.

Methodological sources and the Gaussian/Hermite identification calculation are documented in the parent study's DESIGN.md and Mathematical_argument.md. This extension changes the response's complexity and compares functionals; it does not propose a new general IV estimator or a general weak-instrument robustness property of PEACE.
