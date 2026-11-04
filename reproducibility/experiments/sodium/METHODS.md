# Sodium experiment: design, targets and interpretation

This is an exploratory revision analysis of the original sodium notebook, not a preregistered study or a validation of clinical claims. The original manuscript and notebooks are preserved. This package focuses on normalized **Mean PEACE**, the manuscript's principal metric. Statements about flatness and units below do not automatically apply to unnormalized PEACE.

## Structural model

The baseline equations reproduce the original mechanism, using an independent random-number generator and an explicit seed for every dataset:

\[
A=65+5E_A,\quad S=A/18+E_S,
\]
\[
B=90+2.5S+2.5(S-3.5)_++10e^{0.1(A-65)}+10E_B,
\qquad P=0.4S+0.3B+E_P.
\]

Errors are independent standard normal variables. The graph has A -> S, A -> B, S -> B, S -> P and B -> P. Proteinuria P is not an adjustment variable for estimating sodium's causal response.

The simulator is illustrative: the Gaussian sodium distribution can generate a very small number of negative values, and age interventions are hypothetical. No real patient data or clinically validated effect magnitudes are used.

## Causal response versus observational conditioning

For sodium, age is a baseline covariate. The relevant response can be defined as h_S(s,a) = E[B_s | A=a]. An intervention on age is not required for this sodium estimand. In this baseline SCM it equals the structural mean response g(s,a).

For age's controlled response, sodium is a mediator. Define h_A(a,s) = E[B_{a,s}]. This fixes sodium and removes the age -> sodium -> blood pressure pathway. Its algebraic mean surface again equals g(s,a) in this simulator. Equality of these surfaces is a property of the specified model, not a statement that their causal questions are identical.

For age's total response, use E[B_a], allowing S to respond through its equation. In the baseline SCM:

\[
L_S(s)=\partial_s g(s,a)=2.5+2.5\mathbf 1\{s>3.5\},
\qquad L_A(a)=\partial_a g(s,a)=e^{0.1(a-65)}.
\]

\[
\frac{dE[B_a]}{da}=e^{0.1(a-65)}+\frac{2.5+2.5\Phi(a/18-3.5)}{18}.
\]

The kink at 3.5 is treated through the almost-everywhere derivative and a split integration interval. The response is locally Lipschitz; this numerical treatment is not a proof of every smoothness claim in the manuscript.

## Finite-domain estimands

Use fixed domains D_S=[0,8] and D_A=[50,80]. Training data are not truncated. The target distributions are normalized on these domains. Domain sensitivity uses sodium domains [0,7], [0,9] and [1,7], holding the age domain fixed.

For fixed a:

\[
p_d(s\mid a)=\frac{f_{S\mid A}(s\mid a)^{2d}}
{\int_{D_S}f_{S\mid A}(u\mid a)^{2d}\,du},\qquad
q_r(a)=\frac{f_A(a)^r}{\int_{D_A}f_A(v)^r\,dv}.
\]

\[
\overline{\operatorname{PEACE}}^{(r)}_d(S\to B)
=\int_{D_A}q_r(a)\int_{D_S}|L_S(s)|p_d(s\mid a)\,ds\,da.
\]

For age's controlled response, swap S and A and use |L_A(a)|. At d=0 the inner weight is uniform on the chosen finite interval; at r=0 the outer weight is uniform on its interval. At d=1/2 and r=1 the original conditional density and covariate marginal are used, each with the stated finite-domain normalization. This two-stage truncated target is not exactly the joint distribution conditional on both domain restrictions, and is not exactly the unrestricted observational population.

For joint sodium-age Mean PEACE:

\[
J_d=\frac{\int_{D_S\times D_A}\sqrt{L_S(s)^2+L_A(a)^2}\,f_{S,A}(s,a)^{2d}\,ds\,da}
{\int_{D_S\times D_A}f_{S,A}(s,a)^{2d}\,ds\,da}.
\]

There are no remaining covariates, hence no r. This norm combines raw sodium and age coordinate scales. Its numerical magnitude depends on the selected units; it should not be presented as invariant to arbitrary rescaling or as an interaction effect.

## Separate comparison questions

### Interpretation of different estimands

Exact PEACE, population average derivatives, finite contrasts and quantile treatment effects are calculated from the same SCM. A sodium intervention from 2 to 5 changes B by 11.25 for every unit; all quantile treatment effects for this contrast are therefore also 11.25. Dividing by the three-unit exposure change gives 3.75, which is an interval-average slope, not automatically the PEACE value under other weights.

An auxiliary quadratic sodium-response illustration on [2,5] demonstrates cancellation: g_S(s)=2.5(s-3.5)^2 has a zero endpoint contrast and a zero uniform average signed derivative, but uniform Mean PEACE equals 3.75. This is explicitly a separate oracle illustration, not the original response or a repeated-sample estimator benchmark.

### Estimation of a matched target

Observational PEACE estimates are compared with the exact same PEACE functional, domain and d/r. Two response estimators are used:

1. Structural-basis least squares: features S, (S-3.5)+, exp[0.1(A-65)] and A-65. This is a favorable benchmark with knowledge of the functional basis and threshold; it is not a general applied estimator.
2. Flexible additive cubic splines: seven fixed knots on each target domain, linear extrapolation, unscaled B-spline basis and ridge alpha=0.1. It does not know the threshold's special role in the response fit. It does assume additivity; interaction-rich models are outside this experiment.

Both models estimate conditional mean responses, not unobserved individual structural outcomes. Gaussian treatment/covariate density parameters are estimated from the same sample. Using the same sample for this plug-in estimator is disclosed; no orthogonality or general semiparametric inference theorem is claimed.

Oracle-density versions hold the density at its true value to isolate response estimation. A separate density-only experiment holds response derivatives at truth and compares Gaussian densities with covariance-adapted Gaussian KDE using Scott bandwidth, half Scott and double Scott.

### Causal Forest and DML targets

CausalForestDML and LinearDML are run with continuous treatments, three-fold cross-fitting, linear treatment nuisance models, spline outcome nuisances and fixed seeds. Causal Forest uses 120 trees, depth 8, minimum leaf 10. LinearDML uses spline effect modification. Predictions are explicitly obtained with const_marginal_effect rather than an implicit 0-to-1 call.

For Gaussian S|A, their conditional treatment-linear slope target is

\[
\theta_S(a)=\frac{\operatorname{Cov}(S,B\mid A=a)}{\operatorname{Var}(S\mid A=a)}
=E[L_S(S)\mid A=a].
\]

The final comparator target averages theta_S over the full natural age distribution. For age adjusted for sodium, the analogous Gaussian conditional slope averages the controlled age derivative, not the total age derivative. These equalities use Gaussian Stein identities in this specific simulator. They do not hold automatically for arbitrary non-Gaussian treatments or arbitrary nonlinear estimators.

Comparator values are validated against these own targets, not against a different d/r PEACE target. The finite-domain PEACE targets are close to, but not exactly equal to, these full-population slopes at d=1/2,r=1. The numerical proximity is explained analytically, not presented as proof of universal agreement.

## Simulation design

- Baseline: n=500, 2,000 and 8,000; 100 datasets each.
- Low overlap: sodium residual SD 0.25 instead of 1; n=2,000; 100 datasets.
- High noise: blood-pressure noise SD 20 instead of 10; n=2,000; 100 datasets.
- Hidden confounding: S=A/18+sqrt(1-rho^2)E_S+rho U, with rho=0.5, and B gains 10U; n=2,000 and 8,000; 100 datasets each. Marginal S/A densities and true causal derivatives remain unchanged; the maintained adjustment-based identification assumptions are violated.
- Linear response: B=100+3.75S+(A-65)+10E_B; n=2,000; 100 datasets. Its exact normalized curves are flat.
- Skewed treatment: sodium residual is 0.9 N(-0.3,0.6^2)+0.1 N(2.7,0.6^2), with mean zero; n=2,000; 100 datasets. Gaussian density fitting is now misspecified.

This gives 900 main datasets. Joint PEACE estimation reuses the 300 baseline datasets. Additional blocks use 150 comparator datasets (50 per scenario), 60 density-only datasets (30 per density scenario), 100 bootstrap evaluation datasets with 200 resamples each, 100 collider-adjustment datasets and 100 separability-violation datasets. These blocks have distinct documented seed ranges. Repeated evaluation of multiple targets on one dataset does not create independent replications.

Primary targets: (d,r)=(0,1),(0.25,1),(0.5,1),(1,1),(2,1),(0.5,0),(0.5,2). Oracle curves additionally cover d in [0,3] and r=0,0.5,1,2,3. This range is a sensitivity exercise, not a parameter-selection algorithm.

Report bias, RMSE, sampling SD, and Monte Carlo standard errors for bias and RMSE. Percentile bootstrap coverage and width are empirical diagnostics; neither model validity nor coverage for all targets is guaranteed.

## Separability violation

Add kappa(S-3.5)U to the outcome, with U independent of observed inputs. The conditional mean response does not change, but individual sodium derivatives become L_S(s)+kappa U. Therefore E_U|L_S+kappa U| generally exceeds |E_U(L_S+kappa U)|. A folded-normal identity supplies exact individual-magnitude targets. This explicitly distinguishes two possible estimands; nonidentifiability of the individual-magnitude quantity from the conditional mean alone is not an estimation error for the mean-response quantity. The paper must state which it intends when separability fails.

## Reading the curves

With L(s,a) the magnitude of the relevant local derivative, let m_d(a)=E_{p_d}[L(S,a)]. For fixed r:

\[
\partial_d\overline{\operatorname{PEACE}}^{(r)}_d
=2E_{q_r}\left[\operatorname{Cov}_{p_d}(L,\log f_{S\mid A})\right].
\]

For fixed d:

\[
\partial_r\overline{\operatorname{PEACE}}^{(r)}_d
=\operatorname{Cov}_{q_r}(m_d(A),\log f_A(A)).
\]

Increasing d shifts relative weight toward high-density exposure regions within each covariate stratum. Increasing r shifts outer weight toward high-density covariate regions. Positive/negative curve slopes describe the corresponding weighted covariance, which can change sign as the weights change.

A flat curve is guaranteed by constant local magnitude, but does not prove constant magnitude: covariance can vanish for other reasons. An increasing curve does not show that increasing the exposure increases the outcome; PEACE uses magnitudes. A curved d-plot does not alone prove a nonlinear dose-response. A straight nonflat segment describes approximately constant sensitivity to d over that range, not a linear structural causal law. Turning points are parameter-, density- and domain-dependent, not universal causal thresholds.

## Numerical and development audit

All density powers are normalized in the log domain. Oracle integration is checked against closed-form truncated-normal expressions, finer quadrature and finite differences. The sodium kink is an explicit quadrature split. Final core integration uses 96 Gauss-Legendre nodes per sodium segment and 96 age nodes; bootstrap uses 64; KDE sensitivity uses 48. Selected estimates are checked against substantially finer grids. No nonfinite value is silently replaced by zero.

The initial exploratory implementation standardized spline columns before ridge regression. Sparse tail columns could have nearly zero variance and generate extreme derivatives. The corrected version penalizes the bounded unscaled basis. Initial code/results are preserved under audit/initial_standardized_spline; they are not the final results. A coarse integration check also failed in sparse regions, prompting refinement before final reporting. These are disclosed development corrections, not discarded unfavorable scientific scenarios.

The manuscript's covariance theorem currently displays covariance with raw density, while its appendix derivation uses log density. Curve interpretation here uses the log-density identity, verified numerically. This discrepancy should be corrected when editing the manuscript.

## Scope limits

These are synthetic, low-dimensional, mostly additive mean-response experiments. They do not establish universal superiority of PEACE, a fully general estimator, robustness to arbitrary hidden confounding, or a general inference theorem. There is no instrument in this graph, so weak-instrument validation remains a separate task. Causal feature selection and a broader prior-art comparison also remain separate manuscript tasks.
