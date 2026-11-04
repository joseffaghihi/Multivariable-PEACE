# Customer satisfaction: separate experimental component

This is a simulation and observational-estimation study of the synthetic marketing SEM, not an analysis of real customer data. No manuscript files, sodium results, or estimation-section files are changed. The primary model follows the archived repository notebook. The printed outcome equation is retained as a separate coefficient sensitivity, not silently substituted.

## 1. Sources and decisions

Original simulator: ../../Customer_Satisfaction_Example.ipynb at repository commit 70fb6c2c3f51d0a76fd5ec909e53ade4b18b6307. Cell labels below are zero based. The revised scripts are the supported replication path.

| Issue | Primary analysis | Separate check |
|---|---|---|
| Usage coefficient: manuscript 10×3 versus notebook 10×0.3 | Notebook coefficient b=3 in expanded outcome | Oracle curves with b=30 |
| Graph has usage → bugs faced; generator has independent bugs | Omit that edge, follow generator | Common-seed original-generator intervention check |
| Original simulator has no outcome noise | Preserve original baseline | Add independent Normal(0,0.5²) outcome noise |
| Original includes 45 pair curves | Six preselected pairs with tractable population densities and exact gradients | Three representative joint observational benchmarks; no assertion that all 45 were replicated |
| SHAP treated as a competing causal estimator | Held-out predictive attribution | Compare questions, not numerical errors against causal truth |
| Original DML references undefined X_new/y_new | Explicit data and three-fold cross-fitting | Reproducible repeated simulations |

All new randomness uses local NumPy generators. Thus distributions and mechanisms reproduce the primary SEM; observations need not match the old global-random-state seed sample. The validation script executes only the original simulator cell to verify algebra and the disputed graph edge.

## 2. Structural equations and graph

Let C=Sales calls, I=Interactions, E=Economy, L=Last upgrade, N=Product need, D=Discount, M=Monthly usage, A=Ad spend, B=Bugs faced, R=Bugs reported; σ is the logistic function. All exogenous draws are independent in the primary model.

```
C ~ Uniform(0,4)                         I = C + Normal(.5,.05²)
E ~ Uniform(.5,1)                        L ~ Uniform(0,10)
N = .1 C + Normal(0,.1²)                 D = .25(1−σ(N)) + Uniform(0,.025)
M = σ(.3N + Normal(0,.1²))              A = M Uniform(.95,.99) + .05(10−L)
B ~ Uniform(0,2)                        R = B σ(N)
Y = 5N + bM + 2E + D − .2B + .2C + .3I + 8/(L+1),   b=3.
```

Graph edges are exactly those implied by these assignments. A and R have no outgoing edges. They therefore have zero total effect under the assumed graph, in addition to zero outcome coefficients. The added usage→bugs edge would define a different model and is not used. Fixing a variable replaces its assignment and propagates changes through downstream assignments. Bugs do not change under an intervention on usage.

## 3. Response targets, normalization and domain

For one exposure V, the original notebook computes the total mean response h_V(v)=E[Y_do(V=v)]. All other variables follow their structural equations. It does not fix every variable in the outcome equation. For a pair, h_VW(v,w)=E[Y_do(V=v,W=w)] is a simultaneous intervention. Intervening on a mediator blocks the corresponding ancestor-to-mediator path.

On an explicitly fixed domain Ω, the normalized total Mean PEACE is

```
P_d(h;f,Ω) = ∫Ω ||∇h(x)|| f(x)^(2d) dx / ∫Ω f(x)^(2d) dx.
```

For a scalar exposure, the positive and negative magnitudes replace |h′| by max(h′,0) and max(−h′,0). Their sum is P_d. At d=0 the domain receives uniform volume weight; d=0.5 gives its normalized natural exposure density; d=1 uses squared density. A finite intervention contrast is h(v1)−h(v0), a different quantity. Signed negative components are reported as nonnegative magnitudes.

All natural covariate distributions are already integrated into h. Consequently these targets have no remaining outer context index and no nontrivial r dimension. Inventing an r surface would change the target. A conditional-response extension would have to specify its contexts and density separately.

Single domains: C [0,4], I [.2,4.8], E [.5,1], L [0,10], N [−.3,.7], D [.07,.18], M [.35,.65], A [.3,1.2], B [0,2], R [0,1.5]. These are fixed before fitting and are never replaced by sample extrema. The N domain contains 0.9998089 of its population marginal probability. The domains contain sparse tails; d=0 estimation therefore requires more extrapolation than natural-density weighting. Oracle results are exact properties of the specified response on these domains, not evidence of good observational support throughout them.

Joint domains are Cartesian products except N+D, which uses N∈[−.3,.7] and D∈(.25(1−σ(N)), .25(1−σ(N))+.025), its actual curved observational support. The reported N+D curve is on this curved domain; it must not be equated to a rectangular-domain curve. Exact marginal and joint densities are used where needed. Uniform E,L and independence make E+L weights constant. For M+L, L is independent and uniform and the gradient norm depends only on L, so the M density cancels exactly. The A+R gradient is zero, so its joint density is immaterial. No artificial uniform distribution is asserted for M or A+R.

Native-variable units are retained. Single gradients have outcome units per exposure unit. A joint Euclidean norm depends on the coordinate units. A separate table rescales each coordinate by a fixed population-reference SD, demonstrating that rankings change with scale; it is not presented as an invariant causal ranking.

## 4. Exact derivatives and pathway accounting

Write e_M~Normal(0,.1²). Then

```
h′_N(n) = 5 + .3b E[σ′(.3n+e_M)] − .25σ′(n).
h′_C(c) = .2 + .3 + .1 E[h′_N(.1c+e_N)].
h′_L(l) = −8/(l+1)².
h′_I=.3, h′_E=2, h′_D=1, h′_M=b, h′_B=−.2, h′_A=h′_R=0.
```

For the N+M joint intervention, ∂h/∂n=5−.25σ′(n) and ∂h/∂m=b: fixing M removes the N→M pathway. For C+N, ∂h/∂c=.5 and ∂h/∂n=h′_N(n): fixing N removes C→N. For N+D, ∂h/∂n=5+.3b Eσ′(.3n+e_M) and ∂h/∂d=1: fixing D removes the N→D pathway. The remaining selected pair derivatives follow directly from the single formulas.

All original selected joint responses are additive in their two intervened coordinates, hence their mixed partials are zero. Positive joint PEACE is not synergy or an interaction estimate. An explicitly separate oracle positive control adds κ(N−.2)(M−.515) to Y with κ=10. Its N+M mixed partial is κ and its four-intervention contrast over n∈{.1,.3}, m∈{.48,.55} is .14, versus zero in the original model. This positive control does not replace the primary SEM and has no observational estimation benchmark here.

## 5. Theorem-based curve interpretation

Theorem 8.1 supplies the normalized gradient representation. Using the appendix derivation of Theorem 8.2(1), let p_d∝f^(2d) and J=||∇h||. Then

```
P′_d = 2 Cov_p_d(J, log f).
P″_d = 4 E_p_d[(J−EJ)(log f−E log f)²].
```

The displayed theorem currently uses raw f where the appendix differentiates log f. This package uses the log-density identity and checks it numerically. These identities interpret slopes and curvature in d, not causal direction or interaction. Positive/negative versions determine direction.

Theorem 8.2(2) gives concentration at density maximizers, under its stated regularity conditions. For a unique interior mode the limit is the local magnitude at that mode; for a positive-volume plateau it is the volume average over that plateau. Product need has a unique marginal mode n=.2. Upgrade has a flat uniform density on (0,10): every interior value is a density maximizer, and P_d=8/11 for every d. A nonlinear response can therefore have a perfectly flat PEACE curve. For the separately specified 10×Beta(2,2) upgrade distribution, the unique mode is 5 and the limit is 8/36=2/9; the response formula is unchanged.

Product need's slight decrease in d reflects negative covariance between its local magnitude and log marginal density. The N+D curve slightly increases; N+M slightly decreases. Those signs depend on the response, joint distribution, domain and normalization, not simply on a variable being called influential. A curved d plot is not a proof of nonlinear response; a nonflat straight segment is not a linear causal law. An estimated trend must be separated from response and density estimation artifacts before applying a population theorem to it.

## 6. Observational estimation

Parent adjustment identifies each single response in the fully observed, independent-error primary SEM, subject to the usual consistency and support conditions. The adjustment sets are: none for C,E,L,B; C for I,N; N for D,M; (M,L) for A. These are sufficient sets derived from the equations, not proposed to be variance-optimal sets. Other measured pre-exposure predictors could improve precision. R is excluded from generic parent-adjusted estimation because R=Bσ(N) gives zero residual treatment variance. Its causal null follows the specified graph; generic off-support response regression is not a valid way to verify it.

The flexible benchmark fits an additive cubic-spline conditional outcome mean in treatment and parents (five knots per coordinate, standardized spline features, Ridge α=.01). The conditional mean is additive for these sets in this SEM. Because its treatment derivative is independent of parents, averaging that derivative over the parent distribution yields the total-response derivative. This equality relies on the additive model here; the script does not claim it for arbitrary interactions. Final derivatives are computed analytically from the fitted spline basis and checked against prediction finite differences. Spline knots are fitted to each training sample; extrapolation is linear, matching boundary derivatives and giving a C¹ fitted response. Integration splits at knots and derivative roots, with 24 Gauss–Legendre nodes per segment, refined to 48 for verification. Fixed spline complexity and tail extrapolation are finite-sample modeling choices, not a nonparametric consistency proof.

Exposure densities are scale-adapted Scott-bandwidth Gaussian KDEs. For known bounded supports, reflection at physical boundaries mitigates leakage. The final weights are normalized on the fixed analysis domain in log space. Half and double Scott bandwidths are illustrative sensitivity checks, not validated bandwidth selectors. Oracle-weight results hold the population density fixed to isolate response estimation. Exact densities for N, I, D, M and A follow from the SEM via Gaussian convolutions and quadrature; A density is numerical quadrature, not an elementary exact expression. Constants cancel from normalized weights. Nonfinite estimates are not replaced by zero.

A favorable structural benchmark fits the ten outcome coefficients by ordinary least squares in the correct structural feature basis, including 1/(L+1), A and R. It propagates fitted coefficients through known assignment equations and known exogenous distributions to derive total-response gradients. This uses much more structural information than the parent-adjusted spline and comparator methods. In the original noiseless SEM the outcome lies exactly in that fitted basis, so recovery to numerical precision is expected, not evidence of universal estimator superiority.

Single-exposure replicates: original noiseless model n=5000, 40 datasets; outcome-noise sensitivity n=1000 and 5000, 80 datasets each; hidden-confounding sensitivity n=5000, 40 datasets for N,L,A. Three methods, degrees 0,.5,1,2, signed components, bias, RMSE and replicate SD are retained. Comparisons across methods with different information or sample sizes are descriptive, not a common-budget competition.

Joint estimation: N+M adjusts C; C+N and E+L require no external confounder set. Additive splines estimate both partial derivatives. N+M density factors into a marginal N KDE and a fitted Gaussian model for logit M|N including its Jacobian; C+N uses a marginal C KDE and fitted Gaussian N|C; E+L uses the independent reflected marginal KDE product. Thus joint density estimation exploits the known structural distribution family and independence. It is not a generic two-dimensional KDE benchmark. Sixty independent datasets per n∈{1000,5000}, outcome noise .5, degrees 0,.5,1,2. Oracle-density estimates isolate response error.

## 7. Causal forest and DML: each method's own target

Three-fold cross-fitted CausalForestDML and LinearDML use the same sufficient parent sets and continuous-treatment residual moments. Roots use an intercept-only nuisance model. Conditional treatment mean models are linear for N,I,A, and additive splines for D,M. Conditional outcome nuisance models are additive splines of parents. The forest has 120 trees, minimum leaf size 30, maximum depth 8; nuisance and forest seeds are fixed per replicate. Linear DML has a constant final effect, with parents supplied as W. Forest conditional slopes are averaged over a fixed 4096-observation population-reference parent sample. These are const_marginal_effect estimates, not a default 0-to-1 contrast across unsupported values.

The causal signed target is the conditional linear projection of the conditional intervention mean onto V under its natural assignment distribution. Under the stated identification conditions this equals the observable residual moment θ(c)=Cov(V,Y|parents=c)/Var(V|parents=c). The forest target is its natural-parent average. Linear DML generally uses a residual-variance-weighted average; they agree here because all varying slopes occur for N with constant Gaussian residual variance, or for roots without contexts. Other effects are linear in treatment with constant slopes. For N, Gaussian integration by parts makes the causal projection equal the natural mean total-response derivative. For C and L, use their exact population linear projections, not their average derivative. In particular,

```
Upgrade projection = [8−4.8 log(11)]/(100/12) = −.4211877…
Upgrade Mean PEACE = .7272727…   (on [0,10], all d)
Upgrade do(10)−do(0) = −80/11 = −7.272727…
```

These distinct numbers are all compatible with the same response. The original comparison must not treat disagreement among them as poor estimation. Each comparator is evaluated against its own exact projection target over 30 datasets per scenario, n=3000. Original, independent-outcome-noise and hidden-confounding scenarios are retained; the latter evaluates N,L,A only. No superiority claim follows from these experiments.

## 8. Identification sensitivity

Add latent U~Normal(0,1), set N=.1C+.1(ρU+sqrt(1−ρ²)e_N), and add γU to Y, with ρ=.5, γ=.1 and independent outcome noise .5. Marginal N and the interventional mean functions remain unchanged, but ignoring U shifts the observable need-response slope by γρ/.1=.5. The observable residual projection also shifts; it no longer identifies the unchanged causal projection target. Reported bias is against that causal target, not evidence that an estimator fails to estimate its observable residual moment. L and A are negative controls for this specific confounding path, not universal guarantees under hidden confounding. This is sensitivity analysis under violated identification assumptions, shared by causal methods and metrics; it is not a failure of PEACE as a functional.

## 9. Prediction and uncertainty

The prediction-only analysis trains XGBoost on 6000 cases, selects iterations using a separate 2000-case validation set, and evaluates once on 2000 held-out cases. Tree-path-dependent SHAP is computed on test cases; mean absolute SHAP is a predictive summary. A high SHAP attribution for a causal-null proxy is compatible with predictive performance. Causal interquartile contrasts use quantiles from a separate 200000-case fixed population reference. They share outcome units with SHAP but remain different quantities. Raw native-unit PEACE rankings are not compared to SHAP magnitudes as an accuracy test.

Bootstrap audit: 40 independent n=3000 outcome-noise datasets, 120 resamples each, N,L,A, d=.5; density held at truth. Ordinary 95% percentile intervals are assessed empirically, not assumed valid. PEACE is a magnitude functional and is nonregular at a zero gradient. Positive noisy norm estimates can therefore yield intervals that never contain zero. A separate 400-dataset reference fits a model linear in A with additive parent splines, constructs a robust HC3 signed-slope interval, and maps that entire interval through absolute value. If the signed interval contains zero, its magnitude interval starts at zero. This construction relies on a constant derivative for A; it does not establish general confidence bands for a nonlinear PEACE functional.

Coverage Monte Carlo standard error is sqrt[p(1−p)/R]; 40-dataset coverage proportions have appreciable simulation uncertainty. Bias Monte Carlo standard errors and RMSE Monte Carlo errors are added to summary tables by the report builder. Knot-count sensitivity (3,5,7) uses 40 separate n=5000 datasets for N,L,D,A with oracle densities. Domain and coordinate-scale dependence are reported separately.

## 10. Numerical verification and limits

Analytic outcome algebra is checked against the original simulator; common-random-number finite differences check all ten total derivatives; propagated feature derivatives are checked against the closed forms; oracle joint integrals are recomputed with 100 and 180 nodes per coordinate; density masses and signed decompositions are checked; the log-density covariance identity is checked by finite differences. Estimated single and selected joint integrals are refined from 24 to 48 nodes per knot/root segment. `results/validation.json` records numerical discrepancies and pass status. Log-sum-exp normalization prevents numerical zero denominators. The oracle response integrals use 24-node Gauss–Hermite and up to 180-node Gauss–Legendre rules.

Development audit: a preliminary single-grid 80-versus-320-node check exposed estimated-integral discrepancies as large as 2.21 for discount. Those results and the initial constant-extrapolation code are preserved in `audit/initial_global_quadrature/`; they are superseded. Final results use analytic spline derivatives, linear extrapolation and knot/root-split quadrature. This corrects integration and specifies a smoother boundary model; it is not a post-hoc tuning of the scientific scenarios or omission of unfavorable results. The preliminary pass flag covered oracle checks and was incomplete for estimated integrals; the final pass flag explicitly includes estimated-integration and fitted-derivative checks.

This package supports careful simulation claims and exposes practical estimation limitations. It does not validate real customer causality, a generally optimal smoothing procedure, general nonlinear-functional confidence intervals, or generic observational identification when the graph, overlap or independence assumptions are absent. The printed coefficient decision still needs author resolution before manuscript integration. Six pair analyses and three joint estimation benchmarks replace an exhaustive 45-pair display; the omitted pairs are not claimed to have been reproduced.
