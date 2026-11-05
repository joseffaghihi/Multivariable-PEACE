# Mathematical specification and interpretation

## 1. A valid instrument with fixed confounding

Take mutually independent \(W,U,R,E\sim N(0,1)\), and let
\[
X=\rho W+\tfrac12U+\sqrt{\tfrac34-\rho^2}\,R,
\qquad
Y=h(X)+U+\tfrac12E,
\qquad 0\le\rho\le0.8,
\]
\[
h(x)=x+\tfrac12(x^2-1)+\tfrac25(x^3-3x)
=\tfrac25x^3+\tfrac12x^2-\tfrac15x-\tfrac12.
\]
Here \(W\) is observed and \(U\) is not. The arrows are \(W\to X\to Y\), \(U\to X\), \(U\to Y\), \(R\to X\) and \(E\to Y\). There is no \(W\to Y\) arrow outside the treatment path.

In this particular SCM, the intervention on \(X\) replaces its equation and leaves the exogenous variables unchanged. Thus
\[
Y_x=h(x)+U+\tfrac12E,\qquad
g_{\mathrm{in}}(x)=\mathbb E[Y_x]=h(x).
\]
The target is a population total mean response. With no mediator or covariate, no separate controlled-direct response or covariate degree \(r\) is needed.

For every \(\rho\), \(X\sim N(0,1)\), \(\operatorname{Cov}(X,U)=1/2\), and \(V=Y-h(X)=U+E/2\) has variance \(5/4\). In particular
\[
\mathbb E[V\mid W]=0,\qquad
\mathbb E[V\mid X=x]=\tfrac12x\ne0.
\]
The mean-IV restriction holds while treatment ignorability does not. Since the bivariate normal law of \((X,U)\) is the same for every \(\rho\), the observable law of \((X,Y)\) is also unchanged. Only the information carried by \(W\) varies.

## 2. Why the instrument identifies the response, and why recovery becomes difficult

Let \(H_k=\operatorname{He}_k/\sqrt{k!}\) be orthonormal probabilists' Hermite polynomials. The conditional normal law is
\[
X\mid W=w\sim N(\rho w,1-\rho^2).
\]
Applying the Gaussian moment-generating identity to \(\exp(tX-t^2/2)\) gives
\[
\mathbb E[\exp(tX-t^2/2)\mid W=w]
=\exp(\rho tw-(\rho t)^2/2).
\]
Comparing powers of \(t\) yields
\[
\mathbb E[H_k(X)\mid W]=\rho^k H_k(W).
\]
Consequently, if \(q\in L^2(N(0,1))\) has Hermite coefficients \(a_k\), the conditional expectation operator multiplies coefficient \(k\) by \(\rho^k\). The Hermite expansion and conditional expectation contraction justify this calculation in \(L^2\). When \(\rho>0\), all multipliers are nonzero, so an identically zero conditional expectation implies every \(a_k=0\): completeness holds. This includes both response classes used in the benchmark. At \(\rho=0\), only the constant coefficient remains; all centered responses lie in the null space. A numerical estimate there has no IV identification interpretation.

The cubic causal response has orthonormal coefficients
\[
(\theta_0,\theta_1,\theta_2,\theta_3)
=(0,1,\sqrt2/2,2\sqrt6/5).
\]
An instrument with correlation \(\rho\) attenuates its linear, quadratic and cubic components by \(\rho,\rho^2,\rho^3\). Inverting noisy moments divides by these quantities. Moreover \(H'_k=\sqrt{k}H_{k-1}\), so derivative recovery gives relatively greater importance to higher-order coefficient errors. At \(\rho=0.2\), the cubic multiplier is just \(0.008\), even though the usual linear first-stage F is large at \(n=4{,}000\).

The heuristic component signal scale \(n\rho^{2k}\) helps interpret this design; it is not a universal sample-size guarantee. For the cubic direction at \(\rho=0.2\), it is only \(0.256\) at \(n=4{,}000\), and \(1.024\) at \(n=16{,}000\). Population completeness ensures uniqueness, not uniformly stable inversion or precise finite-sample derivative estimates.

## 3. Exact PEACE target

Declare \(\Gamma=(-L,L)\), \(L=2.5\), and use the restricted observational reference density
\[
f_\Gamma(x)=\frac{\phi(x)}{\Phi(L)-\Phi(-L)},\quad x\in\Gamma.
\]
The entire observed distribution is still used to estimate the IV relation. Only the functional is evaluated on this window. Since
\[
h'(x)=\tfrac65x^2+x-\tfrac15
=\tfrac65(x+1)(x-\tfrac16),
\]
the derivative is negative between \(-1\) and \(1/6\), and positive outside. PEACE aggregates movement magnitudes and does not have the sign of a treatment benefit.

The normalized Mean PEACE and its unnormalized counterpart are
\[
P_d(h)=\frac{\int_{-L}^{L}|h'(x)|f_\Gamma(x)^{2d}\,\mathrm dx}
{\int_{-L}^{L}f_\Gamma(x)^{2d}\,\mathrm dx}
=\frac{\int_{-L}^{L}|h'(x)|e^{-dx^2}\,\mathrm dx}
{\int_{-L}^{L}e^{-dx^2}\,\mathrm dx},
\]
\[
\operatorname{PEACE}_d(h)=P_d(h)\int_{-L}^{L}f_\Gamma(x)^{2d}\,\mathrm dx.
\]
At \(d=0\) the normalized target averages uniformly over the declared window; at \(d=1/2\) it uses the restricted natural treatment law. All reported targets are fixed across instrument strengths. For \(d=(0,1/4,1/2,1,2)\), respectively,
\[
P_d(h)=(2.42703704,\ 1.61422284,\ 1.12233598,\ 0.68522454,\ 0.44209210).
\]

Under the normalized weights, the manuscript's degree-derivative covariance identity specializes to
\[
\frac{\mathrm d}{\mathrm dd}P_d(h)
=2\operatorname{Cov}_{w_d}(|h'(X)|,\log f_\Gamma(X))
=-\operatorname{Cov}_{w_d}(|h'(X)|,X^2).
\]
The values decrease over the evaluated degree grid: larger degree emphasizes the central, higher-density treatment region, where weighted local movement is smaller in this example. The statement concerns these weights and magnitudes, not a negative causal effect and not a universal monotonicity theorem. No \(r\)-surface is present because there are no covariates.

## 4. What each estimator does

Using observed data only, write the fitted response as \(p(x)^\top\theta\), where \(p=(H_0,\ldots,H_K)\) with \(K=3\) or \(5\); instrument columns are \(q(w)=(H_0,\ldots,H_K)\). Intercepts are unpenalized. Center nonconstant columns and outcome, and define
\[
G_{qp}=n^{-1}Q_c^\top P_c,\quad G_{qq}=n^{-1}Q_c^\top Q_c,
\quad g_{qy}=n^{-1}Q_c^\top Y_c,
\]
\[
A=G_{qq}^{-1/2}G_{qp},\qquad b=G_{qq}^{-1/2}g_{qy}.
\]
Ordinary exactly identified series IV solves \(A\widehat\theta=b\). The regularized comparator solves
\[
\widehat\theta_\lambda=(A^\top A+\lambda I)^{-1}A^\top b.
\]
Restore the intercept as \(\overline Y-\overline p^\top\widehat\theta\). No true \(\rho\), \(U\), latent residual, or population operator is supplied to this algorithm. Ordinary least squares minimizes squared outcome residuals using the same cubic response columns; its population response is \(h(x)+x/2\), so increasing sample size cannot remove its causal bias.

The penalty is on the declared orthonormal response coefficients. At the population level, ridge transforms coefficient \(k\ge1\) into
\[
\theta_{\lambda,k}^{\mathrm{pop}}
=\frac{\rho^{2k}}{\rho^{2k}+\lambda}\theta_k.
\]
This explains the variance/bias tradeoff. A larger penalty can remove real curvature along weakly identified directions. At fixed \(\lambda>0\), increasing sample size approaches a penalized response, not generally the unpenalized causal response. Near-truth scalar PEACE under one penalty does not prove recovery of the underlying response.

For common known normalized weights \(w_d\), the reverse triangle inequality also gives
\[
|P_d(\widehat h)-P_d(h)|
\le\int_\Gamma |\widehat h'(x)-h'(x)|w_d(x)\,\mathrm dx.
\]
The functional therefore inherits the difficulty of estimating the causal derivative. Tuning its degree changes the target and cannot create information lost through a weak instrument.

## 5. Linear inference check

For the companion response \(h(x)=\beta x\), \(\beta=1\), the target is \(|\beta|\) at every degree. Let \(S_{ab}\) denote centered sample cross-products. The IV slope is \(\widehat\beta=S_{WY}/S_{WX}\). For a proposed coefficient \(b\), regress \(Y-bX\) on an intercept and \(W\). Under the true coefficient, its disturbance is independent normal \(U+E/2\), so the slope t test has an exact \(t_{n-2}\) null distribution even when the instrument is weak or irrelevant.

With \(c=t_{n-2,.975}\), inversion gives the coefficient confidence set of all \(b\) satisfying
\[
(n-2+c^2)(S_{WY}-bS_{WX})^2
\le c^2 S_{WW}(S_{YY}-2bS_{XY}+b^2S_{XX}).
\]
This quadratic inequality may define a bounded interval, two unbounded rays, or the entire line (including limiting half-line cases). The code retains its coefficients and maps the **whole set**, not just endpoints of a selected component, through \(b\mapsto|b|\). This gives a PEACE confidence set with at least the coefficient set's coverage; it can overcover because either sign maps to the same magnitude. Unbounded sets honestly express limited information.

The conventional comparator projects the usual homoskedastic IV Wald coefficient interval through absolute value. Its finite-sample coverage is measured, not presumed to be 95%. Neither inference procedure is presented as a validated confidence interval for the nonlinear regularized PEACE estimator.

## 6. Scope

This study shows the consequences of instrument weakness for the paper's **specified mean-response IV route**. It is not a violation-of-exclusion experiment and does not infer that every valid instrument identifies a causal mean curve under arbitrary heterogeneous mechanisms. The Gaussian additive model explicitly satisfies the causal mean restriction, and completeness holds at every positive strength. The zero-strength endpoint fails completeness. The numerical estimators impose finite polynomial classes. The marginal-density check is parametric and correctly specified. These restrictions delimit the conclusions.
