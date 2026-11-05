# Estimating a finite contrast and Mean PEACE from the same causal response

Let W,U,R,E be independent standard normal variables. The full SEM is
\[
X=\rho W+\tfrac12U+\sqrt{\tfrac34-\rho^2}\,R,
\qquad
Y=X+\tfrac34(X^2-1)+U+\tfrac12E.
\]
Only (W,X,Y) are observed. U is a hidden common cause of treatment and outcome. W has no direct outcome effect. Only the treatment equation is replaced under intervention, giving
\[
Y_x=x+\tfrac34(x^2-1)+U+\tfrac12E,
\qquad g_{\mathrm{in}}(x)=h(x)=\mathbb E[Y_x]=x+\tfrac34(x^2-1).
\]
The instrument supplies identifying moments and is not conditioned into this response. In this specified SEM E[Y−h(X)|W]=0, while E[Y−h(X)|X=x]=x/2. Gaussian completeness holds for every positive instrument strength; zero strength is an unidentified IV endpoint. The treatment marginal remains N(0,1) across strengths.

For the declared window \(\Gamma=(-2.5,2.5)\), define
\[
w_d(x)=\frac{e^{-dx^2}}{\int_\Gamma e^{-du^2}\,\mathrm du}.
\]
The two summaries are
\[
\Delta=h(1)-h(0)=\int_0^1h'(x)\,\mathrm dx=1.75,
\qquad
P_d=\int_\Gamma|h'(x)|w_d(x)\,\mathrm dx.
\]
At \(d=1/2\), the latter is normalized Mean PEACE under the restricted natural treatment law, with truth \(P_{1/2}=1.41838949\ldots\). All observations enter IV estimation; only the functional evaluation uses the declared window. There are no covariates or mediators, hence no covariate degree r.

The derivative is the infinitesimal local contrast rate:
\[
h'(x)=\lim_{\delta\to0}\frac{h(x+\delta)-h(x)}{\delta}.
\]
Mean PEACE aggregates magnitudes of these local rates across the domain. The single finite contrast from 0 to 1 does not determine this aggregation. Its integral permits cancellation and only covers its declared endpoints; Mean PEACE uses magnitudes and its specified domain/weights.

## Error propagation

Write a fitted response as \(\widehat h(x)=\widehat c+\widehat a x+\widehat b(x^2-1)\). The true coefficients are \(c=0,a=1,b=3/4\), but they are estimated from the observed data in a declared polynomial class. The intercept cancels from both summaries:
\[
\Delta(a,b)=a+b,\qquad P_d(a,b)=\int|a+2bx|w_d(x)\,\mathrm dx.
\]
Thus \(\widehat\Delta-\Delta=e_a+e_b\). For the natural weights at the true coefficients, differentiation under the integral is valid locally: the single zero of the slope has zero continuous reference mass and parameter derivatives are dominated on the bounded window. With \(L=2.5\), \(Z=\Phi(L)-\Phi(-L)\), and \(c_*=-a/(2b)=-2/3\),
\[
P_a=1-2\frac{\Phi(c_*)-\Phi(-L)}{Z}=0.50123999\ldots,
\qquad P_b=4\frac{\phi(c_*)-\phi(L)}{Z}=1.22286600\ldots.
\]
Therefore for small coefficient errors,
\[
\widehat P-P=0.50124e_a+1.22287e_b+o\!\left(\sqrt{e_a^2+e_b^2}\right).
\]
If \(e_a=-e_b\), the finite contrast is exact, while the first-order PEACE error is about \(0.72163e_b\). Error compensation in one finite contrast need not preserve local slopes or their magnitudes. This is a reason to study paired recovery, not a ranking of causal metrics by usefulness or a universal variance theorem.

## Paired simulation diagnostic

For each simulated dataset, define
\[
A_\Delta=\{|\widehat\Delta-\Delta|\le0.1|\Delta|\},\qquad
A_P=\{|\widehat P-P|\le0.1P\}.
\]
The marginal frequencies assess recovery of each target against its own truth. The conditional diagnostic is
\[
\widehat{\Pr}(A_P\mid A_\Delta)=\frac{\#(A_P\cap A_\Delta)}{\#A_\Delta}.
\]
It is undefined if no contrast estimate qualifies. The four states—both accurate, contrast only, PEACE only, neither—are retained for every cell. They sum to the full 500 repetitions; no dataset is removed from the underlying results.

At rho=0.2, n=64,000, 487 contrasts qualify and 436 of these also have accurate Mean PEACE, giving 89.5% conditional recovery. At n=256,000 the count is 499/500, or 99.8%. These are descriptive frequencies across repeated simulations with known truth. They are not an accuracy certificate obtainable from real observational data without truth, and do not imply that an accurate finite contrast guarantees an accurate local aggregation.

The study retains weak-information cases, all response-class/penalty sensitivity and the original cubic stress analysis. Its purpose is to characterize recovery from an estimated causal response and the information required for local aggregation.
