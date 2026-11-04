# Exact separation of matched causal summaries

The purpose is to show information that a two-parameter causal-variation surface contains beyond several fixed summaries. It is not to declare those summaries incorrect, nor to claim that no other weighted derivative functional can distinguish the mechanisms.

Let the three treatment intervals have width w=1/3 and densities rho=(1/2,1,3/2). Let profile probabilities be pi=(.9,.1). For an integer k>0 and local coordinate u in [0,1], put v_k(u)=sin²(pi*k*u). In common profiles, mechanism A uses frequencies b=(3,3,1) and B uses a=(4,1,2). In rare profiles, A uses 4a and B uses 4b. Add independent Gaussian output noise of SD .02. The scalar causal response is the conditional intervention mean h(t,z), not a predictive attribution. T is randomized independently of Z, and h(t,z)=E[Y|T=t,Z=z] under the maintained model. There are no mediators: direct and total treatment responses coincide here. Additive independent noise makes the structural treatment gradient and conditional-mean gradient equal, so this example satisfies the paper’s observable-gradient bridge rather than relying on heterogeneous structural effects.

## 1. Regularity and local variation

Every v_k is zero at both endpoints and has zero endpoint derivative. Responses join continuously with continuous first treatment derivative across the three intervals. For one treatment interval, integral |h'|=2k and integral h'^2=pi²*k²/(2w). Positive and negative variation are each k, so they cancel in every signed boundary contrast. There is no conflict with an endpoint causal effect equal to zero: local movement is a different estimand.

## 2. Matched derivative summaries

The identities sum a=sum b=7, rho·a=rho·b=6, and rho·a²=rho·b²=15 establish exact equality of uniform-treatment total variation, natural mean absolute derivative and natural mean squared derivative within each profile. Rare profiles multiply the absolute quantities by four and squared quantities by sixteen. Consequently every profile reweighting of those matched conditional quantities remains equal. Natural profile-averaged values are respectively 18.2, 15.6, and 56.25*pi²≈555.165248.

## 3. Matched distributions and information strength

For U uniform on [0,1], v_k(U) has Beta(1/2,1/2) distribution for every integer k. Constant within-interval density powers preserve a uniform local phase, and categorical profile reweighting only mixes identical conditional output laws. The entire output law is therefore the same for A and B under the natural law AND the full escort-weight family. Gaussian convolution preserves equality. This implies equality of output quantiles, moments and entropy under each such reference law; it does not imply equality of Y under every specified intervention value t.

For the independent-parent graph T→Y←Z, the KL edge-cut strength of T→Y reduces to conditional mutual information (Janzing et al., Theorem 4; the continuous-density version uses integrals):

C_(T→Y)=I(T;Y|Z)=H(Y|Z)−H(Y|T,Z)=H(Beta(1/2,1/2)+epsilon)−H(epsilon).

Both entropies are identical across mechanisms and finite because noise variance is positive. The edge-cut distribution is p(t)p(z) integral p(y|t',z)p(t')dt'=p(t)p(z)p(y|z), so the equality also follows directly from the KL definition. This compares the declared treatment edge, not every edge of the two DAGs.

The common and rare frequencies in each interval are unequal. Orthogonality of their centered cosines gives covariance zero within every interval. Thus Var(h)=1/8, Var(E[h|T])=(.9²+.1²)/8=.1025, Var(E[h|Z])=0, and treatment/profile interaction variance=.0225 for both mechanisms. The corresponding first/total Sobol quantities match. This equality also persists if the same constant-within-interval treatment escort and profile escort are used for both: replace profile probabilities by their escort probabilities in the variance formulas.

## 4. PEACE separation and crossover

Let B_d=w sum_j rho_j^(2d), q_r(z)=pi_z^r/sum_l pi_l^r. Then

P_M(d,r)=sum_z q_r(z) [sum_j 2*k_M,jz*rho_j^(2d)]/B_d.

Since a−b=(1,−2,1),

P_A−P_B = [2(.5^(2d)−2+1.5^(2d))/B_d] × [(4*.1^r−.9^r)/(.9^r+.1^r)].

The treatment factor is zero at d=0 and d=.5, negative for 0<d<.5 and positive for d>.5 by strict concavity/convexity of x^(2d). The profile factor changes sign at r*=log(4)/log(9)≈.630929754. Therefore the surface has analytically specified equality lines and opposite ranking regions. In particular:

- d=1,r=1: A=96/7≈13.714286, B=99/7≈14.142857, so B has more weighted movement.
- d=1,r=0: A=192/7≈27.428571, B=183/7≈26.142857, so A has more weighted movement.

The mechanism is fixed when preferences change. Larger PEACE does not mean greater benefit or better outcome; it means greater local movement under the declared weights. At r=0, rare profiles receive .5 weight instead of .1. At d=1, density-squared weighting emphasizes frequently used treatment regions. These are explicit target choices, not parameters optimized to obtain a preferred conclusion.

## 5. Stronger derivative comparator and boundaries of novelty

On the same escort laws, a weighted squared-derivative measure also distinguishes the systems. Its difference is proportional to [7*.5^(2d)−8+3*1.5^(2d)] × [16*.1^r−.9^r], with positive normalizers. Its profile crossover is log(16)/log(9)≈1.261859507, different from PEACE's. It measures derivative energy; PEACE measures accumulated absolute movement. Neither is universally the right scientific objective. A weighted absolute-derivative functional using the identical laws is precisely PEACE's normalized gradient representation, so no separation from that identical functional is possible.

Complete conditional response curves, suitably chosen interior contrasts, or a tailored derivative metric can reveal the mechanism differences. The separation is from the listed scalar summaries and distribution-based treatment-edge strength, not from all causal information or all conceivable metrics. A constructed example does not prove historical novelty of the formula. It makes its particular estimand and added information explicit.

Sources: [Janzing et al., Quantifying causal influences](https://webdav.tuebingen.mpg.de/causality/AOS2013-Janzing.pdf); [Sobol' and Kucherenko, derivative-based sensitivity](https://www.sciencedirect.com/science/article/pii/S0378475409000354); [Hines et al., weighted average derivative effects, original preprint](https://arxiv.org/abs/2109.13124).
