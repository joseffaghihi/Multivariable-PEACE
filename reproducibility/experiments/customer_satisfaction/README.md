# Customer satisfaction

The structural simulator fixes a causal graph with sales calls, interactions, economy, upgrade, product need, discount, usage, ad spend and bug variables. The primary usage coefficient is 3. All observations are synthetic.

From the repository root:

    python reproduce.py --study customer --mode smoke
    python reproduce.py --study customer --mode validate
    python reproduce.py --study customer --mode figures
    python reproduce.py --study customer --mode full

The full run covers three complementary analyses:

- The parent experiment supplies the simulator, intervention truth, predictive attribution, comparator targets and sensitivity analyses.
- flexible_estimation/ compares linear regression, additive splines, an interaction-capable smooth model and exploratory nonlinear-component selection without supplying the true response shapes.
- learned_mechanisms/ estimates parameters of supplied structural families and propagates interventions through the fitted mechanisms. It uses stronger modeling information.

For a total population response, signed derivatives are standardized over valid baseline adjustment variables before taking magnitudes. Supported conditional targets for discount and ad spend are reported separately. Deterministic reported bugs is excluded from generic observational derivative estimation. High predictive accuracy or a small error under a correct mechanism family does not remove these support restrictions.

METHODS.md gives the parent model and analysis design. Each follow-up has its own README and results. The root result map identifies the files used in the revised manuscript.
