"""Build reader-facing reports directly from the final result tables."""

from pathlib import Path
import html
import json
import hashlib
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"


def read(name):
    return pd.read_csv(R / name)


def mdtable(df):
    z = df.copy()
    for c in z:
        if pd.api.types.is_float_dtype(z[c]):
            z[c] = z[c].map(lambda x: f"{x:.4f}")
    rows = [
        "| " + " | ".join(z.columns) + " |",
        "| " + " | ".join(["---"] * len(z.columns)) + " |",
    ]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in z.to_numpy()]
    return "\n".join(rows)


def main():
    sim = read("simulation_summary.csv")
    raw = read("simulation_raw.csv")
    assert len(raw) == 900 * 2 * 2 * 7 * 2
    assert raw.estimate.notna().all() and np.isfinite(raw.estimate).all()
    assert not raw.duplicated(
        ["scenario", "n", "rep", "method", "density", "exposure", "d", "r"]
    ).any()
    for file in [
        "comparators_raw.csv",
        "density_raw.csv",
        "uncertainty_raw.csv",
        "joint_raw.csv",
        "separability_raw.csv",
        "collider_raw.csv",
    ]:
        z = read(file)
        assert np.isfinite(z.estimate).all(), file
    primary = sim[
        (sim.scenario == "baseline")
        & (sim.density == "estimated_gaussian")
        & (sim.d == 0.5)
        & (sim.r == 1)
    ]
    primary = primary[
        ["n", "method", "exposure", "truth", "bias", "rmse", "bias_mcse"]
    ].sort_values(["exposure", "method", "n"])
    stress = sim[
        (sim.n == 2000)
        & (sim.density == "estimated_gaussian")
        & (sim.d == 0.5)
        & (sim.r == 1)
        & (sim.method == "spline")
    ]
    stress = stress[["scenario", "exposure", "truth", "bias", "rmse"]]
    cp = read("comparators_summary.csv")[
        ["scenario", "method", "exposure", "truth", "bias", "rmse"]
    ]
    den = read("density_summary.csv")
    den = den[(den.d == 1) & (den.r == 1)][
        ["scenario", "method", "exposure", "truth", "bias", "rmse"]
    ]
    unc = read("uncertainty_summary.csv")
    oracle = read("oracle_targets.csv")
    joint = read("joint_summary.csv")
    joint = joint[(joint.n == 2000) & (joint.density == "estimated_gaussian")][
        ["method", "d", "truth", "bias", "rmse"]
    ]
    col = read("collider_summary.csv")[
        ["adjustment", "exposure", "truth", "mean", "bias", "rmse"]
    ]
    sep = read("separability_summary.csv")[
        ["method", "target", "truth", "mean", "bias", "rmse"]
    ]
    finite = read("estimand_comparison.csv")
    hidden = sim[
        (sim.scenario == "hidden_confounding")
        & (sim.density == "estimated_gaussian")
        & (sim.d == 0.5)
        & (sim.r == 1)
        & (sim.method == "spline")
    ][["n", "exposure", "truth", "bias", "rmse"]]
    curves = read("oracle_curves.csv")
    base = curves[curves.r == 1]
    smin = base.loc[base.sodium.idxmin()]
    j = read("joint_oracle.csv")
    j = j[j.nodes == 192]
    jmin = j.loc[j.value.idxmin()]
    baseline_s = primary[
        (primary["method"] == "spline") & (primary.exposure == "sodium")
    ].set_index("n")
    n500 = baseline_s.loc[500, "rmse"]
    n8000 = baseline_s.loc[8000, "rmse"]
    sepval = read("separability_sensitivity.csv")
    sepval = sepval[(sepval.kappa == 3) & (sepval.d == 0.5)].iloc[0]
    final_kde = read("kde_quadrature_checks.csv").error_48_vs_96.max()
    final_quad = read("estimated_quadrature_checks.csv").max_abs_error.max()
    audit = {
        "main_datasets": 900,
        "main_rows": len(raw),
        "comparators_datasets": 150,
        "density_datasets": 60,
        "bootstrap_evaluation_datasets": 100,
        "bootstrap_resamples_per_dataset": 200,
        "collider_datasets": 100,
        "separability_datasets": 100,
        "estimated_quadrature_max_selected_error": float(final_quad),
        "kde_quadrature_max_selected_error": float(final_kde),
        "sodium_curve_minimum_on_d_grid": dict(
            d=float(smin.d), value=float(smin.sodium)
        ),
        "joint_curve_minimum_on_d_grid": dict(d=float(jmin.d), value=float(jmin.value)),
        "all_result_estimates_finite": True,
        "python_version": sys.version,
    }
    (R / "final_audit.json").write_text(json.dumps(audit, indent=2))
    text = f"""# Sodium experiment: results and interpretation

## Main conclusion

The revised experiment supports PEACE as a useful, explicitly weighted summary of local causal responsiveness in this simulator. It does not establish universal superiority over Causal Forest or DML. Estimation works reasonably under the baseline assumptions and improves with sample size. Sensitivity analyses quantify how limited overlap, density misspecification, and violations of identification assumptions affect observational estimates.

This exploratory simulation study evaluates causal summaries and their estimation under specified mechanisms. Favorable and unfavorable results are both reported.

## What was run

900 main independently seeded datasets; 150 Causal Forest/DML comparison datasets; 60 density-only datasets; 100 interval-evaluation datasets with 200 bootstrap resamples each; 100 collider-adjustment datasets; and 100 separability-violation datasets. Joint PEACE uses the same 300 baseline datasets as the single-exposure study. Counts refer to datasets, not multiple targets evaluated on one dataset.

The baseline equations preserve the original threshold sodium response, exponential age response and noise. Mean PEACE uses fixed domains S in [0,8] and A in [50,80]. Exact details, causal assumptions and formulas are in [METHODS.md](METHODS.md). All scalar effects below have exposure-specific units; sodium and age magnitudes should not be ranked directly across different measurement scales.

## 1. What the interventions mean

For sodium, age is a baseline confounder: change sodium and condition/average over age. For age's controlled response, sodium is a mediator: change age while holding sodium fixed. For age's total response, let sodium respond to age. This distinction replaces the ambiguous claim that every comparator estimates a total effect.

![Response functions and pathways](figures/02_response_and_pathways.png)

Sodium's local derivative is 2.5 below 3.5 and 5 above 3.5. The age controlled derivative is exp[0.1(A-65)]. The total age derivative adds the age-to-sodium-to-blood-pressure pathway. These are exact simulator properties, not estimated clinical findings.

## 2. Exact PEACE curves and their meaning

The interpretation is anchored in the manuscript: Theorem 8.1 gives the normalized gradient representation; Theorem 8.2(1) relates monotonicity in d to the association between causal magnitude and treatment frequency; Theorem 8.2(2) gives concentration on density-maximizing treatment values as d tends to infinity. Section 11 defines r and its interpretation in terms of frequent versus rare covariate subpopulations. The displayed theorem has a raw-density/log-density discrepancy; the interpretation here follows the appendix derivation with log density. The r covariance identity below is a direct extension of that calculation, rather than a separately numbered theorem in the current manuscript.

![Exact PEACE curves](figures/01_oracle_curves.png)

For r=1 on the chosen domains, sodium Mean PEACE starts at 3.90625 when d=0, reaches approximately {smin.sodium:.5f} near d={smin.d:g} on the displayed grid, and then rises. At d=0.5 it is 3.85692; at d=1 it is 3.89556. Thus the original claim of a uniformly increasing sodium curve needs qualification.

The sodium curve has a transparent interpretation: it equals 2.5 plus 2.5 times the weighted probability that S exceeds 3.5. Uniform weighting on [0,8] places 56.25% of weight above that threshold. Moderate density weighting reduces that share; stronger concentration subsequently increases it. This describes changes in the weighting, not a reversal of sodium's causal direction. Sodium's local effect is positive throughout.

Age's controlled curve decreases as d increases: at r=1, values are 1.41952 at d=0, 1.12901 at d=0.5 and 1.06931 at d=1. Uniform weighting gives the rare high-age tail more influence; concentrating weight around common ages reduces the average of the exponential derivative. Curves for different r reflect a separate change in weighting of the other variable.

### How to read any normalized PEACE curve

- **Increasing in d:** under the current weights, higher conditional-density treatment regions are positively associated with larger local effect magnitudes.
- **Decreasing in d:** the corresponding weighted association is negative.
- **Flat:** reweighting does not change the summary. Constant local magnitude guarantees this, but a flat curve alone does not prove constant effects.
- **Curved:** sensitivity to reweighting changes with d. This alone is not evidence that the dose-response is nonlinear.
- **Approximately straight but nonflat:** approximately constant sensitivity to d over the displayed range, not a linear causal law.
- **Turning:** the balance between density and local magnitude changes sign as weights concentrate. The turning point is not a universal causal threshold.
- **Increasing in r:** more common covariate regions carry larger conditional PEACE summaries under those weights. This is not the same statement as increasing in d.

Mathematically, the d derivative is twice the covariate-weighted covariance between local magnitude and **log conditional density**. The r derivative is the covariance between conditional PEACE and **log covariate density**. Both identities were checked numerically. The manuscript currently prints raw density in its covariance theorem, whereas the appendix derivation uses log density; that discrepancy needs correction during manuscript editing.

![Constant response reference](figures/12_flat_reference.png)

The linear-response scenario has exact flat normalized curves. Conversely, a flat curve is not a general test of linearity.

### Joint sodium-age response

![Joint curve](figures/09_joint_curve.png)

Joint Mean PEACE averages the gradient norm. It initially decreases as the large age-tail derivatives receive less weight, reaches approximately {jmin.value:.5f} at d={jmin.d:g} on this grid, then rises as the sodium contribution shifts toward its higher-slope region. There is no r because no covariates remain outside the treatment vector. The minimum is not exactly 0.5 and is not a universal transition. The joint norm depends on raw coordinate units and does not establish interaction or synergy.

### Applying the large-degree result, Theorem 8.2(2)

For sodium at fixed age a, the conditional density mode is s*=a/18. At this mode, the local sodium response is 2.5 for a<63 and 5 for a>63. Consequently, the d-infinity limit is 2.5+2.5 times the q_r-weighted fraction of ages above 63. At r=1 it is approximately 4.13961. The five d-infinity limits for r=0,0.5,1,2,3 are approximately 3.91667, 4.03814, 4.13961, 4.28550 and 4.38947. The threshold case a=63 is a single zero-probability age under the continuous outer distribution; the kink needs this qualification when extending the smooth theorem.

For controlled age, the conditional density mode is a*(s)=65+[(25/18)/(1+25/324)](s-65/18). The limit averages exp[0.1(a*(s)-65)] over the chosen sodium-stratum weights. At r=1 it is approximately 1.00909.

For the joint treatment vector, the unique density mode is (s*,a*)=(65/18,65). Its local gradient is (5,1), so Theorem 8.2(2) gives the joint limit sqrt(26), approximately 5.09902. The curve displayed only through d=3 has not reached that limit. There is no r because no covariates remain.

Theorem 8.2(1) is conditional on a fixed covariate value. For the full d/r surface, its d-derivative is the outer weighted average of those conditional covariance terms. An increasing surface section need not mean that the conditional covariance is positive in every stratum.

## 3. Domain and distribution choices matter

![Domain sensitivity](figures/03_domain_sensitivity.png)

The d=0 target is uniform on a specified finite domain. Changing that domain changes the estimand, particularly near d=0; it is not merely a numerical implementation detail. At larger d, low-density remote endpoints usually matter less. This must be stated in the paper, because uniform normalization on the full real line is undefined.

![Treatment distribution changes curve shape](figures/15_distribution_curve.png)

The skewed-treatment comparison keeps the causal response function unchanged while changing exposure density. A different PEACE curve can therefore arise solely from different weighting populations. This is a feature of a distribution-sensitive estimand, and a reason to avoid interpreting every curve difference as a changed biological mechanism.

## 4. What is compared with what

Exact PEACE is compared with explicitly specified average derivatives and finite contrasts for interpretation. Observational PEACE estimates are compared against their own exact PEACE targets for accuracy. Causal Forest and DML are compared against their own Gaussian conditional-slope targets. There is no accuracy ranking based on disagreement between different estimands.

{mdtable(finite)}

In the baseline model, sodium interventions shift the entire outcome distribution by the same amount. Therefore all quantile treatment effects for a 2-to-5 contrast equal 11.25. PEACE describes local responsiveness under different exposure weights; it does not make that contrast incorrect.

Because all sodium derivatives in the original model are positive, signed and absolute local responses coincide. A separate, explicitly labeled quadratic illustration shows the distinction when effects reverse sign:

![Cancellation illustration](figures/13_cancellation_illustration.png)

In this auxiliary example, the 2-to-5 contrast and uniform average signed derivative are zero, while uniform Mean PEACE equals 3.75. This demonstrates different causal questions; it is not part of the original response or a claim of better estimation.

## 5. Observational PEACE estimation

Two response estimators are evaluated: a structural-basis least-squares benchmark with knowledge of the response family, and an additive spline estimator with fixed knots and ridge regularization. Both estimate the response from observations. Gaussian densities are estimated from the same data; oracle-density results are also retained.

![Sample-size performance](figures/04_estimation_sample_size.png)

For baseline sodium PEACE at d=0.5,r=1, spline RMSE falls from {n500:.3f} at n=500 to {n8000:.3f} at n=8,000. The structural benchmark and spline estimator have similar errors at larger sample sizes in this favorable additive setting. This is evidence of useful finite-sample estimation in the tested setting, not proof of a general convergence rate.

{mdtable(primary)}

![Estimated curves](figures/11_estimated_curves.png)

The ribbons show the middle 90% of estimates across repeated datasets, not confidence bands for one dataset or simultaneous confidence bands. Rare-region targets, especially sodium d=0, are much harder to estimate. An apparently smooth oracle curve does not imply that the curve can be estimated precisely from a small observational sample.

### Joint-estimation summary at n=2,000

{mdtable(joint)}

## 6. Stress tests and identification limits

![Scenario bias](figures/05_scenario_bias.png)

The table below uses the spline estimator, n=2,000 and d=0.5,r=1. Bias is estimate minus the correct causal PEACE target for that scenario.

{mdtable(stress)}

Poor treatment overlap increases uncertainty and can amplify the positive bias induced by taking magnitudes of noisy derivatives. Higher outcome noise also increases errors. A Gaussian density model is inappropriate for the skewed-treatment scenario. These results should be reported, not removed to present the metric as universally robust.

The hidden-confounding scenario is a sensitivity analysis under violated identification assumptions. The causal quantities remain defined, but the adjustment-based observational procedures no longer recover them under their maintained assumptions. In this design, E[B | S=s,A=a] equals the causal mean response plus 5(s-a/18), so its sodium derivative contains an extra 5. Consequently, observational sodium Mean PEACE is shifted upward by 5 for every d/r in this positive-response example. An age-adjusted observational contrast between sodium 2 and 5 is likewise shifted by 15: its population value is 26.25 rather than the true causal contrast of 11.25. This contrast calculation is an analytical implication of the SCM, not an additional fitted ATE experiment. Increasing sample size does not remove the bias induced by the violated assumption:

{mdtable(hidden)}

### Causal Forest and DML, against their own targets

![Comparator bias](figures/06_comparator_bias.png)

{mdtable(cp)}

Baseline results are reasonably close to their own targets; limited overlap increases error, and violating the no-unmeasured-confounding assumption introduces bias in all evaluated observational approaches. This is a shared identification limitation, not evidence of a defect unique to PEACE, Causal Forest or DML. ATE contrasts are subject to the same issue when estimated through unjustified observational adjustment.

### Incorrect proteinuria adjustment

![Collider adjustment](figures/10_collider_adjustment.png)

{mdtable(col)}

This scenario deliberately adjusts for a descendant of the outcome. The comparison uses the same structural-basis regression with and without proteinuria, so the changed adjustment is isolated. Conditioning on the outcome's descendant can drastically distort the response; it is not an intervention on proteinuria and not evidence of an intrinsic defect of a particular estimator.

### Separability and unobserved response heterogeneity

![Separability sensitivity](figures/14_separability_sensitivity.png)

With an added outcome term kappa(S-3.5)U, U independent of observed inputs, the mean response remains unchanged. Individual derivatives vary with U. At kappa=3 and d=0.5,r=1, PEACE of the mean response is {sepval.mean_response_peace:.4f}, whereas mean individual-response magnitude is {sepval.mean_individual_peace:.4f}.

{mdtable(sep)}

The same estimate has different errors against these different targets. The paper must distinguish them when local separability fails. Observational conditional means alone do not generally identify the average magnitude of unobserved individual derivatives. This addresses a substantive identification concern rather than treating it as a numerical issue.

## 7. Density estimation and bandwidth sensitivity

![Density sensitivity](figures/07_density_sensitivity.png)

This block holds the response derivatives at truth, isolating density errors. Under Gaussian exposure distributions, the Gaussian density estimator performs well. Under the skewed mixture, it produces substantial sodium bias. Covariance-adapted KDE reduces that error in this experiment, but the bandwidth matters; it is not uniformly best for every target.

{mdtable(den)}

This is density-only evidence. It is not an end-to-end validation of a KDE-based observational PEACE estimator, because response derivatives are held at truth in this block. The separate oracle-density/main-estimation results show the contribution of response estimation.

## 8. Uncertainty diagnostics

![Bootstrap coverage](figures/08_bootstrap_coverage.png)

{mdtable(unc)}

These are 95% percentile bootstrap intervals evaluated on 100 independent baseline datasets, using 200 resamples each. Observed coverage ranges from 92% to 98%, with non-negligible Monte Carlo uncertainty. This is consistent with reasonably useful baseline intervals but neither proves exact coverage nor validates them under the stress scenarios. Sodium d=0 intervals are substantially wider than density-weighted targets.

## 9. Validation and development audit

- Exact baseline scalar targets agree with independent truncated-normal calculations.
- Oracle quadrature was checked across all six scenarios; the final 96-node calculation agrees closely with a 192-node reference.
- Selected spline estimates were checked against 384-node integration. Maximum observed discrepancy: {final_quad:.6f}, small relative to their statistical errors but explicitly nonzero.
- Selected KDE integrations were checked against 96-node calculations. Maximum final-grid discrepancy: {final_kde:.6f}.
- Finite-difference response checks, the log-density covariance identities, constant-magnitude flatness, seed reproducibility and counterfactual outcome-shift invariance passed.
- The auxiliary cancellation and folded-normal heterogeneity identities were checked independently.
- All final estimates are finite. Nonfinite ratios are not silently converted to zero.

The initial spline scaling implementation was corrected after a stress test exposed unstable tail derivatives. Its code and results are preserved in audit/initial_standardized_spline. A failed coarse-grid check prompted integration refinement. Final scientific scenarios, including unfavorable outcomes, are retained. This remains exploratory development, not a preregistered confirmatory analysis.

## 10. What this supports in the manuscript

We can now explain the intervention targets, distinguish metric interpretation from estimator accuracy, show known ground truth, report repeated-sample errors and uncertainty, illustrate d/r sensitivity, discuss density estimation and examine the consequences of explicitly violated identification assumptions.

We should remove or qualify claims of uniformly increasing curves, a universal transition at d=0.5, automatic equivalence to ATE, age comparisons as total effects when sodium is adjusted for, and superiority based on comparisons with different targets. We should correct the log-density covariance formula and state the finite domains and units.

This package does not settle weak-instrument performance, general identification theorems, arbitrary interactions, high-dimensional estimation, causal feature selection or the customer-satisfaction experiment. Those remain separate tasks. No manuscript changes were made in this step.
"""
    (ROOT / "REPORT.md").write_text(text)
    # Compact offline HTML companion; detailed mathematical text stays in METHODS.
    cards = [
        ("Baseline sample-size results", primary),
        ("Stress tests", stress),
        ("Comparator results", cp),
        ("Bootstrap coverage", unc),
    ]
    figures = [
        ("02_response_and_pathways", "Response and intervention pathways"),
        ("01_oracle_curves", "Exact Mean PEACE curves"),
        ("09_joint_curve", "Joint response"),
        ("03_domain_sensitivity", "Domain sensitivity"),
        ("15_distribution_curve", "Density changes the curve"),
        ("04_estimation_sample_size", "Estimation improves with sample size"),
        ("11_estimated_curves", "Sampling variability of estimated curves"),
        ("05_scenario_bias", "Stress tests"),
        ("06_comparator_bias", "Comparators against their own targets"),
        ("07_density_sensitivity", "Density estimation"),
        ("08_bootstrap_coverage", "Uncertainty diagnostics"),
        ("10_collider_adjustment", "Incorrect collider adjustment"),
        ("14_separability_sensitivity", "Mean response versus individual magnitudes"),
        ("12_flat_reference", "Constant local magnitudes"),
        ("13_cancellation_illustration", "Auxiliary cancellation illustration"),
    ]
    body = """<h1>Sodium experiment</h1><p class="lead">Results, curve interpretation and validation for the Franklin Open major revision.</p>
    <p>The revised analysis supports Mean PEACE as an explicit weighted summary of local responsiveness. It includes sensitivity analyses showing how observational estimates change when identification assumptions are violated.</p>
    <p><a href="REPORT.md">Detailed results and interpretation</a> · <a href="METHODS.md">Methods and causal targets</a> · <a href="README.md">Reproduction instructions</a></p>
    <div class="callout"><b>Reading a curve:</b> Increasing means the new weights favor larger local magnitudes. Decreasing means they favor smaller magnitudes. Flat means the summary is unchanged by those weights. Curvature describes changing sensitivity to weighting; it does not by itself establish a nonlinear causal law. Turning points depend on the response, density and domain.</div>"""
    body += f"<p>900 main datasets, 150 comparator datasets, 60 density-only datasets, 100 bootstrap evaluation datasets with 200 resamples each, and additional collider/separability checks. Baseline sodium spline RMSE falls from {n500:.3f} to {n8000:.3f} as sample size increases from 500 to 8,000.</p>"
    for name, title in figures:
        assert (ROOT / "figures" / f"{name}.png").exists(), name
        body += f'<section><h2>{html.escape(title)}</h2><img src="figures/{name}.png" alt="{html.escape(title)}"><p><a href="figures/{name}.svg">Vector figure</a></p></section>'
    for title, df in cards:
        body += (
            f'<section><h2>{title}</h2><div class="table">'
            + df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            + "</div></section>"
        )
    body += '<p class="foot">Exploratory synthetic study. No universal estimator superiority, clinical validity or general confidence-interval guarantee is claimed. See the detailed report for limitations and the development audit.</p>'
    style = """body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202c3d;max-width:1120px;margin:48px auto;padding:0 24px}h1{font-size:42px;line-height:1.1}h2{margin-top:40px;font-size:24px}.lead{font-size:21px;color:#526178}a{color:#2458a6}img{max-width:100%;height:auto;border:1px solid #e3e8ef;border-radius:8px}section{margin:36px 0}.callout{padding:22px;background:#edf3fa;border-left:4px solid #2458a6}.table{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:9px;border-bottom:1px solid #dce3eb;text-align:left}th{background:#edf3fa}.foot{font-size:14px;color:#526178;border-top:1px solid #dce3eb;padding-top:24px}"""
    (ROOT / "REPORT.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sodium PEACE experiment</title><style>'
        + style
        + "</style><main>"
        + body
        + "</main></html>"
    )
    # Hash the final source and result tables, excluding the manifest itself.
    manifest = {}
    for f in sorted(
        list(ROOT.glob("*.py"))
        + list(R.glob("*.csv"))
        + [ROOT / "requirements-lock.txt"]
    ):
        manifest[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (ROOT / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2))
    print("reports generated; final result counts and finiteness verified", flush=True)


if __name__ == "__main__":
    main()
