"""Generate reader-facing reports only from final results; audit is excluded."""

from pathlib import Path
import hashlib, html, json, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"


def read(name):
    return pd.read_csv(R / name)


def table(z):
    z = z.copy()
    for c in z:
        if pd.api.types.is_float_dtype(z[c]):
            z[c] = z[c].map(lambda x: f"{x:.4f}")
    return "\n".join(
        [
            "| " + " | ".join(z.columns) + " |",
            "| " + " | ".join(["---"] * len(z.columns)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in z.to_numpy()]
    )


def mcse(file, keys):
    df = read(file)
    df["error"] = df.estimate - df.truth

    def summarize(z):
        err = z.error.to_numpy()
        rmse = np.sqrt(np.mean(err**2))
        n = len(err)
        return pd.Series(
            {
                "replicates": n,
                "truth": z.truth.iloc[0],
                "mean_estimate": z.estimate.mean(),
                "bias": err.mean(),
                "bias_mcse": np.std(err, ddof=1) / np.sqrt(n),
                "rmse": rmse,
                "rmse_mcse": np.std(err**2, ddof=1) / (2 * rmse * np.sqrt(n))
                if rmse > 1e-12
                else 0,
            }
        )

    out = (
        df.groupby(keys, sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    out.to_csv(R / file.replace("_replicates.csv", "_summary_mcse.csv"), index=False)
    return out


def main():
    validation = json.loads((R / "validation.json").read_text())
    assert validation["passed"]
    expected = {
        "observational_replicates.csv": 23040,
        "comparator_replicates.csv": 1260,
        "joint_replicates.csv": 2880,
        "bootstrap_intervals.csv": 120,
        "null_linear_interval_replicates.csv": 400,
        "spline_sensitivity_replicates.csv": 960,
    }
    for file, count in expected.items():
        z = read(file)
        assert len(z) == count, (file, len(z))
        assert np.isfinite(z.select_dtypes("number").to_numpy()).all(), file
        assert not z.isna().any().any(), file
    raw = read("observational_replicates.csv")
    assert not raw.duplicated(["scenario", "n", "rep", "variable", "method", "d"]).any()
    cp = mcse("comparator_replicates.csv", ["scenario", "variable", "method"])
    obs = mcse(
        "observational_replicates.csv", ["scenario", "n", "variable", "method", "d"]
    )
    joint = mcse("joint_replicates.csv", ["n", "pair", "d", "method"])
    single = read("oracle_single.csv")
    oracletable = (
        single[single.d.isin([0, 0.5, 1])]
        .pivot(index="variable", columns="d", values="peace")
        .reset_index()
    )
    oracletable.columns = ["Variable", "d=0", "d=0.5", "d=1"]
    names = [
        "Sales calls",
        "Interactions",
        "Economy",
        "Last upgrade",
        "Product need",
        "Discount",
        "Monthly usage",
        "Ad spend",
        "Bugs faced",
        "Bugs reported",
    ]
    oracletable = oracletable.set_index("Variable").loc[names].reset_index()
    est = obs[
        (obs.scenario == "outcome_noise")
        & (obs.n == 5000)
        & (obs.d == 0.5)
        & obs.method.str.endswith("estimated density")
    ][["variable", "truth", "mean_estimate", "bias", "rmse", "bias_mcse"]]
    est = est.set_index("variable").loc[names[:-1]].reset_index()
    cpt = cp[
        (cp.scenario == "original")
        & cp.variable.isin(
            ["Sales calls", "Product need", "Last upgrade", "Discount", "Ad spend"]
        )
    ][["variable", "method", "truth", "mean_estimate", "bias_mcse", "rmse"]]
    jtab = joint[
        (joint.n == 5000) & (joint.d == 0.5) & (joint.method == "Fitted joint density")
    ][["pair", "truth", "mean_estimate", "rmse", "bias_mcse"]]
    boot = read("bootstrap_summary.csv")
    boot["coverage_mcse"] = np.sqrt(boot.coverage * (1 - boot.coverage) / 40)
    null = json.loads((R / "null_linear_interval_summary.json").read_text())
    pred = json.loads((R / "prediction_metrics.json").read_text())
    hidden = obs[
        (obs.scenario.isin(["outcome_noise", "hidden_confounding"]))
        & (obs.n == 5000)
        & (obs.variable == "Product need")
        & (obs.d == 0.5)
        & obs.method.str.endswith("oracle weight")
    ][["scenario", "truth", "mean_estimate", "bias", "bias_mcse"]]
    from experiment import need_deriv

    cov = read("theorem_interpretation.csv")
    mode = float(need_deriv(np.array([0.2]))[0])
    # Re-render the sample-size figure on a common horizontal scale.
    from experiment import plt, savefig

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True)
    for ax, n in zip(axes, [1000, 5000]):
        z = (
            obs[
                (obs.scenario == "outcome_noise")
                & (obs.n == n)
                & (obs.d == 0.5)
                & obs.method.str.endswith("estimated density")
            ]
            .set_index("variable")
            .loc[names[:-1]]
        )
        ax.barh(names[:-1], z.rmse)
        ax.set(
            title=f"Outcome-noise sensitivity: n={n}, 80 replicates",
            xlabel="RMSE of Mean PEACE at d=0.5",
        )
    fig.tight_layout()
    savefig("06_observational_rmse")
    text = f"""# Customer satisfaction experiment: results and interpretation

This standalone component reconstructs the original synthetic experiment, defines its causal targets explicitly, and evaluates observational estimates against known truth. It includes 16 figures, repeated single and joint estimation, causal forest/DML comparisons, a held-out SHAP analysis, density and smoothing checks, and uncertainty audits. The manuscript, sodium experiment and estimation document remain separate and unchanged.

The corrected experiment supports a clear mathematical illustration of PEACE. It also shows substantial limitations of the flexible observational estimator, especially for discount, interactions and a causal-null proxy. It does not support a claim that PEACE automatically estimates causal importance more accurately than causal forest or DML.

## 1. What is being intervened on?

For each variable V, we replace its assignment and let its descendants respond according to the SEM. The response is h_V(v)=E[Y_do(V=v)]. This is a total mean intervention response, not the partial derivative of the outcome equation holding all other variables fixed. For a pair, both assignments are replaced simultaneously. A pair containing an ancestor and its mediator blocks that mediated path.

The expanded primary outcome is

```
Y = 5 ProductNeed + 3 MonthlyUsage + 2 Economy + Discount
    − .2 BugsFaced + .2 SalesCalls + .3 Interactions + 8/(LastUpgrade+1).
```

The manuscript prints a usage coefficient of 30 instead of the simulator's 3. We retain 3 for primary replication and show 30 separately in oracle figures and tables. The displayed usage→bugs edge is also absent from the simulator: bugs are independent. These discrepancies require explicit resolution before manuscript integration.

![Oracle responses](figures/02_total_responses.png)

Ad spend and bugs reported have no outgoing paths to the outcome, so their total response is constant. Upgrade has a decreasing nonlinear response. Bugs faced has a decreasing linear response. The other six variables have positive derivatives in this model. A null variable should not be described as having a positive causal effect merely because its negative component vanishes.

## 2. Exact PEACE and the paper's interpretation theorems

We apply P_d(h)=∫|h′|f^(2d)/∫f^(2d) on fixed domains. Pair analyses replace |h′| by the joint gradient norm. At d=0 the domain receives uniform volume weight; at d=.5 it receives normalized natural-density weight. There is no remaining outer covariate context after defining these total responses, so the original target has no nontrivial r dimension.

{table(oracletable)}

These values retain native exposure units. Their ranking is not a unit-invariant ranking of causal importance. A separate coordinate-scale table shows how pair rankings change under fixed-reference SD rescaling.

![Oracle single-variable curves](figures/01_oracle_single.png)

The interpretation follows Theorem 8.1 and Theorem 8.2, using the log-density calculation in the appendix. For J=|h′| or ||∇h||,

```
d P_d / d d = 2 Cov_p_d(J, log f),       p_d ∝ f^(2d).
```

The covariance identity uses log reference density. The numerical covariance check here agrees with differentiation to {validation["max_covariance_identity_error"]:.2g}.

**Product need:** PEACE decreases slightly from {cov.peace.iloc[0]:.6f} at d=0 to {cov.peace.iloc[-1]:.6f} at d=3. Its weighted covariance with log density is negative throughout the displayed range. Increasing d favors common need values whose local magnitudes are slightly smaller. This says nothing about a negative causal direction: the derivative is positive throughout. Its density has a unique mode at .2, and Theorem 8.2(2) gives the limiting local magnitude {mode:.6f}. The change over d∈[0,3] is very small and is shown on a magnified scale in the diagnostic figure.

**Last upgrade:** f is constant on (0,10), so the covariance is exactly zero and every degree gives 8/11=0.727273. Its nonlinear response therefore has a flat PEACE curve. The current manuscript's claim that a decreasing curve establishes a stronger effect at infrequent upgrade values is unsupported by this generator. Low upgrade values have larger derivative magnitudes, but they are not less frequent under its uniform distribution.

![Theorem-based diagnostics](figures/13_theorem_interpretation.png)

Theorem 8.2(2) also explains the flat upgrade limit: the entire interval is a density-maximizing plateau. In the separate beta-distribution sensitivity, keeping the response unchanged but taking upgrade=10×Beta(2,2), the curve decreases toward the derivative magnitude at its unique mode 5, namely 2/9. This is a different exposure distribution, explicitly labelled as such.

![Distribution sensitivity](figures/03_upgrade_distribution.png)

Flat curves can result from constant local magnitudes or a constant density. Curvature describes changing sensitivity to density reweighting; it does not alone establish response nonlinearity. Direction comes from the signed components, and joint PEACE is not an interaction coefficient.

## 3. Joint interventions and interaction

Six pairs cover a mediator intervention, an ancestor intervention, a second mediator path, nonlinear upgrade, and two null features. Three are additionally estimated from observational samples. We do not claim to have reproduced every original pair curve.

For need+usage, fixing usage removes need→usage, yielding ∂h/∂need=5−.25σ′(need), ∂h/∂usage=3. For calls+need, fixing need removes calls→need, leaving the calls derivative .5. These are different from combining two separate total-response gradients.

![Joint curves](figures/04_joint_curves.png)

Need+usage decreases slightly with d while need+discount increases slightly. Their different covariance signs explain the opposite directions. Economy+upgrade remains flat because the joint density is uniform. Usage+upgrade is also flat: upgrade is uniform and independent, and the usage marginal factor cancels because the gradient norm depends only on upgrade. Ad spend+reported bugs is identically zero.

Every selected original joint response is additive in its two intervention coordinates. For example h(need,usage)=g(need)+3usage, so its mixed partial is zero despite positive joint PEACE. An optional, separate positive control adds 10(need−.2)(usage−.515) to the outcome. Its mixed partial is 10 and its four-intervention interaction contrast is .14, versus zero in the original SEM. This distinguishes joint responsiveness from interaction without relabelling either quantity.

![Interaction positive control](figures/14_interaction_positive_control.png)

## 4. Observational estimation: what worked and what did not

The flexible estimator uses parent-adjusted additive cubic splines and scale-adapted, boundary-reflected KDE. A separate benchmark estimates outcome coefficients in the correct structural basis and propagates them through known assignment equations. The latter uses substantially more prior information; its near-exact noiseless recovery is expected because the outcome is deterministic in that basis.

The main sampling check retains the original noiseless model and adds an explicitly separate outcome-noise sensitivity. The following table is for independent outcome noise SD=.5, n=5000, 80 datasets, d=.5, with estimated density. Bias Monte Carlo errors quantify simulation precision.

{table(est)}

![Sample-size estimation errors](figures/06_observational_rmse.png)

Need and calls are estimated reasonably in these settings. Upgrade retains smoothing bias with five knots. Discount and ad spend are particularly difficult; estimating derivative magnitudes can create substantial upward bias from noisy signed derivatives. A causal null is not automatically estimated as zero. Economy, bugs and interactions also show magnitude-estimation bias; these unfavorable results are retained.

This does not contradict the definition or theorems of PEACE. It limits the practical claims we can make for this particular estimator. The current experiment does not establish robust observational feature ranking or a generally reliable PEACE estimation pipeline.

Joint observational results, outcome noise SD=.5, n=5000, 60 datasets, d=.5:

{table(jtab)}

![Joint estimation](figures/11_joint_estimation.png)

The sparse corners of fixed rectangular domains are especially difficult at d=0. Density weighting helps some targets in this example but is a change of estimand, not a license to select whichever degree produces the best-looking result.

## 5. Overlap, density, domains and smoothing

![Residual treatment variation](figures/15_overlap.png)

Interactions and ad spend have very narrow residual variation after parent adjustment. Discount also has a small absolute residual variance. Bugs reported is exactly determined by its parents: its generic adjusted treatment variance is zero. Its graph-implied causal null is retained in oracle and prediction illustrations, while it is excluded from the generic regression/forest/DML benchmark.

Density-only checks hold upgrade's derivative at truth. KDE can nevertheless produce a nonflat estimated d curve around its truly flat population curve. Reflection reduces boundary leakage but does not eliminate sampling or bandwidth sensitivity. Those estimated trends must not be interpreted as established causal-frequency relationships.

![Density-only artifacts](figures/16_density_only_upgrade.png)

![Illustrative estimated curves](figures/07_estimated_curves_bandwidth.png)

Three-, five- and seven-knot checks expose smoothing sensitivity. Seven knots reduce upgrade bias in this setting; they can worsen magnitude error for other targets. We retain the five-knot primary analysis rather than silently selecting the best configuration per variable. Upgrade's d=0 value also changes from .727273 on [0,10] to .507937 on [.5,9.5], .4 on [1,9], and .296296 on [2,8]. Those are different declared domains, not conflicting estimates of the same target.

## 6. Causal forest and DML comparisons

These implementations estimate signed residual projection slopes using three-fold cross-fitting, sufficient parent sets, and continuous-treatment marginal-effect output. They are compared against their own exact population targets, not against PEACE magnitudes or unsupported default 0-to-1 contrasts.

For upgrade, the signed linear projection is −.421188, PEACE is .727273, and the do(10)−do(0) contrast is −7.272727. All three are compatible with the same nonlinear response. For need, Gaussian residual assignment makes the projection equal the natural average response derivative, approximately 5.162591.

Selected original-model results: n=3000, 30 datasets.

{table(cpt)}

![Comparator estimates against their own targets](figures/08_comparator_targets.png)

Calls, need and upgrade are recovered reasonably at these sample sizes. Discount, interactions and ad spend have large sampling variability. A noisy negative estimate of a small positive effect is not evidence that the method estimates the wrong causal quantity. Nor does the structural oracle's knowledge advantage establish PEACE estimator superiority.

## 7. Identification-assumption sensitivity

The latent-confounding variant preserves the exposure marginal and causal mean response but creates an unobserved need–outcome common cause. It predicts a +.5 bias in the observational need slope if the latent variable is ignored. The flexible need PEACE estimator shows the corresponding shift:

{table(hidden)}

Causal forest and DML also shift upward by approximately .47 in their need-slope estimates. The observable residual projection changes in this scenario and no longer identifies the unchanged causal projection: reported bias is against the causal target. Upgrade remains largely unaffected by this specific path. Ad spend remains a difficult noisy null. These results are sensitivity analysis under violated identification assumptions, shared across causal workflows, not a failure of PEACE as a mathematical metric.

![Identification sensitivity](figures/10_identification_sensitivity.png)

## 8. Predictive importance is a different question

XGBoost uses a 6000/2000/2000 train/validation/test split; the test set is not used for early stopping. Held-out R² is {pred["test_r2"]:.6f}, RMSE {pred["test_rmse"]:.6f}. Tree-path-dependent SHAP assigns substantial predictive importance to ad spend despite its zero intervention effect. This is compatible with proxy prediction; it should not be called an incorrect causal estimate by SHAP.

![Prediction versus intervention](figures/09_prediction_vs_intervention.png)

Discount and usage have small natural interquartile ranges, so their causal interquartile contrasts are small despite nonzero derivatives. The contrast panel is in outcome units and still measures a different quantity from mean absolute SHAP. Raw slope magnitudes, predictive attribution, and natural-range intervention contrasts should not share an unqualified importance ranking.

## 9. Uncertainty audit

For the flexible estimator, ordinary nominal 95% percentile bootstrap intervals use 120 draws on each of 40 independent n=3000 noisy-outcome datasets, d=.5, with density held at truth:

{table(boot)}

![Bootstrap coverage audit](figures/12_bootstrap_audit.png)

The ad-spend intervals never include its true zero in this audit. A nonnegative norm of a noisy derivative has nonregular behavior at zero; naive percentile intervals are not a validated uncertainty procedure for the null. Need/upgrade coverage also does not justify a blanket nominal-coverage claim, and 40 datasets leave substantial simulation uncertainty.

For a separate constant-derivative reference model for ad spend, we first construct an HC3 robust interval for the signed slope, then map the entire interval through absolute value. When the signed interval crosses zero, the magnitude interval starts at zero. Coverage is {null["coverage"]:.2%} across 400 independent datasets, Monte Carlo SE {null["coverage_MCSE"]:.2%}. This is useful evidence for that linear-response construction, not a general solution for nonlinear functional confidence intervals.

## 10. What this contributes to the revision

| Reviewer concern | Evidence here | Remaining scope |
|---|---|---|
| R2.1 / R3.1: interpretation and established effects | Explicit total responses, mediated-path accounting, distinct slopes/contrasts/PEACE | Integrate into the general response-function exposition later |
| R2.2 / R3.3: identification versus mathematical metric | Graph-based adjustment, positivity diagnostics, latent-confounding sensitivity | General identification and IV arguments remain separate |
| R2.3 / R3.4: controlled validation | Known truth, repeated bias/RMSE, joint benchmarks, CF/DML own-target checks | No claim of uniform practical superiority; flexible-estimator limitations remain |
| R2.4 / R3.2: d/r guidance | Fixed d grid, exact covariance interpretation, density/domain/scale checks | No universal optimal d; r is absent for this total-response target |
| R3.6: density error and bandwidth | Oracle versus estimated density, reflection, half/double bandwidth sensitivity | Does not settle mixed discrete–continuous or high-dimensional estimation |

This study separates predictive attribution, causal estimands and estimation accuracy. It does not include an instrument or test weak-instrument identification, and it does not establish universal estimator superiority.

## 11. Verification and development record

All final tables have the expected row counts and finite estimates. Original outcome algebra agrees to {validation["original_outcome_algebra_max_error"]:.2g}; common-random-number checks agree with the analytic total derivatives within {max(validation["simulator_derivative_errors"].values()):.2g}; oracle pair refinement differences are below {max(validation["pair_quadrature_refinement_errors"].values()):.2g}. The final knot/root-split estimated integrals agree under refinement within {validation["estimated_quadrature_max_difference"]:.2g} for the audited single fits and {validation["estimated_joint_quadrature_max_difference"]:.2g} for selected joint fits.

A preliminary global-grid check exposed a discrepancy of about 2.21 for discount. Its code/results are preserved in the audit folder. Final calculations use analytic spline derivatives, linear tail extrapolation and integration split at fitted knots and derivative roots. The final pass flag includes the estimated-integral checks. This is exploratory development; the record is not a preregistered confirmatory analysis. No manuscript integration was performed.
"""
    if (ROOT / "learned_mechanisms/REPORT.html").exists():
        notice = "**Follow-up:** [Learned-mechanism observational estimator](learned_mechanisms/REPORT.html) substantially improves point estimation under declared structural assumptions. Its results and limits are separate from the initial benchmarks below. The follow-up also updates the standalone estimation document; the main manuscript remains unchanged.\n\n"
        text = text.replace("\n\n", "\n\n" + notice, 1)
    if (ROOT / "flexible_estimation/REPORT.html").exists():
        text = text.replace(
            "\n\n",
            "\n\n**Latest follow-up:** [Linear and flexible estimation without exact structural forms](flexible_estimation/REPORT.html). Full estimates, separate supported targets, unknown-shape checks and limitations.\n\n",
            1,
        )
    (ROOT / "REPORT.md").write_text(text)
    figures = [
        (
            "02_total_responses",
            "Total intervention responses",
            "Descendants respond to intervention; a nonlinear response need not yield a nonflat d curve.",
        ),
        (
            "01_oracle_single",
            "Exact single-variable PEACE",
            "Solid: committed simulator coefficient 3. Dashed: printed coefficient 30, a separate oracle sensitivity.",
        ),
        (
            "13_theorem_interpretation",
            "Theorem-based curve interpretation",
            "Slopes follow the log-density covariance identity from the appendix.",
        ),
        (
            "03_upgrade_distribution",
            "Distribution and modal limits",
            "Original uniform upgrade stays flat. The beta distribution is a separate sensitivity case.",
        ),
        (
            "04_joint_curves",
            "Joint interventions",
            "Six selected pairs. Gradient norm measures joint responsiveness, not interaction.",
        ),
        (
            "05_local_joint_magnitudes",
            "Local joint gradient magnitudes",
            "Coordinates retain native units; joint rankings depend on scale.",
        ),
        (
            "14_interaction_positive_control",
            "Interaction positive control",
            "The original selected joint responses are additive; the added interaction is explicitly a different SEM.",
        ),
        (
            "06_observational_rmse",
            "Observational estimation errors",
            "Flexible derivative-magnitude estimation has substantial error for several variables.",
        ),
        (
            "11_joint_estimation",
            "Joint observational estimation",
            "Sixty datasets per sample size; sparse tails are difficult at d=0.",
        ),
        (
            "15_overlap",
            "Overlap diagnostics",
            "Reported bugs has zero residual variation; its bar uses a display floor.",
        ),
        (
            "16_density_only_upgrade",
            "Density-only artifacts",
            "A nonflat estimated curve can arise even with the true derivative of a flat-curve target.",
        ),
        (
            "07_estimated_curves_bandwidth",
            "Smoothing sensitivity",
            "Illustrative curves from one noisy-outcome sample; not confidence bands.",
        ),
        (
            "08_comparator_targets",
            "Causal forest and DML",
            "Signed estimates versus their own population projection targets. Error bars are replicate SD, not confidence intervals.",
        ),
        (
            "10_identification_sensitivity",
            "Identification-assumption sensitivity",
            "Hidden need–outcome confounding shifts observational estimates across causal workflows.",
        ),
        (
            "09_prediction_vs_intervention",
            "Predictive SHAP and causal contrast",
            "Held-out predictive attribution and intervention effects answer different questions.",
        ),
        (
            "12_bootstrap_audit",
            "Bootstrap coverage",
            "Naive percentile intervals are not validated at the null; only 40 coverage-evaluation datasets.",
        ),
    ]
    body = f'<h1>Customer satisfaction experiment</h1><p class="lead">Standalone analysis for the Franklin Open revision</p><p>Known-truth intervention responses, theorem-based interpretation, repeated estimation and uncertainty audits. All results are generated from the specified synthetic model.</p><p><a href="REPORT.md">Detailed findings</a> · <a href="METHODS.md">Methods and definitions</a> · <a href="README.md">Reproduction and audit</a></p><div class="callout"><b>Main findings:</b> Uniform upgrade produces a flat exact PEACE curve despite its nonlinear response. Joint gradient norms do not establish interaction. Flexible observational magnitude estimates can be noisy and upward biased; ordinary percentile bootstrap intervals miss the causal-null ad-spend effect. The analysis does not establish uniform estimator superiority.</div><p>Primary model: simulator usage coefficient 3, independent bugs. Printed coefficient 30 is retained as an oracle sensitivity. The primary analysis follows the simulator; the coefficient-30 variant is a separate sensitivity.</p>'
    for name, title, caption in figures:
        assert (ROOT / "figures" / f"{name}.png").exists()
        body += f'<section><h2>{html.escape(title)}</h2><img src="figures/{name}.png" alt="{html.escape(title)}"><p>{html.escape(caption)} <a href="figures/{name}.svg">Vector figure</a></p></section>'
    for title, z in [
        ("Oracle targets", oracletable),
        ("Flexible observational results: n=5000, d=.5, outcome noise SD=.5", est),
        ("Selected comparator results: original SEM, n=3000", cpt),
        ("Joint observational results: n=5000, d=.5", jtab),
        ("Bootstrap audit", boot),
    ]:
        body += f'<section><h2>{title}</h2><div class="table">{z.to_html(index=False, float_format=lambda v: f"{v:.4f}")}</div></section>'
    body += f"<p>Signed-linear ad-spend interval mapped through absolute value: coverage {null['coverage']:.2%} over 400 datasets. This is a model-specific reference, not a general PEACE uncertainty procedure.</p><p>All final numerical checks passed. See the detailed report for methodological limitations, validation coverage and the preserved development audit.</p>"
    if (ROOT / "learned_mechanisms/REPORT.html").exists():
        body = (
            '<div class="callout"><b>Mechanism-based estimation:</b> <a href="learned_mechanisms/REPORT.html">Learned-mechanism observational estimation</a> evaluates point accuracy under explicit model assumptions. The benchmark results are reported below.</div>'
            + body
        )
    if (ROOT / "flexible_estimation/REPORT.html").exists():
        body = (
            '<div class="callout"><b>Flexible estimation:</b> <a href="flexible_estimation/REPORT.html">Linear and flexible estimation without exact structural forms</a>: actual estimates, supported versus global targets, unknown-shape checks and precision limits.</div>'
            + body
        )
    style = 'body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#203044;max-width:1160px;margin:48px auto;padding:0 24px}h1{font-size:42px;line-height:1.15}h2{margin-top:36px;font-size:24px}.lead{font-size:22px;color:#53647a}a{color:#245c9d}img{max-width:100%;height:auto;border:1px solid #dde5ed;border-radius:7px}.callout{background:#edf3fa;border-left:4px solid #245c9d;padding:20px}section{margin:36px 0}.table{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce4ed}th{background:#edf3fa}'
    (ROOT / "REPORT.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Customer satisfaction — PEACE</title><style>'
        + style
        + "</style><main>"
        + body
        + "</main></html>"
    )
    sources = [
        ROOT / "experiment.py",
        ROOT / "README.md",
    ]
    provenance = {
        "repository": "https://github.com/joseffaghihi/Multivariable-PEACE",
        "primary_usage_coefficient": 3,
        "printed_equation_oracle_sensitivity_coefficient": 30,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sources": {
            str(p.relative_to(ROOT.parent.parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
    }
    (ROOT / "source-provenance.json").write_text(json.dumps(provenance, indent=2))
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and "audit" not in p.parts
        and "__pycache__" not in p.parts
        and p.name != "manifest-sha256.json"
    }
    (ROOT / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Reports built; counts, finiteness and numerical validation verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
