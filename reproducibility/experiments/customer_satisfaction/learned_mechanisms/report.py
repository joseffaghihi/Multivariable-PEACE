from study import (
    A,
    B,
    C,
    D,
    DS,
    E,
    FIG,
    I,
    L,
    M,
    N,
    R,
    RES,
    ROOT,
    calculate_curve,
    fit,
    generate,
    hashlib,
    json,
    logsumexp,
    norm,
    np,
    pd,
    plt,
    quad,
    stress,
    true_value,
    weighted,
)
import html, platform


def summary(raw, keys):
    raw = raw.copy()
    raw["error"] = raw.estimate - raw.truth

    def one(z):
        e = z.error.to_numpy()
        n = len(e)
        rmse = np.sqrt(np.mean(e * e))
        return pd.Series(
            {
                "replicates": n,
                "truth": z.truth.iloc[0],
                "mean": z.estimate.mean(),
                "bias": e.mean(),
                "bias_mcse": np.std(e, ddof=1) / np.sqrt(n),
                "rmse": rmse,
                "rmse_mcse": np.std(e * e, ddof=1) / (2 * rmse * np.sqrt(n))
                if rmse > 1e-12
                else 0,
            }
        )

    return raw.groupby(keys, sort=False).apply(one, include_groups=False).reset_index()


def table(z):
    z = z.copy()
    for c in z:
        if pd.api.types.is_float_dtype(z[c]):
            z[c] = z[c].map(lambda v: f"{v:.5f}")
    return "\n".join(
        [
            "| " + " | ".join(z.columns) + " |",
            "| " + " | ".join(["---"] * len(z.columns)) + " |",
        ]
        + ["| " + " | ".join(map(str, v)) + " |" for v in z.to_numpy()]
    )


def save(name):
    plt.savefig(FIG / (name + ".png"), dpi=175, bbox_inches="tight")
    plt.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close()


def theory_checks():
    t, w = quad(-1, 1, 200)
    theta = np.array([0.2, 1, 0.1, np.log(0.7)])
    errors = []

    def functional(theta, d):
        g = 2 * theta[0] * t + theta[1]
        lf = norm.logpdf(t, theta[2], np.exp(theta[3]))
        return weighted(g, lf, w, d)[0]

    for d in DS:
        sd = np.exp(theta[3])
        g = 2 * theta[0] * t + theta[1]
        lf = norm.logpdf(t, theta[2], sd)
        p = np.exp(np.log(w) + 2 * d * lf - logsumexp(np.log(w) + 2 * d * lf))
        score = np.column_stack(
            [
                np.zeros_like(t),
                np.zeros_like(t),
                (t - theta[2]) / sd**2,
                (t - theta[2]) ** 2 / sd**2 - 1,
            ]
        )
        gradJ = np.column_stack(
            [2 * t, np.ones_like(t), np.zeros_like(t), np.zeros_like(t)]
        )
        analytic = p @ gradJ + 2 * d * ((p * g) @ score - (p @ g) * (p @ score))
        finite = []
        for j in range(4):
            plus = theta.copy()
            minus = theta.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            finite.append((functional(plus, d) - functional(minus, d)) / (2e-6))
        errors.append(float(np.max(np.abs(analytic - finite))))
    mapped = True
    for low, high in [(-2, 1), (-3, -1), (0.1, 2), (-1, 0), (0, 1)]:
        ml = 0 if low <= 0 <= high else min(abs(low), abs(high))
        mh = max(abs(low), abs(high))
        mapped = mapped and bool(
            np.all(
                (np.abs(np.linspace(low, high, 101)) >= ml)
                & (np.abs(np.linspace(low, high, 101)) <= mh)
            )
        )
    out = {
        "delta_derivative_max_finite_difference_error": max(errors),
        "signed_interval_image_event_implication": mapped,
        "passed": bool(max(errors) < 1e-8 and mapped),
    }
    (RES / "theory_checks.json").write_text(json.dumps(out, indent=2))
    assert out["passed"]
    return out


def main():
    raw = pd.read_csv(RES / "replicates.csv")
    stress = pd.read_csv(RES / "stress_replicates.csv")
    jr = pd.read_csv(RES / "joint_replicates.csv")
    intervals = pd.read_csv(RES / "bootstrap_intervals.csv")
    null = pd.read_csv(RES / "expanded_null_intervals.csv")
    assert (
        len(raw) == 71200
        and len(stress) == 3600
        and len(jr) == 3600
        and len(intervals) == 960
        and len(null) == 1600
    )
    for z in [raw, stress, jr, intervals, null]:
        assert (
            np.isfinite(z.select_dtypes("number").to_numpy()).all()
            and not z.isna().any().any()
        )
    assert not raw.duplicated(
        ["scenario", "n", "rep", "model", "density", "variable", "d"]
    ).any()
    s = summary(raw, ["scenario", "n", "model", "density", "variable", "d"])
    s.to_csv(RES / "summary.csv", index=False)
    st = summary(stress, ["scenario", "model", "variable", "d"])
    st.to_csv(RES / "stress_summary.csv", index=False)
    js = summary(jr, ["n", "pair", "d"])
    js.to_csv(RES / "joint_summary.csv", index=False)
    cov = (
        intervals.groupby(["n", "variable"])
        .agg(
            coverage=("covered", "mean"),
            datasets=("covered", "size"),
            mean_width=("upper", lambda v: 0.0),
        )
        .reset_index()
    )
    widths = (
        intervals.assign(width=intervals.upper - intervals.lower)
        .groupby(["n", "variable"])
        .width.mean()
    )
    cov["mean_width"] = [widths.loc[(v.n, v.variable)] for v in cov.itertuples()]
    cov["coverage_mcse"] = np.sqrt(cov.coverage * (1 - cov.coverage) / cov.datasets)
    cov.to_csv(RES / "bootstrap_summary.csv", index=False)
    nc = (
        null[null.scenario == "outcome_noise"]
        .groupby(["n", "variable"])
        .agg(
            coverage=("covered", "mean"),
            mean_lower=("magnitude_lower", "mean"),
            mean_upper=("magnitude_upper", "mean"),
        )
        .reset_index()
    )
    nc["coverage_mcse"] = np.sqrt(nc.coverage * (1 - nc.coverage) / 100)
    nc.to_csv(RES / "expanded_null_coverage.csv", index=False)
    val = json.loads((RES / "validation.json").read_text())
    assert val["passed"]
    checks = theory_checks()
    graph = s[(s.model == "graph") & (s.density == "fitted_model") & (s.d == 0.5)]
    noise = graph[graph.scenario == "outcome_noise"]
    names = [C, I, E, L, N, D, M, B]
    earlier = pd.read_csv(ROOT.parent / "results/observational_summary.csv")
    earlier = earlier[
        (earlier.scenario == "outcome_noise")
        & (earlier.n == 5000)
        & (earlier.d == 0.5)
        & earlier.method.str.endswith("estimated density")
    ].set_index("variable")
    current = noise[noise.n == 5000].set_index("variable")
    compare = pd.DataFrame(
        {
            "variable": names,
            "earlier_spline_rmse": [earlier.loc[v, "rmse"] for v in names],
            "learned_mechanism_rmse": [current.loc[v, "rmse"] for v in names],
        }
    )
    compare.to_csv(RES / "comparison_with_earlier.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(names))
    for j, row in compare.iterrows():
        ax.plot(
            [row.earlier_spline_rmse, row.learned_mechanism_rmse],
            [j, j],
            color="#b8c5d2",
            zorder=1,
        )
    ax.scatter(
        compare.earlier_spline_rmse,
        positions,
        label="Earlier parent-adjusted spline",
        zorder=2,
    )
    ax.scatter(
        compare.learned_mechanism_rmse,
        positions,
        label="Learned mechanisms, graph-informed",
        zorder=2,
    )
    ax.set_yticks(positions, names)
    ax.set_xscale("log")
    ax.set(
        xlabel="RMSE, log scale",
        title="Noisy outcome, n=5000, d=0.5 — different modeling information",
    )
    ax.legend()
    fig.tight_layout()
    save("01_estimator_comparison")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, name in zip(axes.flat, names):
        a = noise[noise.variable == name]
        ax.loglog(a.n, a.rmse, "o-")
        ax.set(title=name, xlabel="Sample size", ylabel="RMSE")
        ax.grid(alpha=0.2)
    fig.suptitle("Learned-mechanism estimation: independent outcome noise SD=0.5")
    fig.tight_layout()
    save("02_sample_size")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    expanded = s[
        (s.model == "expanded")
        & (s.density == "fitted_model")
        & (s.d == 0.5)
        & (s.scenario == "outcome_noise")
    ]
    for ax, name in zip(axes, [A, R]):
        z = expanded[expanded.variable == name]
        ax.loglog(z.n, z.rmse, "o-", label="RMSE of estimated magnitude")
        ax.loglog(z.n, z["mean"], "s--", label="Mean estimated magnitude")
        ax.set(
            title=name + " — zero not imposed",
            xlabel="Sample size",
            ylabel="Magnitude (true value zero)",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Expanded parametric model; reported-bugs interpretation requires extrapolation"
    )
    fig.tight_layout()
    save("03_estimated_nulls")
    fig, ax = plt.subplots(figsize=(8, 4))
    for pair, z in js[js.d == 0.5].groupby("pair"):
        ax.loglog(z.n, z.rmse, "o-", label=pair)
    ax.set(
        xlabel="Sample size",
        ylabel="Joint Mean PEACE RMSE",
        title="Three jointly intervened pairs, d=0.5",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save("04_joint_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, [C, N, L]):
        for den, label in [
            ("fitted_model", "Fitted mechanism density"),
            ("kde", "Estimated KDE"),
        ]:
            z = s[
                (s.scenario == "outcome_noise")
                & (s.n == 5000)
                & (s.model == "graph")
                & (s.density == den)
                & (s.variable == name)
                & (s.d > 0)
            ]
            ax.plot(z.d, z.rmse, "o-", label=label)
        ax.set(title=name, xlabel="d", ylabel="RMSE")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save("05_density_options")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h = st[(st.scenario == "hidden_confounding") & (st.d == 0.5)]
    axes[0].bar(h.variable, h.bias)
    axes[0].set(title="Hidden confounding persists", ylabel="Bias against causal PEACE")
    z = st[(st.scenario == "wrong_outcome_shape") & (st.variable == L) & (st.d == 0.5)]
    axes[1].bar(["Original fitted shape", "Add the required quadratic term"], z["mean"])
    axes[1].axhline(z.truth.iloc[0], color="black", ls="--", label="True magnitude")
    axes[1].set(
        title="Upgrade model specification matters", ylabel="Estimated Mean PEACE"
    )
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save("06_assumption_sensitivity")
    pivot = cov.pivot(index="variable", columns="n", values="coverage").loc[names]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot, vmin=0.85, vmax=1, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, label="Empirical nominal-95% coverage")
    ax.set_yticks(range(8), names)
    ax.set_xticks(range(2), ["n=5000", "n=20000"])
    for i in range(8):
        for j in range(2):
            ax.text(
                j, i, f"{pivot.iloc[i, j]:.1%}", ha="center", va="center", color="black"
            )
    ax.set_title("Full-refit bootstrap audit: 60 datasets per sample size")
    fig.tight_layout()
    save("07_uncertainty_audit")
    x, y = generate(5000, seed=810001, noise=0.5)
    fg = fit(x, y)
    fe = fit(x, y, "expanded")
    dense = np.linspace(0, 3, 61)
    curves = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name in zip(axes.flat, [C, N, L, D, A, R]):
        fitted = fe if name in [A, R] else fg
        truth = [true_value(name, float(d)) for d in dense]
        estimated = [v[0] for v in calculate_curve(fitted, name, dense)]
        ax.plot(dense, truth, color="black", label="Population truth")
        ax.plot(dense, estimated, "--", label="Learned mechanisms")
        if name in [C, N, L]:
            kd = [v[0] for v in calculate_curve(fitted, name, dense, "kde", x)]
            ax.plot(dense, kd, ":", label="Same response, KDE weights")
        ax.set(
            title=name + (" (expanded model)" if name in [A, R] else ""),
            xlabel="d",
            ylabel="Mean PEACE",
        )
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.grid(alpha=0.2)
        if np.ptp(np.r_[truth, estimated]) < 1e-8:
            center = truth[0]
            pad = max(0.01, 0.05 * abs(center))
            ax.set_ylim(center - pad, center + pad)
        curves.extend(
            dict(variable=name, d=d, truth=t, estimate=e)
            for d, t, e in zip(dense, truth, estimated)
        )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "One fixed observational sample: n=5000, outcome noise SD=0.5; no confidence bands"
    )
    fig.tight_layout()
    save("08_estimated_curves")
    pd.DataFrame(curves).to_csv(RES / "illustrative_curves.csv", index=False)
    original = graph[
        (graph.scenario == "original")
        & (graph.n == 5000)
        & graph.variable.isin([C, N, L, D, M, I])
    ][["variable", "truth", "mean", "rmse"]]
    noisy = noise[noise.n.isin([5000, 100000])][
        ["n", "variable", "truth", "mean", "rmse", "bias_mcse"]
    ]
    noisy = noisy[noisy.variable.isin(names)]
    nullexp = expanded[expanded.variable.isin([A, R])][
        ["n", "variable", "truth", "mean", "rmse"]
    ]
    jointtab = js[(js.d == 0.5) & js.n.isin([5000, 20000])][
        ["n", "pair", "truth", "mean", "rmse"]
    ]
    txt = f"""# Learned-mechanism observational PEACE: follow-up results

The new estimator learns the required assignment coefficients/noise scales and outcome coefficients from observational samples. It then propagates interventions through those fitted mechanisms and integrates the resulting response-gradient magnitude. It does not receive the true numerical mechanism coefficients. The graph, functional families, fixed analysis domains, and uniform calls/upgrade reference laws are supplied modeling assumptions.

**Result:** this works very well in the original correctly specified noiseless experiment and substantially improves most noisy-outcome estimates. It does not make every effect precise at n=5000. Discount remains difficult under added noise. Hidden confounding and model misspecification remain consequential.

## 1. Setting and what is learned

The original code-faithful model, with usage coefficient 3, is retained. Calls→need and calls→interactions are fitted by regression. Logit usage is regressed on need; its Gaussian residual scale is estimated. Discount is regressed on sigmoid need. The outcome coefficients are fitted in the structural feature basis including reciprocal upgrade. The need conditional Gaussian scale is also estimated. Required mediated response gradients and the need marginal density follow from the fitted mechanisms, not their true values. Expanded-model ad-spend and reported-bugs mean mechanisms are also fitted. The reported-bugs contribution to a bugs-faced response uses empirical marginal standardization of sigmoid need; root bugs means are estimated empirically.

Two outcome models are evaluated on the same datasets. **Graph-informed:** the eight assumed outcome parents are included; ad spend and reported bugs are excluded by the supplied causal graph. Their exact nulls are structural assumptions, not zeros discovered from data. **Expanded:** both variables are included as candidate linear outcome terms and their coefficients are estimated. This is a strong parametric model. Reported bugs remains deterministic given its parents; its expanded-model causal interpretation depends on functional-form extrapolation and is not generic nonparametric identification.

Bounded conditional supports for discount and ad spend move with their parents. Positive residual variance alone does not guarantee global conditional positivity for all fixed interventions. The fitted structural families identify parameters from observed variation and extrapolate their stated response outside unsupported combinations. The accuracy gain must be interpreted conditional on those restrictions, not as an assumption-free observational solution.

For each original/noisy-outcome scenario, 100 datasets are generated at n=1000,5000,20000,100000. Noise SD=.5 is an explicitly separate stress setting; the original has no outcome noise. The fitted-model density and a marginal KDE alternative are compared for calls, need and upgrade. Degrees 0,.5,1,2 are retained. Other scalar gradients are constant under the fitted family, making their density weights cancel. Three joint pairs use fitted joint density factors and 100 datasets at n=1000,5000,20000.

## 2. Original noiseless model

{table(original)}

Most outcome coefficients are recovered to numerical precision because the noiseless outcome lies exactly in the assumed feature basis. Need and calls still have nonzero estimation error because their mediated effects depend on learned assignments. This distinguishes the new benchmark from the earlier one that supplied assignment coefficients. Near-perfect noiseless recovery is conditional on correct functional families and must not be presented as generic causal discovery.

## 3. Noisy-outcome accuracy and sample size

{table(noisy)}

![Comparison](figures/01_estimator_comparison.png)

![Sample-size sensitivity](figures/02_sample_size.png)

The earlier flexible and new structural estimators use different modeling information and independent simulation seeds. The comparison is descriptive, not a controlled equal-information superiority test. The improvement demonstrates the value of a justified structural estimator in this particular experiment.

At n=5000, product need RMSE is about .069 and upgrade about .0032. Discount RMSE remains about .861, falling to about .212 at n=100000. Narrow residual discount variation limits precision even with a correct model. A mean estimate close to one is not evidence that every dataset gives an accurate estimate. No procedure here creates information absent from the observed treatment variation.

## 4. Null effects without imposing zero

{table(nullexp)}

![Estimated null magnitudes](figures/03_estimated_nulls.png)

The expanded model estimates rather than fixes the two coefficients. Their magnitudes shrink with sample size; small upward magnitude bias remains. These are parametric-model estimates, particularly restrictive for deterministic reported bugs. Graph-informed exact zeros are not used as evidence of empirical null-estimation accuracy.

Signed coefficient HC3 intervals are mapped through absolute value. Coverage in the noisy-outcome setting:

{table(nc)}

Noiseless Wald coverage is not interpreted: degenerate residual variance and floating-point coefficient errors make that a numerical rather than regular inferential setting.

## 5. Joint interventions and density options

{table(jointtab)}

![Joint accuracy](figures/04_joint_accuracy.png)

For need+usage, the fitted usage pathway is removed from the need derivative because usage is also intervened on. For calls+need, the calls→need pathway is removed. Joint gradient norms remain responsiveness summaries, not interaction coefficients.

![Density options](figures/05_density_options.png)

The primary need marginal is a Gaussian convolution with estimated intercept, slope and residual scale. Uniform calls/upgrade laws are declared reference models, so their normalized weights are constant. KDE relaxes that reference-family choice and introduces additional density error, particularly apparent for uniform upgrade. This separates response estimation from reference-density estimation. It does not identify a universally optimal bandwidth or degree.

![Illustrative fitted curves](figures/08_estimated_curves.png)

These are curves from one fixed independent sample, not confidence bands. Ad spend and reported bugs use the expanded outcome model so their nulls are not imposed. The original theorem-based interpretations remain those of the population functional in the earlier report. Estimated deviations can reflect coefficient and density uncertainty; they do not establish new population causal-frequency relationships.

## 6. Assumption sensitivity

{table(st[st.d == 0.5][["scenario", "model", "variable", "truth", "mean", "rmse"]])}

![Assumption sensitivity](figures/06_assumption_sensitivity.png)

The hidden-confounding setting is the same latent need–outcome perturbation as the earlier experiment. The learned estimator still shows approximately +.5 need bias: fitting observational mechanisms cannot repair the missing causal identifying information. This is sensitivity under violated identification assumptions, not failure of the PEACE functional.

A separately labelled model-specification challenge adds .03 upgrade² to the outcome. Fitting only the original reciprocal upgrade term gives upgrade PEACE around .420 against truth .760. Including and estimating the additional quadratic term recovers approximately .760. This is evidence that functional-family correctness matters, not evidence that the procedure automatically discovers the correct family. All primary and unfavorable stress results are retained.

## 7. Uncertainty and theoretical scope

Whole-observation bootstrap resamples refit all outcome and node models and the mechanism-implied density: 60 datasets at each n=5000 and n=20000, 150 resamples each. Regular need/calls/upgrade targets use percentile intervals. Constant signed-slope intervals are mapped through absolute value, retaining zero whenever the signed interval spans it. Graph-implied nulls are excluded from the coverage audit because their zero is imposed by the graph.

{table(cov)}

![Uncertainty audit](figures/07_uncertainty_audit.png)

Observed coverage ranges from {cov.coverage.min():.1%} to {cov.coverage.max():.1%}; with only 60 datasets per cell it has appreciable Monte Carlo uncertainty. The need interval at n=20000 undercovers in this audit. These results are not a blanket guarantee of nominal finite-sample coverage, and interval calibration remains a practical limitation. The point-estimation improvement does not depend on claiming universally valid intervals.

The separate estimation section now includes the fitted-mechanism intervention formula and sufficient consistency conditions. Its additional delta-method result requires a correctly specified identified finite-dimensional model, regular jointly asymptotically normal parameter fitting, a nonzero causal gradient on the target domain, differentiable positive density weights, and negligible integration error at the required scale. Same-sample parameter covariances must be retained. General zero-gradient norm inference is excluded; a separate signed-interval image argument covers constant slopes. These are additional conditional theoretical results, not statements already established in the original manuscript.

## 8. Verification and reproducibility

The estimator functions do not call the oracle derivative or density functions. Those functions supply simulation truth only. All result counts and finite values are checked. Fitted-sem response derivatives are independently checked by common-random-number interventions; scalar and joint integrals are refined; the fitted need density integrates to one; changing observed data changes the learned need slope. The new parameter-derivative formula for PEACE agrees with an independent finite difference within {checks["delta_derivative_max_finite_difference_error"]:.2g}; interval-image event inclusion is checked separately. See results/validation.json and results/theory_checks.json.

Rebuild with the parent scientific environment: `python study.py all`, then `python report.py`. Parent requirements-lock.txt records the packages. Fixed seeds and single-thread numerical calculations are used. Root graph/families/support assumptions are modeling inputs; this is exploratory development, not a preregistered or assumption-free validation. Earlier experiment results remain intact. The main manuscript and sodium files were not edited.
"""
    (ROOT / "REPORT.md").write_text(txt)
    sections = [
        (
            "01_estimator_comparison",
            "Improvement over the earlier flexible estimator",
            "Different structural information is used; compare each estimator against its own truth.",
        ),
        (
            "02_sample_size",
            "Sample size and noisy-outcome accuracy",
            "Discount remains imprecise at n=5000; increasing sample size helps.",
        ),
        (
            "03_estimated_nulls",
            "Nulls estimated in the expanded model",
            "Zero coefficients are not imposed here. Reported bugs requires parametric extrapolation.",
        ),
        (
            "04_joint_accuracy",
            "Joint intervention estimation",
            "The derivative calculation accounts for paths blocked by simultaneous interventions.",
        ),
        (
            "05_density_options",
            "Mechanism density and KDE",
            "Reference-density assumptions and estimation errors remain explicit.",
        ),
        (
            "08_estimated_curves",
            "Illustrative estimated curves",
            "One fixed observational sample, not confidence bands. Null variables use the expanded model.",
        ),
        (
            "06_assumption_sensitivity",
            "Identification and specification sensitivity",
            "Hidden confounding persists; an incorrect response family can give incorrect PEACE.",
        ),
        (
            "07_uncertainty_audit",
            "Uncertainty audit",
            "Coverage is not uniformly nominal. Sixty datasets per cell give a limited calibration check.",
        ),
    ]
    body = '<h1>Learned mechanisms for observational PEACE</h1><p class="lead">Customer-satisfaction follow-up</p><p><a href="REPORT.md">Detailed findings</a> · <a href="README.md">Setting and reproduction</a> · <a href="../REPORT.html">Earlier experiment</a></p><div class="callout"><b>Finding:</b> learning the required mechanism parameters substantially improves estimation under the specified causal model. It does not remove identification assumptions, guarantee precision for weakly varying treatments, or validate every confidence interval.</div><p>The graph, functional families and root reference laws are supplied; unknown required numerical coefficients and noise scales are estimated. Graph-implied nulls are separated from null coefficients estimated in an expanded model.</p>'
    for name, title, caption in sections:
        body += f'<section><h2>{title}</h2><img src="figures/{name}.png" alt="{html.escape(title)}"><p>{caption} <a href="figures/{name}.svg">Vector figure</a></p></section>'
    for title, z in [
        ("Original noiseless model, n=5000", original),
        ("Noisy-outcome estimates", noisy),
        ("Expanded null estimates", nullexp),
        ("Joint targets", jointtab),
        ("Full-refit bootstrap coverage", cov),
    ]:
        body += f'<h2>{title}</h2><div class="table">{z.to_html(index=False, float_format=lambda x: f"{x:.5f}")}</div>'
    style = 'body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#203044;max-width:1120px;margin:44px auto;padding:0 24px}h1{font-size:40px;line-height:1.2}h2{font-size:24px;margin-top:40px}.lead{font-size:22px;color:#53647a}a{color:#245c9d}.callout{background:#edf3fa;border-left:4px solid #245c9d;padding:22px}img{max-width:100%;height:auto;border:1px solid #dce4ec;border-radius:6px}.table{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #dce4ec}th{background:#edf3fa}'
    (ROOT / "REPORT.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Learned-mechanism PEACE</title><style>'
        + style
        + "</style><main>"
        + body
        + "</main></html>"
    )
    sources = [
        ROOT / "study.py",
        ROOT.parent / "experiment.py",
        ROOT.parent / "requirements-lock.txt",
    ]
    (ROOT / "provenance.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "primary_usage_coefficient": 3,
                "root_reference_laws": "Declared Uniform calls [0,4] and upgrade [0,10]",
                "source_hashes": {
                    str(f.relative_to(ROOT.parent)): hashlib.sha256(
                        f.read_bytes()
                    ).hexdigest()
                    for f in sources
                },
                "row_counts": {
                    "main": len(raw),
                    "joint": len(jr),
                    "stress": len(stress),
                    "bootstrap_intervals": len(intervals),
                },
                "all_unknown_required_coefficients_and_scales_fitted": True,
            },
            indent=2,
        )
    )
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "manifest-sha256.json"
    ]
    (ROOT / "manifest-sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)
            },
            indent=2,
        )
    )
    print(
        "Learned-mechanism reports, figures, checks and manifests generated", flush=True
    )


if __name__ == "__main__":
    main()
