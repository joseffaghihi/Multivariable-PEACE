"""Build review artifacts from retained matched simulation results."""

from study import (
    A,
    B,
    BASE,
    C,
    D,
    E,
    FIG,
    I,
    L,
    M,
    N,
    RES,
    ROOT,
    fit_models,
    generate,
    hashlib,
    json,
    np,
    pd,
    plt,
    truth_conditional,
    truth_total,
)
from models import selected_additive
import html
import re

LABEL = {
    "linear": "Linear",
    "additive": "Smooth additive",
    "interaction": "Interaction capable",
    "selected_additive": "Selected additive",
}
ORDER = list(LABEL)
COLORS = dict(zip(ORDER, ["#777777", "#2378ac", "#b34d73", "#26836d"]))


def summarize(frame, keys, prefix):
    assert np.isfinite(frame[["estimate", "truth"]].to_numpy()).all()
    r = frame.groupby(keys + ["rep"], as_index=False).agg(
        estimate=("estimate", "mean"), truth=("truth", "first")
    )
    r["error"] = r.estimate - r.truth

    def agg(g):
        return pd.Series(
            dict(
                truth=g.truth.iloc[0],
                mean=g.estimate.mean(),
                bias=g.error.mean(),
                rmse=np.sqrt(np.mean(g.error**2)),
                sd=g.estimate.std(ddof=1),
                mcse=g.estimate.std(ddof=1) / np.sqrt(len(g)),
                replicates=len(g),
            )
        )

    s = r.groupby(keys, sort=False).apply(agg, include_groups=False).reset_index()
    r.to_csv(RES / (prefix + "_replicate_estimates.csv"), index=False)
    s.to_csv(RES / (prefix + "_summary.csv"), index=False)
    assert (s.replicates == 40).all()
    return r, s


def table(s, scenario, n, degree=True):
    z = s[(s.scenario == scenario) & (s.n == n)]
    if degree:
        z = z[z.d == 0.5]
    out = [
        "| Variable | Truth | Linear | Smooth additive | Interaction capable | Selected additive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in [C, N, I, E, L, D, M, A, B]:
        row = z[z.variable == v]
        values = [
            v + (" †" if degree and v in [D, A] else ""),
            f"{row.truth.iloc[0]:.5f}",
        ]
        for m in ORDER:
            q = row[row.method == m]
            values.append(f"{q['mean'].iloc[0]:.5f}" if len(q) else "Not identified")
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def save(name):
    plt.savefig(FIG / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close()


def mdtable(frame):
    rows = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in row)
            + " |"
        )
    return "\n".join(rows)


def render_md(source):
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)

    blocks = []
    for block in source.strip().split("\n\n"):
        lines = block.splitlines()
        if lines[0].startswith("|"):
            rows = [l for l in lines if not re.fullmatch(r"[|: \-]+", l)]
            blocks.append(
                "<table>"
                + "".join(
                    "<tr>"
                    + "".join(
                        f"<{'th' if k == 0 else 'td'}>{inline(c.strip())}</{'th' if k == 0 else 'td'}>"
                        for c in l.strip("|").split("|")
                    )
                    + "</tr>"
                    for k, l in enumerate(rows)
                )
                + "</table>"
            )
        elif lines[0].startswith("#"):
            level = len(lines[0]) - len(lines[0].lstrip("#"))
            blocks.append(
                f"<h{level}>" + inline(lines[0].lstrip("# ")) + f"</h{level}>"
            )
        elif all(l.startswith("- ") for l in lines):
            blocks.append(
                "<ul>"
                + "".join("<li>" + inline(l[2:]) + "</li>" for l in lines)
                + "</ul>"
            )
        else:
            blocks.append("<p>" + inline(block) + "</p>")
    return "\n".join(blocks)


def main():
    FIG.mkdir(exist_ok=True)
    gf = pd.concat(
        [
            pd.read_csv(RES / x)
            for x in ["global_fold_results.csv", "selected_global_fold_results.csv"]
        ],
        ignore_index=True,
    )
    cf = pd.concat(
        [
            pd.read_csv(RES / x)
            for x in [
                "conditional_fold_results.csv",
                "selected_conditional_fold_results.csv",
            ]
        ],
        ignore_index=True,
    )
    gr, gs = summarize(gf, ["scenario", "n", "variable", "method", "d"], "global")
    cr, cs = summarize(cf, ["scenario", "n", "variable", "method"], "conditional")
    signed = cf.groupby(
        ["scenario", "n", "variable", "method", "rep"], as_index=False
    ).signed.mean()
    signed.to_csv(RES / "conditional_signed_replicate_estimates.csv", index=False)
    signed.groupby(["scenario", "n", "variable", "method"]).signed.agg(
        ["mean", "std", "count"]
    ).reset_index().to_csv(RES / "conditional_signed_summary.csv", index=False)
    jr, js = summarize(
        pd.read_csv(RES / "joint_fold_results.csv"),
        ["scenario", "n", "method"],
        "joint",
    )
    sr, ss = summarize(
        pd.read_csv(RES / "shape_challenge_fold_results.csv"),
        ["n", "method"],
        "shape_challenge",
    )
    shape = pd.read_csv(RES / "shape_challenge_fold_results.csv")
    shape = shape.groupby(["n", "method", "rep"], as_index=False)[
        ["positive", "negative"]
    ].mean()
    shape["signed"] = shape.positive - shape.negative
    shape.groupby(["n", "method"])[
        ["positive", "negative", "signed"]
    ].mean().reset_index().to_csv(RES / "shape_direction_summary.csv", index=False)
    assert np.allclose(
        shape.positive + shape.negative,
        sr.set_index(["n", "method", "rep"]).loc[
            pd.MultiIndex.from_frame(shape[["n", "method", "rep"]]), "estimate"
        ],
    )
    checks = {
        f: json.loads((RES / f).read_text())
        for f in [
            "validation.json",
            "selected_validation.json",
            "joint_validation.json",
            "integration_audit.json",
            "selected_integration_audit.json",
        ]
    }
    assert all(c["passed"] for c in checks.values())
    # Actual estimates, rather than an error-only comparison.
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    for ax, v in zip(axs.flat, [C, N, I, E, L, D, M, A, B]):
        z = gs[
            (gs.scenario == "original")
            & (gs.n == 5000)
            & (gs.variable == v)
            & (gs.d == 0.5)
        ]
        for k, m in enumerate(ORDER):
            q = z[z.method == m]
            if len(q):
                ax.errorbar(
                    k,
                    q["mean"].iloc[0],
                    yerr=q.sd.iloc[0],
                    fmt="o",
                    color=COLORS[m],
                    capsize=3,
                )
        ax.axhline(z.truth.iloc[0], color="black", ls="--", lw=1)
        ax.set_title(v + (" †" if v in [D, A] else ""))
        ax.set_xticks(range(4), ["Lin", "Add", "Int", "Sel"])
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Original simulator: total-response Mean PEACE, n=5,000\nPoints = 40-dataset mean; bars = replicate SD; dashed = truth. † Model extrapolation.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save("01_actual_global_estimates")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, v in zip(axs, [L, D, A]):
        for m in ORDER:
            z = cs[
                (cs.scenario == "outcome_noise") & (cs.variable == v) & (cs.method == m)
            ].sort_values("n")
            ax.errorbar(
                z.n,
                z["mean"],
                yerr=z.sd,
                marker="o",
                label=LABEL[m],
                color=COLORS[m],
                capsize=3,
            )
        ax.axhline(z.truth.iloc[0], ls="--", color="black")
        ax.set_xscale("log")
        ax.set_title(v)
        ax.set_xlabel("Sample size")
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("Supported conditional Mean PEACE Q")
    axs[-1].legend(fontsize=8)
    fig.suptitle(
        "Outcome noise SD=0.5: weak variation limits slope precision\nBars are replicate SD, not confidence intervals"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save("02_noise_supported_estimates")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, s, title, true in zip(
        axs,
        [gs, cs],
        ["Total-response P", "Supported conditional Q"],
        [3, truth_conditional(M, "interaction")],
    ):
        for m in ORDER:
            z = s[(s.scenario == "interaction") & (s.variable == M) & (s.method == m)]
            if "d" in z:
                z = z[z.d == 0.5]
            z = z.sort_values("n")
            ax.errorbar(
                z.n,
                z["mean"],
                yerr=z.sd,
                marker="o",
                label=LABEL[m],
                color=COLORS[m],
                capsize=3,
            )
        ax.axhline(true, ls="--", color="black", label=f"Truth {true:.5f}")
        ax.set_title(title)
        ax.set_xlabel("Sample size")
        ax.grid(alpha=0.2)
    axs[1].legend(fontsize=8)
    fig.suptitle(
        "Interaction challenge: these are different causal functionals\nBars = replicate SD"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save("03_interaction_distinct_targets")
    x, y = generate(20000, seed=930000, noise=0.5)
    y = y + 0.03 * x[L].to_numpy() ** 2
    X = x[BASE].to_numpy()
    fits = fit_models(X[:10000], y[:10000])
    fits["selected_additive"] = selected_additive(X[:10000], y[:10000])
    j = BASE.index(L)
    t = np.linspace(0, 10, 400)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(
        t,
        -8 / (t + 1) ** 2 + 0.06 * t,
        color="black",
        ls="--",
        label="True signed derivative",
    )
    for m, model in fits.items():
        axs[0].plot(
            t,
            model.standardized_derivative(t, model.prepare_reference(X[10000:], j)),
            color=COLORS[m],
            label=LABEL[m],
        )
    axs[0].axhline(0, color="black", lw=0.5)
    axs[0].set_xlabel("Last upgrade")
    axs[0].set_ylabel("Signed response derivative")
    axs[0].set_title("One illustrative fit; 10,000 training observations")
    axs[0].legend(fontsize=8)
    for k, m in enumerate(ORDER):
        z = ss[(ss.n == 20000) & (ss.method == m)]
        axs[1].errorbar(
            k, z["mean"].iloc[0], yerr=z.sd.iloc[0], fmt="o", color=COLORS[m], capsize=4
        )
    axs[1].axhline(ss.truth.iloc[0], color="black", ls="--")
    axs[1].set_xticks(range(4), ["Linear", "Additive", "Interaction", "Selected"])
    axs[1].set_ylabel("Mean PEACE")
    axs[1].set_title("40 datasets; bars = replicate SD")
    fig.suptitle(
        "Unknown-shape challenge: Y gains 0.03 × upgrade²\nNo quadratic term or reciprocal transformation supplied to the estimators"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save("04_unknown_shape_and_direction")
    # The degree interpretation is tied to the paper's log-density covariance identity.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, v in zip(axs, [L, N]):
        for m in ORDER:
            z = gs[
                (gs.scenario == "original")
                & (gs.n == 5000)
                & (gs.variable == v)
                & (gs.method == m)
            ].sort_values("d")
            ax.plot(z.d, z["mean"], marker="o", color=COLORS[m], label=LABEL[m])
        d = np.linspace(0, 2, 50)
        ax.plot(
            d,
            [truth_total(v, float(q), "original") for q in d],
            color="black",
            ls="--",
            label="True curve",
        )
        ax.set_title(v)
        ax.set_xlabel("Degree d")
        ax.set_ylabel("Total-response Mean PEACE")
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=8)
    fig.suptitle(
        "Degree curves: nonlinear response does not imply a nonflat PEACE curve"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save("05_degree_curves")
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ["linear", "additive", "interaction"]:
        z = js[(js.scenario == "outcome_noise") & (js.method == m)].sort_values("n")
        ax.errorbar(
            z.n,
            z["mean"],
            yerr=z.sd,
            marker="o",
            capsize=3,
            label=LABEL[m],
            color=COLORS[m],
        )
    ax.axhline(
        z.truth.iloc[0], color="black", ls="--", label=f"Truth {z.truth.iloc[0]:.5f}"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Joint total-response Mean PEACE")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_title(
        "Economy + upgrade: joint gradient norm\nOutcome noise SD=.5; bars = replicate SD"
    )
    save("06_joint_root_response")
    pred = pd.concat(
        [
            pd.read_csv(RES / x)
            for x in [
                "prediction_fold_results.csv",
                "selected_prediction_fold_results.csv",
            ]
        ]
    )
    pred = pred.groupby(
        ["scenario", "n", "rep", "design", "method"], as_index=False
    ).heldout_mse.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in ORDER:
        q = cr[
            (cr.scenario == "outcome_noise")
            & (cr.n == 5000)
            & (cr.variable == A)
            & (cr.method == m)
        ].merge(
            pred[
                (pred.scenario == "outcome_noise")
                & (pred.n == 5000)
                & (pred.design == "ad")
                & (pred.method == m)
            ],
            on=["scenario", "n", "rep", "method"],
        )
        ax.scatter(
            q.heldout_mse, q.estimate, alpha=0.45, s=20, label=LABEL[m], color=COLORS[m]
        )
    ax.axhline(0, color="black", ls="--")
    ax.set_xlabel("Held-out outcome prediction MSE")
    ax.set_ylabel("Estimated supported conditional PEACE of ad spend")
    ax.legend(fontsize=8)
    ax.set_title(
        "Good prediction alone does not validate derivative magnitudes\nOutcome noise SD=.5, n=5,000; each point is one dataset"
    )
    save("07_prediction_and_derivatives")
    sel = pd.read_csv(RES / "nonlinear_component_selection.csv").fillna(
        {"nonlinear_components": ""}
    )
    z = sel[
        (sel.scenario == "outcome_noise") & (sel.n == 5000) & (sel.design == "outcome")
    ]
    freq = [
        z.nonlinear_components.str.split(";")
        .apply(lambda s: v in [a.strip() for a in s])
        .mean()
        for v in BASE
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(BASE, freq, color=COLORS["selected_additive"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of training folds selected")
    ax.set_title(
        "Nonlinear components chosen from data, not from the true SEM\nOutcome model, noise SD=.5, n=5,000; 80 training folds"
    )
    ax.tick_params(axis="x", rotation=25)
    save("08_selected_nonlinearity")
    joint_table = mdtable(
        js[["scenario", "n", "method", "truth", "mean", "rmse", "mcse"]]
    )
    shape_table = mdtable(ss[["n", "method", "truth", "mean", "rmse", "mcse"]])
    text_report = f"""# Customer satisfaction: estimating PEACE without exact structural forms

This is a separate synthetic observational study. It leaves the submitted manuscript and sodium experiment unchanged. The reference is the committed customer simulator (monthly-usage coefficient 3, not the printed 30). Known mechanisms generate data and score estimates; their logistic/reciprocal transformations and assignment-noise distributions are not supplied to the estimators.

## What was assumed and fitted

We retain the causal graph, absence of unmeasured confounding in the primary simulation, valid adjustment sets, intervention consistency, and the response/support assumptions stated below. The simulator has independent disturbances. We do not learn the graph. Raw predictors enter (1) linear least squares, (2) smooth additive cubic splines, (3) smooth regression with generic pairwise products and 96 Gaussian-kernel random Fourier features, and (4) an additive model that selects nonlinear components from data. The fourth method was added after inspecting noisy derivative instability; this is an exploratory development study, not a preregistered comparison. All earlier results are retained.

Every case uses 40 independent datasets and two-fold cross-fitting: train on half, evaluate on the other half, swap, and average the two estimates. Fold estimates are averaged before calculating RMSE or Monte Carlo uncertainty. Original and outcome-noise SD=.5 scenarios use n=1,000, 5,000, 20,000; an interaction challenge uses n=5,000 and 20,000. These are 320 matched datasets. Separate joint and changed-shape checks add 200 and 80 datasets. Nested sample sizes share random seeds; comparisons across sizes are not independent.

Additive/interaction fits use knots (6,12,20, subject to sample-size constraints) and ridge smoothing. Training generalized cross-validation selects the additive knot count and each model’s penalty; the interaction model inherits the selected additive knot count. The selected additive fit retains all linear terms, uses a forward BIC-style residual-sum-of-squares criterion to select nonlinear spline groups and knot count, then fits the selected basis. It does not preassign upgrade as nonlinear. Selection does not establish a consistency theorem, optimal derivative tuning, double robustness, or confidence-interval validity. All derivatives are analytic. These are established regression ingredients, not a new identification method.

Adjustment/precision predictors are: calls → [calls,economy,upgrade,bugs faced]; need → [need,calls,interactions,economy,upgrade,bugs faced]; interactions/economy/upgrade/discount/usage/bugs faced → [calls,interactions,economy,upgrade,need,discount,usage,bugs faced]; ad spend → that last set plus ad spend. For each treatment, the remaining variables in its particular regression are unaffected by that treatment; descendants are excluded. Calls and need use reduced regressions so downstream variables follow the intervention rather than being fixed. No learner is told that the ad-spend coefficient is zero.

## Two targets; they must remain distinct

**Global total response:** h_T(t)=E[Y under do(T=t)], with descendants following the intervention. Mean PEACE is P_d=∫|h′_T(t)|f_T(t)^(2d)dt / ∫f_T(t)^(2d)dt on a declared fixed domain. The signed derivative is standardized over unaffected covariates BEFORE taking its magnitude. At d=.5, held-out treatment observations in the domain provide empirical natural-density weights; other degrees use training KDE and quadrature. Physical root supports are known; the root uniform density is not supplied to the learner.

Domains: calls [0,4], interactions [.2,4.8], economy [.5,1], upgrade [0,10], need [−.3,.7], discount [.07,.18], usage [.35,.65], ad spend [.3,1.2], bugs faced [0,2]. Root treatments have support for standardization in this independent-disturbance simulator. Need/interactions/usage have positive but often extremely weak conditional overlap. Discount and ad spend have moving bounded conditional supports: generic global standardization is not identified. Their linear/additive global values below are extrapolations under model restrictions (†). The unrestricted interaction-capable model therefore has no global estimate for these two variables. Reported bugs is deterministic given its parents: no unrestricted conditional derivative is identified, and it is not estimated here. Its zero effect follows only with a graph exclusion assumption.

**Supported conditional response:** μ(t,w)=E[Y_t|W=w], Q_T=E|∂_t μ(T,W)| over the full natural joint distribution. It is conditional-response Mean PEACE at d=.5 with full conditional support and normalizer one. Held-out gradients are evaluated at observed combinations, without trimming or KDE. It avoids unsupported global intervention combinations but still requires causal identification, smoothness and sufficient local variation. It is not a substitute estimate for P. In the interaction challenge, usage has P=.3×10=3 but Q=3.520166. There is no universal ordering between them under different weighting laws.

## Actual global Mean PEACE estimates: original simulator, n=5,000

Numbers are the means of 40 independently generated dataset estimates, d=.5. Full individual estimates, bias, RMSE, replicate SD and Monte Carlo SE are in the downloadable CSVs.

{table(gs, "original", 5000)}

† Model extrapolation. Reported bugs: not estimated. Positive PEACE measures magnitude, not direction. Bugs faced has signed effect −.2; its PEACE is +.2. The linear upgrade coefficient is a best linear projection, approximately −.4212, rather than the true variable derivative −8/(1+t)². Its resulting PEACE near .421 is therefore not the nonlinear-response target .72727. Curvature elsewhere can also distort linear coefficients through correlation.

## Supported conditional estimates: outcome noise SD=.5, n=5,000

These are Q estimates, not the unsupported global P for discount/ad spend.

{table(cs, "outcome_noise", 5000, False)}

Flexibility is useful but does not automatically stabilize derivatives. Narrow independent discount/ad-spend variation amplifies slope uncertainty, and absolute values turn zero-centered slope noise into positive magnitudes. Ad spend is approximately .97×usage−.05×upgrade plus a constant, with a very narrow assignment residual. When the exact reciprocal upgrade shape is no longer supplied, distinguishing its response from the ad-spend contribution is much harder. The earlier exact-feature structural benchmark is a more restrictive reference, not evidence that the same precision is available with weaker shape assumptions. Selected additive smoothing reduces several spurious oscillations, but its ad-spend estimate remains imprecise. Larger samples improve precision; they do not fill absent global support. This is an estimation/support limitation, not a failure of PEACE.

## Unknown-shape check and interpretation

We add .03×upgrade² to the outcome, retaining noise SD=.5. No special quadratic or reciprocal feature is supplied. The true signed derivative −8/(1+t)²+.06t crosses zero at 4.464754. Its negative contribution to Mean PEACE is .593805, its positive contribution .166533, and their sum .760338. The signed average is −.427273; cancellation is precisely why signed average and PEACE differ. A linear projection gives a magnitude about .12 and cannot express this change of direction. The flexible methods approximate the shape and produce substantially closer PEACE estimates, with residual boundary/smoothing bias.

{shape_table}

For degree curves, use manuscript Theorem 8.2 as derived in its appendix: ∂P_d/∂d=2 Cov_(f^(2d))( |h′|, log f). The printed raw-f version needs correction. An increasing curve means larger derivative magnitudes receive higher-density weight; a decreasing curve means the opposite covariance. This concerns magnitude, not positive/negative causal direction. Original upgrade has a uniform density, hence zero covariance and a flat true curve .727273 at every d despite its curved response. An estimated nonflat upgrade curve can reflect density/derivative estimation. Need has a small negative covariance and a slightly decreasing true degree curve; larger d favors its common values, whose local magnitudes are slightly smaller. The modal-concentration part of Theorem 8.2 gives its limiting local magnitude about 5.162361. No empirical wiggle alone establishes such a structural conclusion.

## Interaction and joint checks

The interaction challenge adds 20(need−.2)(usage−.515). Usage's standardized derivative is 3, while its conditional derivative is 3+20(need−.2). Thus P=3 and Q=3.520166. At n=20,000 the interaction-capable estimator gives mean P={gs[(gs.scenario == "interaction") & (gs.n == 20000) & (gs.variable == M) & (gs.method == "interaction") & (gs.d == 0.5)]["mean"].iloc[0]:.5f} and mean Q={cs[(cs.scenario == "interaction") & (cs.n == 20000) & (cs.variable == M) & (cs.method == "interaction")]["mean"].iloc[0]:.5f}. It learns heterogeneity, but derivative-magnitude variance remains. Additive models cannot represent this interaction and must be interpreted as misspecified here; the selected additive method also retains that restriction.

For joint economy+upgrade intervention, average the signed two-coordinate gradient over unaffected covariates first, then take its Euclidean norm. The truth is 2.313550. This is joint responsiveness in native coordinate units, not evidence of synergy. Only the original three methods were run for this auxiliary joint check.

{joint_table}

## Validation, theory and practical conclusion

Analytic derivatives agree with finite differences (base maximum error <5×10⁻⁹); efficient standardization agrees with explicit empirical averaging (<10⁻⁸), including joint interventions. The selected additive derivative and separate quadrature checks also pass. Quadrature refinement from 24 to 96 nodes per spline segment changes checked values by at most .005308 (scaled error <.000728); this is finite numerical error, smaller than the difficult-estimation errors, not a claim of exact integration. Replicate SD bars are simulation dispersion, not confidence intervals. No bootstrap inference is validated for these flexible norm estimates.

The separate estimation draft now distinguishes raw-coordinate flexible regression, support-dependent total standardization, and the conditional Q target. It gives a sufficient cross-fitted Q consistency result under joint-law L1 gradient consistency and moment control, and explains why global P requires stronger product-law gradient control/support. A fixed finite basis does not automatically satisfy those asymptotic conditions. Original correctly specified mechanism results remain separate.

The practical answer is yes: compare a linear baseline with smooth methods without prescribing exact structural forms. These comparisons recover important curvature and expose the limitations of a purely linear response. They still require causal assumptions and explicit response/support choices. For the paper, retain the full favorable and unfavorable results, identify extrapolations, and avoid presenting prediction accuracy or a single RMSE as proof of causal-gradient accuracy.

## Sources and files

- [Generalized cross-validation: Golub, Heath and Wahba (1979)](https://doi.org/10.1080/00401706.1979.10489751).
- [Random features: Rahimi and Recht (2007)](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html).
- [BIC: Schwarz (1978)](https://doi.org/10.1214/aos/1176344136).
- [Global estimates, bias and RMSE](results/global_summary.csv); [all global dataset estimates](results/global_replicate_estimates.csv).
- [Conditional estimates, bias and RMSE](results/conditional_summary.csv); [all conditional dataset estimates](results/conditional_replicate_estimates.csv).
- [Signed conditional derivative estimates](results/conditional_signed_summary.csv); [changed-shape positive/negative contributions](results/shape_direction_summary.csv).
- [Joint summary](results/joint_summary.csv); [changed-shape summary](results/shape_challenge_summary.csv).
- [Earlier exact-feature benchmark](../learned_mechanisms/REPORT.html); [initial customer experiment](../REPORT.html).
- [Estimation and replication notes](../../../docs/REPRODUCIBILITY.md).
"""
    # Avoid implying the literal response equals .3×10 as an estimator operation.
    text_report = text_report.replace("P=.3×10=3", "P=3")
    (ROOT / "REPORT.md").write_text(text_report)
    body = render_md(text_report)
    captions = [
        ("01_actual_global_estimates", "Actual global estimates"),
        ("02_noise_supported_estimates", "Supported conditional estimates under noise"),
        ("03_interaction_distinct_targets", "Different targets under interaction"),
        ("04_unknown_shape_and_direction", "Unknown shape and signed direction"),
        ("05_degree_curves", "Degree curves interpreted by the manuscript theorem"),
        ("06_joint_root_response", "Joint root intervention"),
        ("07_prediction_and_derivatives", "Prediction versus derivative accuracy"),
        ("08_selected_nonlinearity", "Data-selected nonlinear components"),
    ]
    gallery = "".join(
        f'<h2>{caption}</h2><a href="figures/{name}.pdf"><img src="figures/{name}.png" alt="{caption}"></a>'
        for name, caption in captions
    )
    (ROOT / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Flexible customer PEACE estimation</title><style>body{max-width:1160px;margin:35px auto;padding:0 25px;font:17px/1.6 system-ui;color:#182a35}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:7px;border-bottom:1px solid #ccd5da}th{background:#eaf0f2}img{width:100%;height:auto}a{color:#126a88}</style>'
        + body
        + gallery
    )
    (ROOT / "README.md").write_text("""# Flexible customer estimation follow-up

Read REPORT.html (full tables, figures and limitations) or REPORT.md. This folder is independent of the manuscript and sodium study. The exact-feature benchmark remains in ../learned mechanisms.

To reproduce, use a Python environment with numpy, scipy, pandas, scikit-learn and matplotlib. Set numerical-library thread counts to 1 for stable resource use. Run, from this folder: `python study.py validate`, `python study.py run --reps 40`, `python selected_study.py`, `python joint.py`, `python shape_challenge.py`, `python audit.py`, `python selected_audit.py`, `python report.py`. The study imports the preserved parent experiment's synthetic generator and oracle scoring helpers. Truth functions never choose regression basis, regularization or nonlinear components. No real customer observations are used.

All fold CSVs are retained. Summary CSVs first average two folds within each dataset; Monte Carlo SE is replicate SD/sqrt(40), not a causal confidence interval. Shape challenges and joint checks have separate seeds. Generic global estimates are withheld for discount/ad spend; linear/additive global values there explicitly require extrapolation. Reported bugs is not fitted because conditional variation is deterministic. Native-unit magnitudes should not be interpreted as scale-invariant variable rankings.
""")
    provenance = {
        "dataset_cases": 320,
        "joint_dataset_cases": 200,
        "shape_dataset_cases": 80,
        "replicates_per_case": 40,
        "folds": 2,
        "global_fold_rows": len(gf),
        "conditional_fold_rows": len(cf),
        "checks": checks,
        "assumptions": "Known graph and valid unaffected adjustment sets, causal identification and support; no exact structural response functions supplied. Linear/additive global discount/ad estimates require extra extrapolation restrictions.",
        "development": "Selected additive was added after noisy derivative instability; earlier results retained.",
        "generator_sha256": hashlib.sha256(
            (ROOT.parent / "experiment.py").read_bytes()
        ).hexdigest(),
        "repository_reference": "https://github.com/joseffaghihi/Multivariable-PEACE at 70fb6c2c3f51d0a76fd5ec909e53ade4b18b6307",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.glob("*.py")
        },
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name != "manifest-sha256.json"
    }
    (ROOT / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2))
    print(
        "Reports, eight PNG/PDF figures, full estimates and validation manifest complete."
    )


if __name__ == "__main__":
    main()
