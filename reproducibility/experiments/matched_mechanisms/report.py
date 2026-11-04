from experiment import (
    FIG,
    K,
    NOISE,
    PI,
    RES,
    RHO,
    ROOT,
    WIDTH,
    hashlib,
    json,
    leggauss,
    np,
    oracle,
    oracle_arrays,
    pd,
    response,
    sample_t,
    score,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.optimize import brentq
import html, re, platform

COLOR = {"A": "#166e73", "B": "#b65d46"}


def mdtable(df):
    rows = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row)
            + " |"
        )
    return "\n".join(rows)


def render_md(source):
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
        return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", s)

    out = []
    for block in source.strip().split("\n\n"):
        lines = block.splitlines()
        if lines[0].startswith("|"):
            lines = [l for l in lines if not re.fullmatch(r"[|: \-]+", l)]
            out.append(
                "<table>"
                + "".join(
                    "<tr>"
                    + "".join(
                        f"<{'th' if k == 0 else 'td'}>{inline(c.strip())}</{'th' if k == 0 else 'td'}>"
                        for c in l.strip("|").split("|")
                    )
                    + "</tr>"
                    for k, l in enumerate(lines)
                )
                + "</table>"
            )
        elif lines[0].startswith("#"):
            n = len(lines[0]) - len(lines[0].lstrip("#"))
            out.append(f"<h{n}>" + inline(lines[0].lstrip("# ")) + f"</h{n}>")
        elif all(l.startswith("- ") for l in lines):
            out.append(
                "<ul>"
                + "".join("<li>" + inline(l[2:]) + "</li>" for l in lines)
                + "</ul>"
            )
        else:
            out.append("<p>" + inline(block) + "</p>")
    return "\n".join(out)


def save(name):
    plt.savefig(FIG / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    plt.close()


def summaries(frame):
    def aggregate(g):
        err = g.estimate - g.truth
        return pd.Series(
            dict(
                truth=g.truth.iloc[0],
                mean=g.estimate.mean(),
                bias=err.mean(),
                rmse=np.sqrt(np.mean(err**2)),
                sd=g.estimate.std(ddof=1),
                mcse=g.estimate.std(ddof=1) / np.sqrt(len(g)),
                replicates=len(g),
            )
        )

    return (
        frame.groupby(["n", "model", "d", "r"])
        .apply(aggregate, include_groups=False)
        .reset_index()
    )


def covariances(tv, d, r):
    p = RHO ** (2 * d)
    p /= p.sum()
    q = PI**r
    q /= q.sum()
    J = tv / WIDTH
    profile = J @ p
    dprofile = 2 * ((J * np.log(RHO)) @ p - profile * (p @ np.log(RHO)))
    P = q @ profile
    return float(q @ dprofile), float(q @ (profile * np.log(PI)) - P * (q @ np.log(PI)))


def main():
    d = pd.read_csv(RES / "calibration_results.csv")
    s = summaries(d)
    s.to_csv(RES / "calibration_summary.csv", index=False)
    assert (
        len(d) == 4800
        and np.isfinite(d[["estimate", "truth", "squared_estimate", "squared_truth"]])
        .all()
        .all()
        and (s.replicates == 40).all()
    )
    pair = d.pivot(
        index=["n", "rep", "d", "r"], columns="model", values="estimate"
    ).reset_index()
    pair["estimate_difference"] = pair.A - pair.B
    pair["truth_difference"] = [
        score(oracle_arrays("A")[0], dd, rr) - score(oracle_arrays("B")[0], dd, rr)
        for dd, rr in zip(pair.d, pair.r)
    ]
    pair["error"] = pair.estimate_difference - pair.truth_difference
    pair.to_csv(RES / "paired_surface_differences.csv", index=False)
    ranking = []
    for (n, dd, rr), g in pair[(pair.d == 1) & (pair.r.isin([0, 1]))].groupby(
        ["n", "d", "r"]
    ):
        ranking.append(
            dict(
                n=n,
                d=dd,
                r=rr,
                true_difference=g.truth_difference.iloc[0],
                mean_difference=g.estimate_difference.mean(),
                sd=g.estimate_difference.std(ddof=1),
                mcse=g.estimate_difference.std(ddof=1) / np.sqrt(len(g)),
                correct_rankings=int(
                    (g.estimate_difference * g.truth_difference > 0).sum()
                ),
                replicates=len(g),
            )
        )
    ranking = pd.DataFrame(ranking)
    ranking.to_csv(RES / "ranking_recovery.csv", index=False)
    baseline = pd.read_csv(RES / "matched_baselines.csv")
    oracle = pd.read_csv(RES / "oracle_scores.csv")
    surface = pd.read_csv(RES / "oracle_surfaces.csv")
    steps = pd.read_csv(RES / "finite_intervention_steps.csv")
    audit = pd.read_csv(RES / "response_density_audit.csv")
    errors = []
    for (n, dd, rr), g in audit.groupby(["n", "d", "r"]):
        pp = g.pivot(
            index="rep",
            columns="model",
            values=["truth", "response_only", "density_only", "combined"],
        )
        truth_delta = pp.truth.A - pp.truth.B
        for source in ["response_only", "density_only", "combined"]:
            diff = pp[source].A - pp[source].B
            errors.append(
                dict(
                    n=n,
                    d=dd,
                    r=rr,
                    source=source,
                    difference_rmse=float(np.sqrt(np.mean((diff - truth_delta) ** 2))),
                )
            )
    errors = pd.DataFrame(errors)
    errors.to_csv(RES / "error_sources_summary.csv", index=False)
    covariance = []
    fd_errors = []
    for model in K:
        tv, _ = oracle_arrays(model)
        for dd, rr in [(0, 0.5), (0.25, 1), (0.5, 1), (1, 0), (1, 1), (2, 2)]:
            dc, rc = covariances(tv, dd, rr)
            eps = 1e-5
            df = (score(tv, dd + eps, rr) - score(tv, dd - eps, rr)) / (2 * eps)
            rf = (score(tv, dd, rr + eps) - score(tv, dd, rr - eps)) / (2 * eps)
            fd_errors.extend([abs(dc - df), abs(rc - rf)])
            covariance.append(dict(model=model, d=dd, r=rr, d_slope=dc, r_slope=rc))
    pd.DataFrame(covariance).to_csv(
        RES / "degree_covariance_interpretation.csv", index=False
    )
    assert max(fd_errors) < 1e-7
    plt.rcParams.update({"font.size": 10})
    t = np.linspace(0, 1, 4001)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    axs[0].stairs(RHO, [0, 1 / 3, 2 / 3, 1], color="#555555", linewidth=2)
    axs[0].set_ylabel("Treatment reference density")
    axs[0].set_xlabel("Treatment T")
    axs[0].set_title("Same exposure distribution")
    for m in K:
        axs[1].plot(t, response(t, 0, m), label=m, color=COLOR[m], lw=1.3)
        tt = np.linspace(2 / 3, 1, 2000)
        axs[2].plot(tt, response(tt, 1, m), label=m, color=COLOR[m], lw=1.3)
    axs[1].set_title("Common profile: full response")
    axs[2].set_title("Rare profile: final interval enlarged")
    for ax in axs[1:]:
        ax.set_xlabel("Intervention value T")
        ax.set_ylabel("Conditional mean response")
        ax.legend(title="Mechanism")
    fig.tight_layout()
    save("01_same_distributions_different_paths")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    cols = [
        "uniform_TV_natural_profiles",
        "natural_absolute_derivative",
        "natural_squared_derivative",
        "output_variance",
        "T_first_sobol",
        "T_total_sobol",
        "Z_total_sobol",
        "Janzing_T_edge_strength",
    ]
    labels = [
        "Uniform TV",
        "Natural mean |slope|",
        "Natural mean slope²",
        "Outcome variance",
        "First Sobol T",
        "Total Sobol T",
        "Total Sobol Z",
        "Treatment-edge information",
    ]
    vals = baseline.set_index("model")
    axs[0].barh(
        labels,
        [vals.loc["A", col] / vals.loc["B", col] for col in cols],
        color="#7f9192",
    )
    axs[0].axvline(1, color="black", ls="--")
    axs[0].set_xlim(0, 1.2)
    axs[0].set_xlabel("A / B ratio")
    axs[0].set_title(
        "All listed summaries agree exactly\nEndpoint and signed contrasts also both zero"
    )
    axs[0].invert_yaxis()
    yy = np.linspace(-0.1, 1.1, 601)
    q, w = leggauss(800)
    phase = np.sin((q + 1) * np.pi / 2) ** 2
    dens = (
        np.exp(-0.5 * ((yy[:, None] - phase) / NOISE) ** 2)
        / (np.sqrt(2 * np.pi) * NOISE)
        @ (w / 2)
    )
    axs[1].plot(yy, dens, color=COLOR["A"], label="A")
    axs[1].plot(yy, dens, color=COLOR["B"], ls="--", label="B")
    axs[1].set_xlabel("Outcome Y")
    axs[1].set_ylabel("Density")
    axs[1].set_title(
        "Entire outcome distributions coincide\nFor natural and all declared escort weights"
    )
    axs[1].legend()
    fig.tight_layout()
    save("02_matched_comparators")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, column, title in zip(
        axs,
        ["difference", "squared_difference"],
        ["PEACE: A minus B", "Weighted squared derivative: A minus B"],
    ):
        v = surface.pivot(index="d", columns="r", values=column)
        lim = float(np.abs(v.to_numpy()).max())
        img = ax.pcolormesh(
            v.columns,
            v.index,
            v.to_numpy(),
            shading="auto",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        # Exact equality lines avoid spurious contour branches from floating-point cancellation.
        if column == "difference":
            ax.axhline(0, color="black", lw=1)
            ax.axhline(0.5, color="black", lw=1)
            ax.axvline(np.log(4) / np.log(9), color="black", lw=1)
        else:
            second = brentq(
                lambda dd: 7 * 0.5 ** (2 * dd) - 8 + 3 * 1.5 ** (2 * dd), 0.6, 1
            )
            ax.axhline(0.5, color="black", lw=1)
            ax.axhline(second, color="black", lw=1)
            ax.axvline(np.log(16) / np.log(9), color="black", lw=1)
        ax.set_xlabel("Profile degree r")
        ax.set_ylabel("Treatment degree d")
        ax.set_title(title)
        fig.colorbar(img, ax=ax, label="Difference; positive means A > B")
    fig.tight_layout()
    save("03_distinctive_degree_surface")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, rr in zip(axs[:2], [0, 1]):
        for model in K:
            tv, _ = oracle_arrays(model)
            dd = np.linspace(0, 2, 101)
            ax.plot(
                dd, [score(tv, xx, rr) for xx in dd], color=COLOR[model], label=model
            )
        ax.axvline(0.5, color="grey", ls=":")
        ax.set_xlabel("Treatment degree d")
        ax.set_ylabel("Mean PEACE")
        ax.set_title(f"Profile degree r={rr}")
        ax.legend()
    for model in K:
        rr = np.linspace(0, 2, 101)
        tv, _ = oracle_arrays(model)
        axs[2].plot(
            rr, [score(tv, 1, xx) for xx in rr], color=COLOR[model], label=model
        )
    axs[2].axvline(
        np.log(4) / np.log(9), color="grey", ls=":", label="Exact profile crossover"
    )
    axs[2].set_xlabel("Profile degree r")
    axs[2].set_ylabel("Mean PEACE")
    axs[2].set_title("Treatment degree d=1")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    save("04_interpretable_rank_reversals")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rr in zip(axs, [0, 1]):
        v = ranking[ranking.r == rr]
        ax.errorbar(
            v.n, v.mean_difference, yerr=v.sd, marker="o", capsize=4, color="#166e73"
        )
        ax.axhline(
            v.true_difference.iloc[0], color="black", ls="--", label="Exact difference"
        )
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("Calibration observations")
        ax.set_ylabel("Estimated PEACE A minus B")
        ax.set_title(f"d=1, r={rr}; 40 paired datasets")
        ax.legend(fontsize=8)
    for ax in axs:
        ax.set_xticks([2000, 8000, 32000], ["2,000", "8,000", "32,000"])
        ax.xaxis.set_minor_locator(NullLocator())
    fig.suptitle("Noisy-data recovery: points = mean, whiskers = replicate SD")
    fig.tight_layout()
    save("05_noisy_calibration_recovery")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rr in zip(axs, [0, 1]):
        for source, color in [
            ("response_only", "#287ab1"),
            ("density_only", "#a56a3b"),
            ("combined", "#166e73"),
        ]:
            v = errors[
                (errors.d == 1) & (errors.r == rr) & (errors.source == source)
            ].sort_values("n")
            ax.plot(
                v.n,
                v.difference_rmse,
                marker="o",
                label=source.replace("_", " "),
                color=color,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Calibration sample size")
        ax.set_ylabel("RMSE of A−B difference")
        ax.set_title(f"d=1, r={rr}")
        ax.legend(fontsize=8)
    for ax in axs:
        ax.set_xticks([2000, 8000, 32000], ["2,000", "8,000", "32,000"])
        ax.xaxis.set_minor_locator(NullLocator())
    fig.suptitle("Reference sample remains fixed at 20,000: density error can dominate")
    fig.tight_layout()
    save("06_response_and_density_errors")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rr in zip(axs, [0, 1]):
        for model in K:
            v = steps[
                (steps.model == model) & (steps.d == 1) & (steps.r == rr)
            ].sort_values("step", ascending=False)
            ax.plot(
                v.step,
                v.local_contrast_average,
                marker="o",
                color=COLOR[model],
                label=model + " finite local contrast",
            )
            ax.axhline(v.peace_limit.iloc[0], color=COLOR[model], ls="--")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Intervention step size, decreasing →")
        ax.set_ylabel("Mean absolute local response change / step")
        ax.set_title(f"d=1, r={rr}; dashed = PEACE limit")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save("07_operational_local_change_limit")
    # Actual example calibration observations; oracle response separate from observed outcome.
    n = 8000
    rep = 0
    rng = np.random.default_rng(640000 + rep + n)
    tt = sample_t(n, rng)
    zz = np.r_[np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)]
    rng.shuffle(zz)
    ym = np.column_stack([response(tt, zz, m) for m in K]) + rng.normal(
        0, NOISE, (n, 2)
    )
    pd.DataFrame({"T": tt, "profile": zz, "Y_A": ym[:, 0], "Y_B": ym[:, 1]}).to_csv(
        RES / "example_calibration_observations.csv", index=False
    )
    key = oracle[(oracle.d.isin([0, 0.5, 1])) & (oracle.r.isin([0, 1]))][
        ["model", "d", "r", "peace"]
    ]
    key.to_csv(RES / "key_oracle_scores.csv", index=False)
    selected = s[(s.n == 8000) & (s.d == 1) & (s.r.isin([0, 1]))][
        ["n", "model", "d", "r", "truth", "mean", "bias", "rmse", "mcse"]
    ]
    baseline_long = (
        baseline.set_index("model").T.reset_index().rename(columns={"index": "Summary"})
    )
    text = f"""# Third experiment: identical conventional summaries, different causal-variation geometry

**Main finding:** two explicitly constructed causal mechanisms have exactly the same endpoint effects, signed average derivatives, natural absolute and squared derivative summaries, entire output distributions, Sobol indices and information-based treatment-edge strength. Their PEACE surfaces differ and reverse ordering as the declared profile emphasis changes. This is an exact separation result about information content, not a prediction leaderboard.

The mathematically defensible statement is: **the PEACE degree surface is not determined by the collection of matched summaries in this example.** It is not a claim that PEACE is the only possible weighted sensitivity measure or that all existing causal estimands are inadequate. The strengthened squared-derivative comparator also distinguishes some regimes, but measures a different kind of variation and has different crossovers.

## Mechanisms and scientific question

Think of two controllers whose standardized setting T lies in [0,1], with common and rare operating profiles Z. Both controllers produce outcomes of the same amplitude and the same distribution. They differ in how often and where their response rises and falls as T changes. The question is how local causal movement is distributed over settings and profiles when deciding what requires close response monitoring. Larger PEACE means more weighted movement, not a more beneficial treatment or better controller.

T has three equal-width intervals with densities (.5,1,1.5), probabilities (1/6,1/3,1/2). Profiles have probabilities (.9,.1), and T is independent of Z by randomized assignment. Within interval j, put u=3T−j and h=sin²(pi*k*u). Mechanism A uses cycle counts (3,3,1) in common profiles and (16,4,8) in rare profiles. B uses (4,1,2) and (12,12,4), respectively. Y=h+independent Gaussian noise SD .02. Both responses and their first treatment derivatives join continuously at interval boundaries. All units/supports/amplitudes are identical.

The causal response is h(t,z)=E[Y_do(T=t)|Z=z]. There are no mediators, so direct and total treatment responses coincide. Independent additive noise makes the structural treatment gradient and conditional-mean gradient equal, satisfying the paper's observable-gradient bridge. No hidden confounding is introduced. This is not a claim that the graph or identification assumptions can be discovered from the outcome histogram.

## What the comparators see

The equalities below are analytic, with independent numerical checks. Output-noise variance is .0004. The information quantity is Janzing et al.'s treatment-edge strength in this independent-parent DAG, equal to I(T;Y|Z), in nats. It is not an indiscriminate replacement of their definition by conditional mutual information in general DAGs.

{mdtable(baseline_long)}

All conditional boundary contrasts, including the corresponding quantile treatment contrasts for any quantile level, are zero: both endpoints have the same Gaussian noise distribution. This does NOT mean that quantile treatment effects for arbitrary interior treatment pairs are zero. Complete conditional response curves or well-chosen interior contrasts distinguish the mechanisms. Signed average derivatives also vanish under every declared escort weighting because each treatment interval starts and ends at zero.

For any positive integer k, sin²(pi*k*U) with U uniform has the same Beta(1/2,1/2) distribution. Changing integer cycle count changes the path but not its output histogram. The equality persists under every declared piecewise-constant treatment-density power and categorical profile reweighting. Gaussian convolution preserves it. Thus marginal outcome quantiles/moments and treatment-edge information strength remain equal even when these reference weights change. The Sobol decomposition also remains equal under the same escort laws because the common/rare centered cosine components have different integer frequencies within each interval and are orthogonal. The FULL joint conditional response law given a particular T,Z is different; this is not statistical indistinguishability of the complete experiments.

## What PEACE adds

Within a treatment interval, accumulated absolute movement is 2k. For profile z and mechanism M, use P_M(d,r)=sum_z q_r(z) sum_j [2k_M,jz*rho_j^(2d)]/B_d, where B_d=(1/3)sum_j rho_j^(2d) and q_r(z)=pi_z^r/sum pi^r. Profile Z is categorical and r uses counting measure. Its categories are fixed, not redefined during analysis.

The pair was constructed from a=(4,1,2), b=(3,3,1), which have identical sums, density-weighted sums and density-weighted squared sums. Nevertheless,

P_A−P_B = 2[.5^(2d)−2+1.5^(2d)]/B_d × [4*.1^r−.9^r]/[.9^r+.1^r].

This gives exact equality at d=0, d=.5 and r*=log(4)/log(9)=.630929754, with opposite orderings across those lines. At d=1,r=1, B has greater weighted local movement; at d=1,r=0, A does. The responses are held fixed: changing priorities changes the target functional, not the causal mechanism.

{mdtable(key)}

At r=1, common/rare profile weights are .9/.1. At r=0, they become .5/.5: rare-profile movement receives more relative importance. At d=1, treatment-density-squared weighting emphasizes frequently used settings. This explains the ordering reversal in terms of a declared scientific emphasis. No test outcomes or predictive scores choose degrees; the grid and comparison points were fixed analytically before calibration.

Curve interpretation uses the paper's log-density identity: dP/dd=2 E_(q_r)[Cov_(p_d)(|h'|,log f_T|Z)]. The corresponding profile identity is dP/dr=Cov_(q_r)(P_d(Z),log pi_Z), as derived in the standalone estimation work. These formulas explain changing slopes; a negative degree slope concerns placement of MAGNITUDES, not a negative treatment direction. Positive and negative variation are each half of PEACE here, and their signed difference is zero. Numerical differentiation agrees with the covariance identities to the reported audit tolerance.

## Strong comparator: weighted derivative energy

We also compute E_(p_d q_r)[h'^2] using precisely the same escort laws. It is not forced to remain at its natural reference distribution. It can distinguish the mechanisms, which is retained rather than hidden. Its profile crossover is log(16)/log(9)=1.261859507 rather than .630929754, because it squares cycle-frequency magnitudes. Its treatment factor is 7*.5^(2d)−8+3*1.5^(2d). Hence it answers a derivative-energy question, whereas PEACE answers accumulated-absolute-movement. A weighted absolute-derivative functional with the same laws is exactly the normalized PEACE gradient representation and cannot be distinguished from it. The new experiment therefore isolates PEACE's specific estimand relative to the listed summaries, not exclusive novelty among all weighted derivative measures.

## Recovery from noisy calibration observations

Forty paired calibration datasets are generated at each N=2,000, 8,000, 32,000. Profile-stratified calibration deliberately allocates half the observations to each profile; that sampling law is not the natural profile population. T follows its natural randomized density within each profile. The two mechanisms share setting/profile draws and the unlabeled reference sample but have independent outcome noises. A separate 20,000-observation natural reference sample estimates density-bin and profile probabilities. Its size is held fixed across N.

Unknown responses are fitted in a predeclared finite spectral family: twenty sin²(pi*m*u) candidates per treatment interval and profile. All coefficients are fitted by least squares; the true frequencies/coefficients are not supplied or selected using oracle answers. This family does assume bounded oscillatory responses returning to baseline with zero slopes at interval boundaries. Those boundary restrictions are assumptions, not successful discovery of zero endpoint effects from data. This is model-conditional recovery of an exact separation example, not unrestricted nonparametric estimation. It complements rather than replaces the flexible observational estimators in the first two experiments.

Actual Mean PEACE estimates, N=8,000, averaged over 40 datasets:

{mdtable(selected)}

Rank recovery at the two predeclared profile priorities:

{mdtable(ranking)}

The response-only audit uses known reference weights; the density-only audit uses known responses. They diagnose error sources and are not inputs to the original estimates. With a fixed 20,000-point reference sample, density uncertainty can dominate the combined error even as calibration N increases. The figures and CSVs retain this plateau and nonmonotone finite-replicate fluctuations. A larger response sample alone cannot eliminate a reference-density error floor. No claim of asymptotic rates or confidence intervals for these estimates is made from forty simulations.

## Operational interpretation and resolution limit

For a small admissible treatment step, average |h(t+delta/2,z)−h(t−delta/2,z)|/delta under the declared escort reference law. This finite local-intervention quantity approaches PEACE as delta tends to zero. Centers are restricted to [delta/2,1−delta/2] and their treatment weights renormalized; no silent out-of-domain intervention is used. The finite-step checks show convergence to the analytic values. Coarse steps can miss intervening cycles, so this example also exposes a limit: an infinitesimal variation metric is not automatically the appropriate quantity for large, indivisible treatment changes or a controller with insufficient setting resolution.

## What this establishes for the paper

This is a more direct third main experiment than the earlier prediction/acquisition studies. It supplies an exact witness that a causal-variation surface can contain information not recoverable from a rich collection of equal conventional summaries, including a distribution-aware information measure. Its controlled rank reversals clarify what d and r do. It does not rely on a restricted predictor being deliberately unable to fit rare cases. No claim is made that A or B has greater universally defined causal strength: each metric declares a question and reference law.

The example does not prove historical uniqueness of PEACE's definition, superiority to every causal measure, or inability of full response curves to identify the differences. Weighted derivative energy and complete conditional contrasts can reveal differences too. Its defensible contribution is the explicit mathematical and empirical distinction between effect geometry, output-distribution information, signed averages and reference-priority weighting. The previous acquisition application can remain supplementary; this experiment is the stronger candidate for the main third illustration.

## Verification, sources and files

Matching identities and score fractions are exact. The full output-law equality is verified through a phase coupling, not merely similar histograms. Gaussian-mixture entropy quadrature is refined independently; positive noise makes the information quantity finite. Response derivatives agree with finite differences, and spectral-extremum integration agrees with dense independent variation calculations. Separate direct quadrature of the response verifies the outcome moments, Sobol quantities, PEACE and derivative energy under all twenty declared reference laws. Full expected counts, finite estimates, probability normalizers, paired ranking and covariance identities are checked. The replay of original combined estimates in the error-source audit matches them numerically. All calibration datasets/degree combinations are retained; the original design hashes are retained alongside the published source hash.

- [Mathematical argument and scope](Mathematical_argument.md), [declared design](DESIGN.md).
- [Matched comparator values](results/matched_baselines.csv), [oracle PEACE/energy scores](results/oracle_scores.csv), [dense surfaces](results/oracle_surfaces.csv).
- [Full calibration summary](results/calibration_summary.csv), [individual estimates](results/calibration_results.csv), [rank recovery](results/ranking_recovery.csv).
- [Example observed calibration data](results/example_calibration_observations.csv), [response/density error audit](results/response_density_audit.csv), [finite intervention steps](results/finite_intervention_steps.csv).
- [Covariance interpretation](results/degree_covariance_interpretation.csv), [reproduction instructions](README.md).

Closest primary references: [Janzing et al., Quantifying causal influences, especially Theorem 4](https://webdav.tuebingen.mpg.de/causality/AOS2013-Janzing.pdf); [Sobol' and Kucherenko, derivative-based sensitivity](https://www.sciencedirect.com/science/article/pii/S0378475409000354); [Hines et al., signed weighted average derivative effects, original preprint](https://arxiv.org/abs/2109.13124). Comparator equalities here are new calculations for this construction, not claims that those authors proposed the same example.
"""
    (ROOT / "REPORT.md").write_text(text)
    names = [
        "01_same_distributions_different_paths",
        "02_matched_comparators",
        "03_distinctive_degree_surface",
        "04_interpretable_rank_reversals",
        "05_noisy_calibration_recovery",
        "06_response_and_density_errors",
        "07_operational_local_change_limit",
    ]
    gallery = "".join(
        f'<h2>{n.replace("_", " ")}</h2><a href="figures/{n}.pdf"><img src="figures/{n}.png" alt="{n}"></a>'
        for n in names
    )
    (ROOT / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Matched causal summaries and PEACE geometry</title><style>body{max-width:1180px;margin:35px auto;padding:0 24px;font:17px/1.6 system-ui;color:#20323c}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px;border-bottom:1px solid #ccd5d9}th{background:#eaf1f1}img{width:100%}a{color:#126e78}</style>'
        + render_md(text)
        + gallery
    )
    (ROOT / "README.md").write_text("""# Matched-summary causal mechanism experiment

Read REPORT.html for the complete argument, comparisons, actual estimates and seven figures. Mathematical_argument.md contains the compact proof; DESIGN.md was fixed before calibration. The original estimation document, paper and earlier studies remain unchanged.

Reproduce with NumPy, SciPy, pandas and matplotlib: run python experiment.py validate; python experiment.py run; python density_audit.py; python independent_audit.py; python report.py as separate commands. The generator/source fixes sample sizes, all seeds and degree grid. Calibration oversamples rare profiles, with an independent 20,000-point natural reference sample supplying weights. Independent outcome noise is used for A and B. The response estimator fits an explicit 20-mode bounded spectral family, so this is model-conditional estimation, not unrestricted nonparametric learning. Extrema of the fitted trigonometric polynomial provide total variation. Exact analytic matching, not a finite noisy p-value, is the primary separation argument.

A full weighted squared-derivative comparator is retained. A weighted absolute derivative with the identical weights is PEACE's gradient representation, so exclusive superiority over that same functional is not claimed. Outcome distributions, information treatment-edge strength and the listed Sobol quantities coincide, but full conditional response curves and interior intervention contrasts differ. Larger PEACE means more local movement under specified priorities, not a better treatment/controller.
""")
    record = json.loads((ROOT / "design-record.json").read_text())
    assert (
        hashlib.sha256((ROOT / "experiment.py").read_bytes()).hexdigest()
        == record["published_source_sha256"]
    )
    assert (
        hashlib.sha256((ROOT / "DESIGN.md").read_bytes()).hexdigest()
        == record["design_sha256"]
    )
    independent = json.loads((RES / "independent_audit_checks.json").read_text())
    assert independent["passed"]
    validation = json.loads((RES / "validation.json").read_text())
    densitycheck = json.loads((RES / "density_audit_checks.json").read_text())
    assert validation["passed"] and densitycheck["passed"]
    checks = {
        "calibration_rows": len(d),
        "paired_replicates_per_size": 40,
        "all_finite": True,
        "covariance_finite_difference_max_error": max(fd_errors),
        "validation": validation,
        "density_replay": densitycheck,
        "independent_response_quadrature": independent,
        "passed": True,
    }
    (RES / "report_checks.json").write_text(json.dumps(checks, indent=2))
    provenance = {
        "python": platform.python_version(),
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.glob("*.py")
        },
        "scope": "Exact matched-summary separation; same fixed support/units; finite spectral model-conditional calibration; no exclusive novelty claim.",
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
    print(selected.to_string(index=False))
    print(ranking.to_string(index=False))
    print(
        "Matched-summary report, seven figures, mathematical argument and full calibration checks complete."
    )


if __name__ == "__main__":
    main()
