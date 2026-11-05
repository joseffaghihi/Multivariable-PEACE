"""Produce figures and a readable report from all saved weak-IV results."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment as study

OUT, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
SUMMARY = pd.read_csv(OUT / "nonlinear_summary.csv")
RESPONSES = pd.read_csv(OUT / "response_summary.csv")
COEFFICIENTS = pd.read_csv(OUT / "response_coefficients.csv")
LINEAR = pd.read_csv(OUT / "linear_summary.csv")
TARGETS = pd.read_csv(OUT / "targets.csv")
RHO = np.array(study.CONFIG["instrument_correlations"])
COLORS = {"OLS": "#72777d", "IV": "#2463a6", "Ridge": "#c66820", "truth": "#17242d"}
METHODS = [
    ("OLS cubic", 0.0, "Ordinary regression", COLORS["OLS"]),
    ("IV degree 3", 0.0, "Cubic IV", COLORS["IV"]),
    ("Ridge IV degree 3", 0.0001, "Cubic ridge IV, λ = 10⁻⁴", COLORS["Ridge"]),
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def subset(n, method, penalty=0.0, d=0.5):
    s = SUMMARY[
        (SUMMARY.n == n)
        & (SUMMARY.method == method)
        & np.isclose(SUMMARY.penalty, penalty)
        & np.isclose(SUMMARY.d, d)
    ]
    return s.set_index("rho").reindex(RHO)


def strength_axis(ax):
    ax.set_xticks(range(6), ["0.8", "0.5", "0.2", "0.1", "0.05", "0†"])
    ax.set_xlabel("Instrument correlation ρ (decreasing →)")
    ax.axvspan(4.5, 5.3, color="#eaeaea", zorder=-2)
    ax.set_xlim(-0.3, 5.3)


def figure_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    x = np.linspace(-2.5, 2.5, 600)
    h = 0.4 * x**3 + 0.5 * x**2 - 0.2 * x - 0.5
    derivative = 1.2 * x * x + x - 0.2
    axes[0].plot(x, h, color=COLORS["truth"], lw=2, label="Causal mean h(x)")
    axes[0].plot(
        x,
        h + 0.5 * x,
        "--",
        color=COLORS["OLS"],
        lw=2,
        label="Observational mean h(x) + 0.5x",
    )
    axes[0].set(
        title="The response is fixed across strengths",
        xlabel="Treatment x",
        ylabel="Mean outcome",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(x, derivative, lw=2, color=COLORS["truth"], label="Causal derivative")
    axes[1].plot(
        x, derivative + 0.5, "--", color=COLORS["OLS"], label="Observational derivative"
    )
    axes[1].fill_between(
        x, derivative, 0, where=derivative < 0, color="#c66820", alpha=0.25
    )
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(
        title="PEACE aggregates derivative magnitudes",
        xlabel="Treatment x",
        ylabel="Signed derivative",
    )
    axes[1].legend(fontsize=8)
    r = np.linspace(0.01, 0.8, 200)
    for k, label in [
        (1, "Linear mode: ρ"),
        (2, "Quadratic mode: ρ²"),
        (3, "Cubic mode: ρ³"),
        (5, "Quintic mode: ρ⁵"),
    ]:
        axes[2].plot(r, r**k, lw=1.8, label=label)
    axes[2].set(
        yscale="log",
        title="Information attenuation differs by mode",
        xlabel="Instrument correlation ρ",
        ylabel="Conditional-operator multiplier",
    )
    axes[2].legend(fontsize=8)
    save(fig, "01_mechanism_and_information")


def figure_performance():
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.4), sharey=True, layout="constrained"
    )
    truth = float(TARGETS[TARGETS.d == 0.5].truth.iloc[0])
    for ax, n in zip(axes, study.CONFIG["sample_sizes"]):
        ax.axhline(
            truth, color=COLORS["truth"], lw=1.5, ls="--", label="True Mean PEACE"
        )
        for offset, (method, penalty, label, color) in zip([-0.12, 0, 0.12], METHODS):
            s = subset(n, method, penalty)
            ax.errorbar(
                np.arange(6) + offset,
                s["median"],
                yerr=[s["median"] - s.q05, s.q95 - s["median"]],
                color=color,
                marker="o",
                lw=1.5,
                capsize=2,
                ms=4,
                label=label,
            )
        strength_axis(ax)
        ax.set(
            title=f"n = {n:,}", yscale="log", ylabel="Estimated Mean PEACE (d = 1/2)"
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Medians and 5th–95th percentiles across 500 datasets; logarithmic vertical scale",
        fontsize=12,
    )
    save(fig, "02_weak_instrument_performance")


def figure_response():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.9), layout="constrained")
    x = np.linspace(-2.5, 2.5, 500)
    for col, rho in enumerate([0.8, 0.2, 0.05]):
        for row in [0, 1]:
            ax = axes[row, col]
            truth = (
                0.4 * x**3 + 0.5 * x * x - 0.2 * x - 0.5
                if row == 0
                else 1.2 * x * x + x - 0.2
            )
            ax.plot(
                x,
                truth,
                lw=2.3,
                color=COLORS["truth"],
                ls="--",
                label="True causal response" if row == 0 else "True causal derivative",
            )
            for method, penalty, label, color in METHODS:
                s = COEFFICIENTS[
                    (COEFFICIENTS.n == 4000)
                    & (COEFFICIENTS.rho == rho)
                    & (COEFFICIENTS.repetition == 0)
                    & (COEFFICIENTS.method == method)
                    & np.isclose(COEFFICIENTS.penalty, penalty)
                ].iloc[0]
                coef = study.power_coefficients(
                    s[[f"theta_{j}" for j in range(6)]].to_numpy(dtype=float)
                )
                if row == 1:
                    coef = np.polynomial.polynomial.polyder(coef)
                ax.plot(
                    x,
                    np.polynomial.polynomial.polyval(x, coef),
                    lw=1.7,
                    color=color,
                    label=label,
                )
            ax.axhline(0, color="black", lw=0.6)
            ax.set(
                xlabel="Treatment x",
                ylabel="Mean outcome" if row == 0 else "Signed derivative",
            )
            if row == 0:
                ax.set_title(f"ρ = {rho:g}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Actual fitted curves for the prespecified dataset: repetition 0, n = 4,000",
        fontsize=12,
    )
    save(fig, "03_actual_response_and_derivatives")


def figure_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    truth = float(TARGETS[TARGETS.d == 0.5].truth.iloc[0])
    setups = [
        ("IV degree 3", 0.0, "No penalty"),
        ("Ridge IV degree 3", 1e-6, "λ = 10⁻⁶"),
        ("Ridge IV degree 3", 1e-4, "λ = 10⁻⁴"),
        ("Ridge IV degree 3", 1e-2, "λ = 10⁻²"),
    ]
    for color, (method, penalty, label) in zip(
        ["#2463a6", "#6c83b5", "#c66820", "#b53e4d"], setups
    ):
        s = subset(4000, method, penalty)
        axes[0].plot(range(6), s["median"], "o-", color=color, label=label)
    axes[0].set_title("Cubic model: fixed penalties change estimates")
    for method, penalty, label, color in [
        ("IV degree 3", 0.0, "Cubic IV", "#2463a6"),
        ("IV degree 5", 0.0, "Quintic IV", "#654b9c"),
        ("Ridge IV degree 3", 1e-4, "Cubic ridge IV", "#c66820"),
        ("Ridge IV degree 5", 1e-4, "Quintic ridge IV", "#a2537f"),
    ]:
        s = subset(4000, method, penalty)
        axes[1].plot(range(6), s["median"], "o-", color=color, label=label)
    axes[1].set_title("Additional response flexibility needs more information")
    for ax in axes:
        ax.axhline(truth, ls="--", color=COLORS["truth"], lw=1.5, label="True target")
        strength_axis(ax)
        ax.set(ylabel="Median Mean PEACE (n = 4,000, d = 1/2)")
        ax.legend(fontsize=8, loc="upper left")
    save(fig, "04_penalty_and_response_class")


def figure_degree_density():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    for ax, rho in zip(axes, [0.8, 0.2, 0.05]):
        ax.plot(TARGETS.d, TARGETS.truth, "k--", lw=2, label="True target at each d")
        for method, penalty, label, color in METHODS:
            s = SUMMARY[
                (SUMMARY.n == 4000)
                & (SUMMARY.rho == rho)
                & (SUMMARY.method == method)
                & np.isclose(SUMMARY.penalty, penalty)
            ].sort_values("d")
            ax.plot(s.d, s["median"], "o-", color=color, label=label)
        ax.set(
            title=f"ρ = {rho:g}",
            xlabel="Treatment degree d",
            ylabel="Median Mean PEACE",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(
        "Changing d changes the target; it does not restore instrument information (n = 4,000)",
        fontsize=12,
    )
    save(fig, "05_degree_sensitivity")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.7, 4.2), sharey=True, layout="constrained"
    )
    for ax, name, title in [
        (axes[0], "rmse", "Known marginal density"),
        (axes[1], "estimated_density_rmse", "Fitted normal marginal density"),
    ]:
        for method, penalty, label, color in METHODS:
            s = subset(4000, method, penalty)
            ax.plot(range(6), s[name], "o-", color=color, label=label)
        strength_axis(ax)
        ax.set(title=title, yscale="log", ylabel="Finite-run RMSE (d = 1/2)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle(
        "Density estimation affects precision; weak response recovery dominates at small ρ",
        fontsize=12,
    )
    save(fig, "06_density_sensitivity")


def figure_linear():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), layout="constrained")
    s = LINEAR[LINEAR.n == 1000].set_index("rho").reindex(RHO)
    for ax, suffix, title in [
        (axes[0], "coefficient_coverage", "Coefficient β coverage, n = 1,000"),
        (axes[1], "coverage", "Mean PEACE |β| coverage, n = 1,000"),
    ]:
        for method, label, color in [
            ("wald", "Conventional Wald", COLORS["Ridge"]),
            ("ar", "Anderson–Rubin", COLORS["IV"]),
        ]:
            name = method + "_" + suffix
            ax.errorbar(
                range(6),
                s[name],
                yerr=2 * s[name + "_mcse"],
                marker="o",
                color=color,
                label=label,
                capsize=2,
            )
        ax.axhline(0.95, color="black", ls="--", lw=1)
        strength_axis(ax)
        ax.set(title=title, ylabel="Coverage (±2 Monte Carlo SE)", ylim=(0.93, 1.005))
        ax.legend(fontsize=8)
    for n, color in zip(
        study.CONFIG["sample_sizes"], ["#2463a6", "#c66820", "#654b9c"]
    ):
        z = LINEAR[LINEAR.n == n].set_index("rho").reindex(RHO)
        axes[2].plot(range(6), z.ar_unbounded, "o-", color=color, label=f"n = {n:,}")
    strength_axis(axes[2])
    axes[2].set(
        title="Unbounded Anderson–Rubin sets",
        ylabel="Fraction unbounded",
        ylim=(-0.02, 1.02),
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Linear check: 5,000 datasets per cell; exact coefficient test, conservative magnitude projection",
        fontsize=12,
    )
    save(fig, "07_linear_uncertainty")


def population_controls():
    rows = []
    truth_theta = study.true_theta()
    for rho in RHO:
        for penalty in study.CONFIG["ridge_penalties"]:
            theta = truth_theta.copy()
            for k in range(1, 4):
                theta[k] *= rho ** (2 * k) / (rho ** (2 * k) + penalty)
            targets = study.target_values(theta)[0]
            for d, target in zip(study.DEGREES, targets):
                rows.append(
                    {
                        "rho": rho,
                        "penalty": penalty,
                        "d": d,
                        "population_penalized_target": target,
                        "theta_1": theta[1],
                        "theta_2": theta[2],
                        "theta_3": theta[3],
                    }
                )
    pd.DataFrame(rows).to_csv(
        OUT / "population_regularization_targets.csv", index=False
    )


def md_table(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def report():
    parts_md, parts_html = [], []

    def heading(title, level=2):
        parts_md.append("#" * level + " " + title)
        parts_html.append(f"<h{level}>{html.escape(title)}</h{level}>")

    def para(text):
        parts_md.append(text)
        parts_html.append("<p>" + html.escape(text) + "</p>")

    def equations(latex, readable):
        parts_md.append("\\[\n" + latex + "\n\\]")
        parts_html.append(
            '<div class="equation">'
            + html.escape(readable).replace("\n", "<br>")
            + "</div>"
        )

    def table(df):
        parts_md.append(md_table(df))
        parts_html.append(
            '<div class="tablewrap">'
            + df.to_html(index=False, escape=True, border=0)
            + "</div>"
        )

    def figure(name, caption):
        parts_md.append(f"![{caption}](figures/{name}.png)\n\n{caption}")
        parts_html.append(
            f'<figure><img src="figures/{name}.svg" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )

    heading("Weak instruments and estimation of mean-response PEACE", 1)
    para(
        "Separate numerical study for the Franklin Open major revision. The quadratic follow-up supports the manuscript IV example; the three previously published experiments are unchanged. The design and tuning grid were fixed before the simulation; all estimates, including extreme values and the zero-strength endpoint, are retained."
    )
    para(
        "Main finding: the specified causal response is recovered accurately with a strong valid instrument in the cubic model. Weakening the instrument makes derivative and PEACE estimates unstable, despite unchanged causal truth, confounding and treatment distribution. Regularization controls extreme estimates but can remove real curvature. This is a limitation of finite-sample IV response recovery, not a numerical integration error or a violation of exclusion."
    )

    heading("1. Exact setting and causal target")
    equations(
        r"W,U,R,E\overset{\mathrm{ind}}\sim N(0,1),\quad X=\rho W+\tfrac12U+\sqrt{\tfrac34-\rho^2}R,\quad Y=h(X)+U+\tfrac12E,",
        "W, U, R, E are independent N(0,1)\nX = ρW + 0.5U + √(0.75 − ρ²)R\nY = h(X) + U + 0.5E",
    )
    equations(
        r"g_{\mathrm{in}}(x)=\mathbb E[Y_x]=h(x)=0.4x^3+0.5x^2-0.2x-0.5,\qquad h'(x)=1.2x^2+x-0.2.",
        "g_in(x) = E[Y_x] = h(x) = 0.4x³ + 0.5x² − 0.2x − 0.5\nh′(x) = 1.2x² + x − 0.2",
    )
    para(
        "Only W, X and Y are observed; U is hidden. W changes treatment but has no direct effect on outcome. Interventions on X replace its treatment equation in this specified SCM. E[Y−h(X)|W]=0 holds at all strengths, while E[Y−h(X)|X=x]=0.5x. X is always standard normal and Cov(X,U)=0.5. In fact the entire law of (X,Y) is unchanged across strengths. The target is the population total mean response; there are no covariates or mediators, so r is absent."
    )
    equations(
        r"P_d=\frac{\int_{-2.5}^{2.5}|h'(x)|\exp(-dx^2)\,\mathrm dx}{\int_{-2.5}^{2.5}\exp(-dx^2)\,\mathrm dx}.",
        "Mean PEACE P_d = ∫[−2.5,2.5] |h′(x)| exp(−dx²) dx / ∫[−2.5,2.5] exp(−dx²) dx",
    )
    para(
        "This uses the observational N(0,1) treatment law restricted and normalized on the declared evaluation window. Its normalization constant cancels in Mean PEACE. All observed treatments, including those outside the evaluation window, enter IV fitting. The unnormalized PEACE and the normalizers are also saved in targets.csv and nonlinear_estimates.csv."
    )
    targets = TARGETS[["d", "truth", "population_OLS_target"]].copy()
    targets.columns = ["d", "True Mean PEACE", "Population ordinary-regression target"]
    table(targets.round(6))
    para(
        "The derivative is negative on (−1,1/6) and positive outside. PEACE uses its magnitude, so a larger number means greater weighted local movement, not necessarily a beneficial treatment. The true degree curve decreases over the declared grid: increasing d gives more weight to central, higher-density treatment values. The paper's covariance identity specializes to ∂P_d/∂d = −Cov_wd(|h′(X)|,X²). This describes how the weighting changes the causal movement summary; it does not claim a negative signed effect or universal monotonicity."
    )
    figure(
        "01_mechanism_and_information",
        "The unchanged causal and observational responses, their derivatives, and attenuation of response components by the instrument.",
    )

    heading("2. Identification and what the estimators know")
    para(
        "For normalized Hermite polynomials H_k, E[H_k(X)|W]=ρ^k H_k(W). At every positive ρ, completeness holds in the Gaussian L² response class, hence in the finite classes fitted here. At ρ=0 it fails: the conditional moments do not distinguish centered response curves. Zero is an unidentified reference case, marked † in the plots, not another weak-but-identified scenario."
    )
    para(
        "The main estimator learns a cubic response with empirical series 2SLS, using cubic functions of W as instruments for cubic functions of X. It receives only (W,X,Y), not the true response coefficients, true ρ, U, or a population first-stage operator. A quintic response/instrument class checks additional flexibility. Both classes contain the true curve, but this remains a model-conditional polynomial benchmark rather than unrestricted nonparametric estimation."
    )
    para(
        "Comparators are cubic ordinary regression; unregularized cubic IV; cubic ridge IV with fixed λ=10⁻⁶, 10⁻⁴ and 10⁻²; and unregularized/regularized quintic IV. The primary ridge value 10⁻⁴ was fixed before the run. No penalty is chosen using true PEACE or favorable results. There are 500 nonlinear repetitions for each of 18 n/strength cells (9,000 datasets), yielding 63,000 fitted responses and 315,000 degree-specific estimates. Sample sizes are 1,000, 4,000 and 16,000; strengths are 0.8, 0.5, 0.2, 0.1, 0.05 and 0. The datasets are paired across strengths and nested across n; repetitions within a cell are independent."
    )

    heading("3. Actual PEACE estimates, not just error scores")
    central = (
        SUMMARY[
            (SUMMARY.n == 4000)
            & (SUMMARY.d == 0.5)
            & (
                ((SUMMARY.method == "OLS cubic") & (SUMMARY.penalty == 0))
                | ((SUMMARY.method == "IV degree 3") & (SUMMARY.penalty == 0))
                | (
                    (SUMMARY.method == "Ridge IV degree 3")
                    & np.isclose(SUMMARY.penalty, 1e-4)
                )
            )
        ]
        .copy()
        .sort_values(["rho", "method"], ascending=[False, True])
    )
    labels = {
        "OLS cubic": "Ordinary regression",
        "IV degree 3": "Cubic IV",
        "Ridge IV degree 3": "Cubic ridge IV",
    }
    displayed = pd.DataFrame(
        {
            "ρ": central.rho.map(lambda v: "0†" if v == 0 else str(v)),
            "Method": central.method.map(labels),
            "Mean": central["mean"].map(lambda v: f"{v:.4f}"),
            "Median": central["median"].map(lambda v: f"{v:.4f}"),
            "5th–95th percentiles": [
                f"{lo:.4f}–{hi:.4f}" for lo, hi in zip(central.q05, central.q95)
            ],
            "Bias": central.bias.map(lambda v: f"{v:+.4f}"),
            "Finite-run RMSE": central.rmse.map(lambda v: f"{v:.4f}"),
        }
    )
    table(displayed)
    para(
        "The true value in every row is 1.12233598. The ranges summarize estimates across 500 independently generated datasets; they are not confidence intervals for a single dataset. All means, SDs, biases, RMSEs and conventional plug-in Monte Carlo error diagnostics are available in nonlinear_summary.csv. Just-identified IV estimates can have extremely heavy tails and nonexistent population moments: the displayed mean/bias/RMSE describe this finite run, not guaranteed population moments. Their plug-in mean_mcse and rmse_mcse therefore do not supply justified confidence bounds for population averages in those cases. Medians and percentiles are necessary for judging performance."
    )
    para(
        "At ρ=0.8, cubic IV has median 1.1204, 5th–95th percentiles 1.0755–1.1623 and finite-run RMSE 0.0275. All 500 estimates are within 10% of truth. Ordinary regression is very concentrated around the wrong target (about 1.3935) and none is within 10%. Precision of an observational regression does not eliminate causal confounding bias."
    )
    para(
        "At ρ=0.2, cubic IV has median 1.9476 and 5th–95th percentiles 0.9633–17.1338; only 13.2% are within 10% of truth. The primary ridge estimate has median 1.1913 and a narrower range 0.8451–2.1387, yet only 29.4% are within 10%. Its apparently closer median is not reliable recovery of the causal curve. At still weaker instruments the instability persists. Values at zero strength have no point-identification interpretation."
    )
    figure(
        "02_weak_instrument_performance",
        "Actual Mean PEACE distributions for three sample sizes at d=1/2. Points are medians; bars are 5th–95th percentiles, not confidence intervals. Logarithmic vertical scale. The zero-strength endpoint † is unidentified. Extreme observations beyond these percentiles are retained in the raw results.",
    )

    heading("4. Response curves, sample size and strength diagnostics")
    para(
        "For n=4,000, the median derivative L² error under the restricted natural treatment law is 0.0564 for cubic IV at ρ=0.8, compared with 1.9826 at ρ=0.2. At ρ=0.2 ridge reduces it to 1.0744, which is still substantial. A scalar PEACE estimate near truth can occur even with an inaccurate response derivative, so scalar agreement alone is insufficient. The fitted curves below use the prespecified repetition 0, not a dataset selected after inspecting results; aggregate performance is established by the repeated-simulation tables."
    )
    figure(
        "03_actual_response_and_derivatives",
        "Exact fitted responses and derivatives from one prespecified dataset (repetition 0, n=4,000). Vertical scales vary by panel to retain the complete fitted curves. These panels illustrate possible behavior; the repeated results quantify how often it occurs.",
    )
    sample = (
        SUMMARY[
            (SUMMARY.method == "IV degree 3")
            & (SUMMARY.d == 0.5)
            & SUMMARY.rho.isin([0.8, 0.5, 0.2])
        ]
        .copy()
        .sort_values(["rho", "n"], ascending=[False, True])
    )
    table(
        pd.DataFrame(
            {
                "ρ": sample.rho,
                "n": sample.n,
                "Median": sample["median"].map(lambda v: f"{v:.4f}"),
                "5th–95th percentiles": [
                    f"{lo:.4f}–{hi:.4f}" for lo, hi in zip(sample.q05, sample.q95)
                ],
                "Within 10% of truth": sample.within_10_percent.map(
                    lambda v: f"{100 * v:.1f}%"
                ),
            }
        )
    )
    para(
        "More data help when instrument information is adequate. At ρ=0.5, the fraction within 10% improves from 43.2% at n=1,000 to 98.0% at n=16,000. At ρ=0.2 it improves only from 11.6% to 21.0%; the cubic direction remains poorly measured over this sample-size range. This does not mean an identified positive-strength case can never improve at a much larger n."
    )
    diag = (
        RESPONSES[(RESPONSES.n == 4000) & (RESPONSES.method == "IV degree 3")]
        .set_index("rho")
        .reindex(RHO)
    )
    table(
        pd.DataFrame(
            {
                "ρ": RHO,
                "Median linear first-stage F": diag.median_first_stage_F.round(
                    2
                ).to_numpy(),
                "Population cubic multiplier ρ³": (RHO**3).round(6),
                "nρ⁶ (n=4,000)": (4000 * RHO**6).round(6),
            }
        )
    )
    para(
        "At ρ=0.2 the median linear F is 165.8, but the population cubic multiplier is 0.008 and nρ⁶ is only 0.256. A scalar linear first-stage diagnostic is not sufficient to certify nonlinear response or derivative recovery. The component scale nρ²ᵏ is explanatory for this Gaussian series design, not a universal sample-size theorem or a proposed new weak-instrument threshold."
    )

    heading("5. Penalty, response-class and degree sensitivity")
    para(
        "At n=4,000 and ρ=0.2, cubic median Mean PEACE changes from 1.9476 without a penalty to 1.8740, 1.1913 and 0.7914 for λ=10⁻⁶, 10⁻⁴ and 10⁻². The last value is below truth. Regularization suppresses noise and real response features together; it is not an identification repair. A fixed nonzero penalty also gives population shrinkage θ_k ↦ [ρ²ᵏ/(ρ²ᵏ+λ)]θ_k, so its asymptotic target is generally biased. Population penalized limits are saved separately."
    )
    para(
        "The quintic class introduces additional weak directions even though it contains the true cubic. At ρ=0.5 and n=4,000, unregularized quintic IV has median 1.5059 and a 5th–95th range 0.8837–13.3498, compared with cubic IV median 1.1154 and range 0.9577–1.3117. Thus favorable cubic results depend on the declared response-class restriction; additional flexibility is not cost free. The complete quintic results and every penalty are retained, including outcomes unfavorable to the method."
    )
    figure(
        "04_penalty_and_response_class",
        "Median estimates at n=4,000 and d=1/2 across the full prespecified penalty and response-class comparisons. The dashed line is the common causal target. Smaller visual spread or a median near truth does not establish unbiased recovery.",
    )
    figure(
        "05_degree_sensitivity",
        "Degree sensitivity for the fixed grid d=0, 1/4, 1/2, 1, 2. Each degree has its own true target, shown by the dashed curve. No degree is selected using ground-truth error. There is no covariate weighting parameter r in this design.",
    )
    para(
        "The primary experiment deliberately uses the known marginal density to isolate response estimation. A paired check replaces it with a normal density whose mean and variance are fitted using observed X. At ρ=0.8, the cubic-IV finite-run RMSE rises from 0.0275 to 0.0356. Density uncertainty therefore matters at good response precision. At ρ=0.2 the corresponding values are 14.1323 and 14.0978: density fitting does not explain the weak-IV instability. These are finite-run tail-sensitive RMSEs, not population guarantees. The density check is correctly specified and parametric, not a general robustness result for kernel density estimation. At d=0 the fitted-density and known-density results coincide exactly by construction."
    )
    figure(
        "06_density_sensitivity",
        "Paired known-density and fitted-normal-density calculations on the same estimated responses. Finite-run RMSE is shown on a logarithmic scale; unregularized IV tails require the additional medians and percentiles reported above.",
    )

    heading("6. What uncertainty should look like with a weak instrument")
    para(
        "A companion linear model uses h(x)=βx, β=1, with the same treatment and disturbance equations. Mean PEACE equals |β|=1 at every d. For each cell, 5,000 independent samples are represented by exact Wishart draws of their centered Gaussian cross-products. Conventional IV Wald intervals are compared with Anderson–Rubin confidence sets obtained by inverting the exact t test of W in regression of Y−bX on (1,W). The whole coefficient set, including unbounded/disconnected sets, is mapped through absolute value for PEACE."
    )
    ls = LINEAR[LINEAR.n == 1000].set_index("rho").reindex(RHO)
    table(
        pd.DataFrame(
            {
                "ρ": RHO,
                "Wald coefficient coverage": ls.wald_coefficient_coverage.map(
                    lambda v: f"{100 * v:.2f}%"
                ).to_numpy(),
                "AR coefficient coverage": ls.ar_coefficient_coverage.map(
                    lambda v: f"{100 * v:.2f}%"
                ).to_numpy(),
                "AR PEACE coverage": ls.ar_coverage.map(
                    lambda v: f"{100 * v:.2f}%"
                ).to_numpy(),
                "AR sets unbounded": ls.ar_unbounded.map(
                    lambda v: f"{100 * v:.2f}%"
                ).to_numpy(),
            }
        )
    )
    para(
        "Anderson–Rubin coefficient coverage is compatible with 95% throughout the grid; across all 18 cells its largest deviation is 1.89 Monte Carlo standard errors. Mapping to |β| can make coverage conservative because positive and negative coefficients have the same magnitude. At ρ=0.05 and n=1,000, coefficient coverage is 94.94%, PEACE coverage 97.54%, and 63.70% of the sets are unbounded. At zero strength 95.16% are unbounded. Coverage alone therefore does not mean precise or useful estimation."
    )
    para(
        "In this specific design, conventional Wald coverage becomes conservative as the instrument weakens (97.02% coefficient coverage at ρ=0.05 and n=1,000, and 99.06% at zero). These results do not show Wald undercoverage and should not be described that way. Its always-finite intervals do not supply the exact weak-instrument guarantee of the inverted test. The exact inference claim here depends on the specified Gaussian independent-disturbance linear model; it is not asserted for the nonlinear regularized estimator."
    )
    figure(
        "07_linear_uncertainty",
        "Linear inference check. Coverage error bars are ±2 Monte Carlo standard errors, not confidence intervals for causal effects. Anderson–Rubin sets can become unbounded, honestly reflecting weak information. Magnitude projection can increase coverage above 95%.",
    )

    heading("7. Numerical verification and limits of the conclusion")
    audit = json.loads((OUT / "audit.json").read_text())
    para(
        f"All independent checks passed. The Hermite conditional-expectation identity was checked by Gaussian quadrature; estimator coefficients were checked using independently constructed columns and projection equations. Integration was checked for {audit['integration']['checked_fits']} fitted responses, including extreme estimates, against adaptive integration and a doubled quadrature rule. Maximum relative adaptive error was {audit['integration']['max_relative_adaptive_error']:.2g}. Confidence-set projections were independently checked for 5,000 stored samples and against direct-data regressions. Every finite-run bias/SD/RMSE summary was recomputed. Manuscript hashes are unchanged. See results/audit.json."
    )
    para(
        "What this adds: a controlled numerical illustration of the manuscript's IV mean-response identification route, including valid-but-weak instruments, an explicitly unidentified endpoint, actual effect estimates, bias/variability, derivative recovery, regularization sensitivity and weak-IV-robust uncertainty in a linear check. It provides the missing numerical evidence requested by Reviewer 3 without claiming that PEACE is immune to weak instruments."
    )
    para(
        "What it does not establish: unrestricted nonparametric performance; validity under exclusion violations or arbitrary heterogeneous structural responses; a data-driven optimal penalty or degree; confidence intervals for nonlinear regularized PEACE; a general density estimator guarantee; predictive superiority; or a covariate-degree r result. The finite polynomial response restriction, additive Gaussian construction, fixed treatment window and correctly specified density check remain explicit in the manuscript summary."
    )

    heading("8. Reproduction and files")
    para(
        "Use README.md for commands and requirements-lock.txt for the tested Python environment. DESIGN.md records the prespecified design; Mathematical_argument.md gives the equations, identification argument, estimator and confidence-set derivation. experiment.py regenerates all raw and summary data; audit.py independently checks them; report.py generates this report and seven figures in PNG/SVG formats. The sources and reference results are distributed with the GitHub reproducibility package; the quadratic follow-up supplies the manuscript IV table."
    )
    refs = [
        (
            "Newey and Powell (2003), Instrumental Variable Estimation of Nonparametric Models",
            "https://doi.org/10.1111/1468-0262.00459",
        ),
        (
            "Darolles, Fan, Florens and Renault (2011), Nonparametric Instrumental Regression",
            "https://doi.org/10.3982/ECTA6539",
        ),
        (
            "Andrews, Stock and Sun (2019), Weak Instruments in Instrumental Variables Regression: Theory and Practice",
            "https://doi.org/10.1146/annurev-economics-080218-025643",
        ),
    ]
    heading("Methodological sources")
    for title, url in refs:
        parts_md.append(f"- [{title}]({url})")
        parts_html.append(f'<p><a href="{url}">{html.escape(title)}</a></p>')
    para(
        "Series conditional-moment inversion, regularization and weak-instrument inference follow established IV principles. The Gaussian/Hermite calculations and conclusions about this particular PEACE target are explicitly derived and checked here; they are not a new general IV estimator."
    )
    (ROOT / "REPORT.md").write_text("\n\n".join(parts_md) + "\n")
    style = """body{max-width:1200px;margin:40px auto;padding:0 24px;color:#1c2933;font:17px/1.65 Georgia,serif}h1,h2,h3{font-family:Arial,sans-serif;line-height:1.25}h1{font-size:32px}h2{font-size:24px;margin-top:42px}p{max-width:1060px}.equation{background:#f2f5f7;padding:18px 22px;border-left:4px solid #2463a6;font:18px/1.8 Georgia,serif;overflow-x:auto}.tablewrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;font:14px/1.5 Arial,sans-serif;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #d9e0e5;text-align:right;white-space:nowrap}th{background:#eef2f5}td:nth-child(2){text-align:left}figure{margin:28px 0}img{width:100%;height:auto}figcaption{font:14px/1.5 Arial,sans-serif;color:#4d5c67;margin-top:8px}a{color:#1b5d9e}@media print{body{max-width:none;font-size:12pt}figure,table{break-inside:avoid}}"""
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Weak instruments and PEACE</title><style>'
        + style
        + "</style></head><body>"
        + "\n".join(parts_html)
        + "</body></html>"
    )
    (ROOT / "REPORT.html").write_text(document)


if __name__ == "__main__":
    population_controls()
    figure_mechanism()
    figure_performance()
    figure_response()
    figure_sensitivity()
    figure_degree_density()
    figure_linear()
    report()
    print("Report and seven figures saved")
