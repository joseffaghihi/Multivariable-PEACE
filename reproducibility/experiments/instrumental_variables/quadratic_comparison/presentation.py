"""Estimation of a finite causal contrast and aggregated local causal movement."""

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
from matplotlib.patches import Rectangle

import recovery
from sensitivity import analyze as sensitivity_analysis

OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
S = pd.read_csv(OUT / "comparison_summary.csv")
L = pd.read_csv(OUT / "large_sample_summary.csv")
T = pd.read_csv(OUT / "targets.csv")
RHO = np.array([0.8, 0.5, 0.2, 0.1, 0.05, 0.0])
NAMES = {"peace": "Mean PEACE", "contrast": "Intervention contrast 0 → 1"}
METHODS = [
    ("OLS degree 2", 0.0, "Ordinary regression", "#72777d"),
    ("IV degree 2", 0.0, "Quadratic IV", "#2463a6"),
    ("Ridge IV degree 2", 1e-4, "Quadratic ridge IV, λ = 10⁻⁴", "#c66820"),
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


def select(frame=S, metric="peace", method="IV degree 2", penalty=0.0, n=4000):
    return (
        frame[
            (frame.scenario == "quadratic")
            & (frame.metric == metric)
            & (frame.method == method)
            & np.isclose(frame.penalty, penalty)
            & (frame.n == n)
            & (frame.d == 0.5)
        ]
        .set_index("rho")
        .reindex(RHO)
    )


def strength_axis(ax):
    ax.set_xticks(range(6), ["0.8", "0.5", "0.2", "0.1", "0.05", "0†"])
    ax.set(xlabel="Instrument correlation ρ (decreasing →)", xlim=(-0.3, 5.3))
    ax.axvspan(4.5, 5.3, color="#eaeaea", zorder=-2)


def primary(summary, rho=0.2, scenario="quadratic"):
    degree = 2 if scenario == "quadratic" else 3
    return summary[
        (summary.scenario == scenario)
        & (summary.rho == rho)
        & (summary.method == f"IV degree {degree}")
        & (summary.penalty == 0)
    ].sort_values("n")


def figures(raw, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    for ax, metric in zip(axes, ["contrast", "peace"]):
        ax.axhline(
            select(metric=metric).truth.iloc[0],
            ls="--",
            color="black",
            label="True target",
        )
        for offset, (method, penalty, label, color) in zip([-0.12, 0, 0.12], METHODS):
            a = select(metric=metric, method=method, penalty=penalty)
            ax.errorbar(
                np.arange(6) + offset,
                a["median"],
                yerr=[a["median"] - a.q05, a.q95 - a["median"]],
                marker="o",
                color=color,
                capsize=2,
                ms=4,
                lw=1.3,
                label=label,
            )
        strength_axis(ax)
        ax.set(title=NAMES[metric], ylabel="Estimated value")
        if metric == "peace":
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "Two summaries of the same estimated response: n = 4,000, d = 1/2", fontsize=12
    )
    save(fig, "01_response_summaries")

    a = primary(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(
        a.n, a.peace_accuracy, "o-", color="#72777d", label="All simulated datasets"
    )
    p = a.peace_accuracy_given_contrast_accuracy
    axes[0].errorbar(
        a.n,
        p,
        yerr=[p - a.conditional_wilson_lower, a.conditional_wilson_upper - p],
        marker="o",
        color="#2463a6",
        capsize=2,
        label="Among accurate contrast estimates",
    )
    axes[0].set(
        xscale="log",
        ylim=(-0.02, 1.03),
        xlabel="Sample size n",
        ylabel="Mean PEACE within 10% of truth",
        title="Recovery conditional on contrast accuracy",
    )
    axes[0].set_xticks(a.n, [f"{n // 1000}k" for n in a.n])
    axes[0].legend(fontsize=8)
    bottom = np.zeros(len(a))
    categories = [
        ("both_accurate_count", "Both accurate", "#36816e"),
        (
            "contrast_only_count",
            "Contrast accurate; PEACE outside tolerance",
            "#c66820",
        ),
        ("peace_only_count", "PEACE accurate; contrast outside tolerance", "#2463a6"),
        ("neither_accurate_count", "Neither accurate", "#a5abb0"),
    ]
    for column, label, color in categories:
        values = a[column].to_numpy() / a.repetitions.to_numpy()
        axes[1].bar(range(len(a)), values, bottom=bottom, color=color, label=label)
        bottom += values
    axes[1].set_xticks(range(len(a)), [f"{n // 1000}k" for n in a.n])
    axes[1].set(
        xlabel="Sample size n",
        ylabel="Fraction of 500 datasets",
        ylim=(0, 1.02),
        title="Paired accuracy states",
    )
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.suptitle(
        "Fixed ρ = 0.2; accuracy means relative error ≤10% for each target", fontsize=12
    )
    save(fig, "02_conditional_recovery")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    colors = {
        "both": "#36816e",
        "contrast_only": "#c66820",
        "peace_only": "#2463a6",
        "neither": "#a5abb0",
    }
    for ax, n in zip(axes, [4000, 64000, 256000]):
        g = raw[
            (raw.scenario == "quadratic")
            & (raw.n == n)
            & (raw.rho == 0.2)
            & (raw.method == "IV degree 2")
        ]
        statuses = [
            (g.peace_accurate & g.contrast_accurate, "both", "Both accurate"),
            (
                ~g.peace_accurate & g.contrast_accurate,
                "contrast_only",
                "Contrast accurate only",
            ),
            (
                g.peace_accurate & ~g.contrast_accurate,
                "peace_only",
                "PEACE accurate only",
            ),
            (~g.peace_accurate & ~g.contrast_accurate, "neither", "Neither accurate"),
        ]
        for mask, key, label in statuses:
            ax.scatter(
                g.loc[mask, "contrast_relative_error"],
                g.loc[mask, "peace_relative_error"],
                s=12,
                alpha=0.6,
                color=colors[key],
                label=label,
                rasterized=True,
            )
        ax.add_patch(
            Rectangle((-0.1, -0.1), 0.2, 0.2, fill=False, edgecolor="black", lw=1.2)
        )
        for value in [-0.1, 0.1]:
            ax.axvline(value, color="black", lw=0.6, ls="--")
            ax.axhline(value, color="black", lw=0.6, ls="--")
        ax.set(
            xscale="symlog",
            yscale="symlog",
            title=f"n = {n:,}",
            xlabel="Relative error in contrast",
            ylabel="Relative error in Mean PEACE",
        )
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_yscale("symlog", linthresh=0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(
        "Paired estimation errors from the same response fit: ρ = 0.2", fontsize=12
    )
    save(fig, "03_paired_errors")

    combined = pd.concat([S, L], ignore_index=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.3), sharey=True, layout="constrained"
    )
    for ax, rho in zip(axes, [0.2, 0.1, 0.05]):
        for method, penalty, label, color in METHODS[1:]:
            a = combined[
                (combined.scenario == "quadratic")
                & (combined.metric == "peace")
                & (combined.d == 0.5)
                & (combined.rho == rho)
                & (combined.method == method)
                & np.isclose(combined.penalty, penalty)
            ].sort_values("n")
            ax.plot(a.n, a["median"], "o-", color=color, label=label)
            ax.fill_between(a.n, a.q05, a.q95, color=color, alpha=0.14)
        ax.axhline(1.4183894880371266, color="black", ls="--", label="True Mean PEACE")
        ax.set(
            xscale="log",
            yscale="log",
            title=f"ρ = {rho:g}",
            xlabel="Sample size n",
            ylabel="Estimated Mean PEACE",
        )
        ax.set_xticks(
            [1000, 4000, 16000, 64000, 256000], ["1k", "4k", "16k", "64k", "256k"]
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Precision and weak-information sensitivity: medians and 5th–95th percentiles",
        fontsize=12,
    )
    save(fig, "04_weak_information")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for method, penalty, label, color in [
        ("IV degree 2", 0.0, "No penalty", "#2463a6"),
        ("Ridge IV degree 2", 1e-6, "λ = 10⁻⁶", "#6c83b5"),
        ("Ridge IV degree 2", 1e-4, "λ = 10⁻⁴", "#c66820"),
        ("Ridge IV degree 2", 1e-2, "λ = 10⁻²", "#b53e4d"),
    ]:
        a = select(method=method, penalty=penalty)
        axes[0].plot(range(6), a["median"], "o-", label=label, color=color)
    axes[0].set_title("Fixed penalty sensitivity")
    for method, penalty, label, color in [
        ("IV degree 2", 0.0, "Quadratic class IV", "#2463a6"),
        ("IV degree 3", 0.0, "Expanded cubic class IV", "#654b9c"),
        ("Ridge IV degree 2", 1e-4, "Quadratic class ridge", "#c66820"),
        ("Ridge IV degree 3", 1e-4, "Expanded cubic class ridge", "#a2537f"),
    ]:
        a = select(method=method, penalty=penalty)
        axes[1].plot(range(6), a["median"], "o-", label=label, color=color)
    axes[1].set_title("Response-class sensitivity")
    for ax in axes:
        strength_axis(ax)
        ax.axhline(1.4183894880371266, color="black", ls="--")
        ax.set_ylabel("Median Mean PEACE (n = 4,000)")
        ax.legend(fontsize=8)
    save(fig, "05_estimation_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    truth = T[(T.scenario == "quadratic") & (T.metric == "peace")].sort_values("d")
    for ax, rho in zip(axes, [0.8, 0.2, 0.05]):
        ax.plot(truth.d, truth.truth, "k--", lw=2, label="True target at each d")
        for method, penalty, label, color in METHODS:
            a = S[
                (S.scenario == "quadratic")
                & (S.metric == "peace")
                & (S.n == 4000)
                & (S.rho == rho)
                & (S.method == method)
                & np.isclose(S.penalty, penalty)
            ].sort_values("d")
            ax.plot(a.d, a["median"], "o-", color=color, label=label)
        ax.set(
            title=f"ρ = {rho:g}",
            xlabel="Treatment degree d",
            ylabel="Median Mean PEACE",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(
        "Recovery of aggregated local movement across declared degrees", fontsize=12
    )
    save(fig, "06_degree_recovery")


def md_table(frame):
    return "\n".join(
        [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def recovery_table(summary, rho=0.2):
    a = primary(summary, rho)
    return pd.DataFrame(
        {
            "n": a.n.map(lambda v: f"{v:,}"),
            "Contrast within 10%": a.contrast_accuracy.map(lambda v: f"{100 * v:.1f}%"),
            "Mean PEACE within 10%": a.peace_accuracy.map(lambda v: f"{100 * v:.1f}%"),
            "Mean PEACE accurate among accurate contrasts": [
                f"{row.both_accurate_count}/{row.contrast_accurate_count} ({100 * row.peace_accuracy_given_contrast_accuracy:.1f}%)"
                if row.contrast_accurate_count
                else "No accurate contrasts"
                for row in a.itertuples()
            ],
        }
    )


def compact(summary):
    combined = pd.concat([S, L], ignore_index=True)
    a = combined[
        (combined.scenario == "quadratic")
        & (combined.n == 64000)
        & (combined.rho == 0.2)
        & (combined.method == "IV degree 2")
        & (combined.d == 0.5)
        & combined.metric.isin(NAMES)
    ]
    estimated = pd.DataFrame(
        {
            "Quantity": a.metric.map(NAMES),
            "Truth": a.truth.map(lambda v: f"{v:.4f}"),
            "Median": a["median"].map(lambda v: f"{v:.4f}"),
            "5th–95th percentiles": [
                f"{lo:.4f}–{hi:.4f}" for lo, hi in zip(a.q05, a.q95)
            ],
        }
    )
    parts = [
        "# Estimating a finite contrast and aggregated local causal movement",
        "This is an estimation study of two summaries of the same fitted causal mean response. The finite contrast supplies a reference for studying recovery of Mean PEACE. Their scientific meanings differ, so their accuracy percentages are not a ranking of causal metrics. The selected quadratic results support the manuscript IV example.",
        "The SEM has W,U,R,E independent N(0,1), X=ρW+0.5U+√(0.75−ρ²)R, and Y=X+0.75(X²−1)+U+0.5E. Only W,X,Y are observed; U is hidden confounding. Intervening only on X gives h(x)=E[Y_x]=x+0.75(x²−1). Coefficients are learned in a declared polynomial response class.",
        "The two targets are Δ=h(1)−h(0)=1.75 and Mean PEACE P=∫|h′(x)|w(x)dx=1.4184, with normalized restricted natural treatment weights on (−2.5,2.5), d=1/2. Mean PEACE aggregates infinitesimal local contrast magnitudes across the domain. It is not calculated from the single finite contrast Δ alone.",
        "Actual estimates at ρ=0.2, n=64,000, across 500 simulated datasets (ranges are sampling-distribution percentiles, not confidence intervals):",
        md_table(estimated),
        "Paired recovery at fixed ρ=0.2, d=1/2; 500 datasets per row. An estimate is accurate here when its relative error is at most 10% of its own known truth:",
        md_table(recovery_table(summary)),
        "At n=64,000, 487 contrast estimates are accurate; 436 of these also have accurate Mean PEACE (89.5%). In 51 datasets the contrast is accurate but Mean PEACE is outside tolerance. At n=256,000 the conditional recovery frequency reaches 99.8%. This directly describes how reliably the local aggregation is recovered when the reference contrast is accurately estimated.",
        "An accurate finite contrast does not guarantee accurate local aggregation: errors in different response components can cancel in h(1)−h(0) while affecting |h′| across the treatment domain. For a fitted quadratic c+a x+b(x²−1), Δ=a+b while P=∫|a+2bx|w(x)dx. These conditional percentages are simulation diagnostics using known truth, not an accuracy test available from real data alone.",
        "Candidate explanatory remark for later discussion:",
        "The IV simulation evaluates recovery of a specified causal response through its finite intervention contrast and Mean PEACE. Recovery of the aggregated local variation improves with sample size, while weak information about response curvature can leave Mean PEACE uncertain even when the finite contrast is accurately estimated.",
        "The full report retains weak-information cases, the original cubic stress analysis, fixed-penalty and response-class sensitivity, density checks and all simulation data. [Full report and figures](REPORT.html).",
    ]
    (ROOT / "COMPACT_SUMMARY.md").write_text("\n\n".join(parts) + "\n")


def report(summary):
    md = []
    web = []

    def heading(text):
        md.append("## " + text)
        web.append("<h2>" + html.escape(text) + "</h2>")

    def para(text):
        md.append(text)
        web.append("<p>" + html.escape(text) + "</p>")

    def equation(latex, plain):
        md.append("\\[\n" + latex + "\n\\]")
        web.append(
            '<div class="equation">'
            + html.escape(plain).replace("\n", "<br>")
            + "</div>"
        )

    def table(frame):
        md.append(md_table(frame))
        web.append(
            '<div class="tablewrap">'
            + frame.to_html(index=False, border=0, escape=True)
            + "</div>"
        )

    def figure(name, caption):
        md.append(f"![{caption}](figures/{name}.png)\n\n{caption}")
        web.append(
            f'<figure><img src="figures/{name}.svg" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )

    title = "From an estimated causal response to a contrast and Mean PEACE"
    md.append("# " + title)
    web.append("<h1>" + html.escape(title) + "</h1>")
    para(
        "This separate exploratory IV study investigates recovery of two related summaries of the same fitted causal mean response: a finite intervention contrast and aggregated local causal movement. It evaluates estimation accuracy for each target and their paired recovery. The quantities answer different scientific questions; their accuracy percentages do not rank competing causal metrics. The manuscript and original cubic experiment remain unchanged."
    )
    heading("1. Full SEM, intervention and reference law")
    equation(
        r"W,U,R,E\overset{\mathrm{ind}}\sim N(0,1),\qquad X=\rho W+0.5U+\sqrt{0.75-\rho^2}R,\qquad Y=X+0.75(X^2-1)+U+0.5E,",
        "W,U,R,E independent N(0,1)\nX = ρW + 0.5U + √(0.75−ρ²)R\nY = X + 0.75(X²−1) + U + 0.5E",
    )
    para(
        "Observe W,X,Y. W is the instrument, U a hidden common cause of X,Y, R additional treatment noise, and E additional outcome noise. W→X→Y and U→X,U→Y are the relevant causal paths. Only the X equation is replaced under intervention; exogenous distributions are retained in this specified SEM. Thus h(x)=E[Y_x]=x+0.75(x²−1). The target does not condition on W; W supplies identifying moments. There are no covariates or mediators, hence no r parameter."
    )
    para(
        "Across instrument strengths the marginal law X~N(0,1), Cov(X,U)=0.5 and the entire law of (X,Y) are fixed. The mean-IV restriction E[Y−h(X)|W]=0 holds; ordinary regression instead targets h(x)+0.5x. Positive ρ satisfies Gaussian completeness. At zero the conditional moment operator loses the centered response directions, so that endpoint is explicitly unidentified."
    )
    heading("2. One finite contrast and aggregation of local contrasts")
    equation(
        r"\Delta_{0,1}=h(1)-h(0)=\int_0^1h'(x)\,\mathrm dx,\qquad P_d=\int_{-2.5}^{2.5}|h'(x)|w_d(x)\,\mathrm dx,\qquad w_d(x)=\frac{e^{-dx^2}}{\int_{-2.5}^{2.5}e^{-du^2}\,\mathrm du},",
        "Δ = h(1) − h(0) = ∫[0,1] h′(x) dx\nMean PEACE P_d = ∫[−2.5,2.5] |h′(x)| w_d(x) dx\nw_d(x) = exp(−dx²) / ∫[−2.5,2.5] exp(−du²) du",
    )
    para(
        "At d=1/2 these are normalized natural treatment weights restricted to the declared window. All observations still enter IV fitting. The derivative is the limit of local contrasts divided by treatment increments; PEACE aggregates their magnitudes across the window. The single contrast Δ is therefore a reference summary of the same response, not the sole input from which PEACE can be reconstructed. Here Δ=1.75 and P₁/₂=1.41838949. Each estimate is checked against its own known truth."
    )
    heading("3. Estimation and actual response-summary estimates")
    para(
        "The main response class contains all quadratic polynomials with coefficients estimated from observed data. Fit OLS, empirical series IV, and ridge IV with fixed penalties 10⁻⁶,10⁻⁴,10⁻²; the primary ridge penalty is 10⁻⁴. An expanded cubic class checks modeling sensitivity. Estimators receive no true coefficients, latent variables, true ρ or population conditional operator. This is a declared finite-class benchmark rather than unrestricted response learning."
    )
    para(
        "There are 500 repetitions for each of n=1,000,4,000,16,000 and six strengths 0.8,0.5,0.2,0.1,0.05,0. The additional precision check uses n=64,000,256,000 at strengths 0.2,0.1,0.05, with the same response and primary quadratic class. It was motivated after the first quadratic results; all six additional cells are retained. The original cubic study remains a more demanding sensitivity analysis. Seeds pair treatment/instrument/noise draws and nest sample sizes; datasets within a cell are independent."
    )
    combined = pd.concat([S, L], ignore_index=True)
    a = combined[
        (combined.scenario == "quadratic")
        & (combined.n == 64000)
        & (combined.rho == 0.2)
        & (combined.method == "IV degree 2")
        & (combined.d == 0.5)
        & combined.metric.isin(NAMES)
    ]
    table(
        pd.DataFrame(
            {
                "Quantity": a.metric.map(NAMES),
                "Truth": a.truth.round(6),
                "Median": a["median"].round(6),
                "5th–95th percentiles": [
                    f"{lo:.4f}–{hi:.4f}" for lo, hi in zip(a.q05, a.q95)
                ],
            }
        )
    )
    figure(
        "01_response_summaries",
        "Estimates of the two response summaries across strengths at n=4,000. Points are medians and bars are 5th–95th percentiles across datasets, not confidence intervals. PEACE uses a log vertical axis; the contrast axis is symmetric-log, linear between −1 and 1. No metric superiority is inferred. Zero strength † is unidentified; extreme estimates remain in the raw results.",
    )
    heading("4. When the contrast is accurate, how often is Mean PEACE accurate?")
    equation(
        r"A_\Delta=\{|\widehat\Delta-\Delta|\le0.1|\Delta|\},\qquad A_P=\{|\widehat P-P|\le0.1P\},\qquad\widehat{\Pr}(A_P\mid A_\Delta)=\frac{\#(A_P\cap A_\Delta)}{\#A_\Delta},",
        "Accurate contrast: |Δ̂−Δ| ≤10% of |Δ|\nAccurate Mean PEACE: |P̂−P| ≤10% of P\nConditional recovery frequency = number with both accurate / number with accurate contrast",
    )
    table(recovery_table(summary))
    para(
        "Every row fixes ρ=0.2,d=1/2 and uses 500 paired response fits. At n=64,000, 487 contrasts meet tolerance and 436 of these also have accurate Mean PEACE:89.5%. The remaining 51 datasets have an accurate contrast but Mean PEACE outside tolerance. At n=256,000, 499 of 500 qualify for both, giving 99.8% conditional recovery. Counts expose the denominator and prevent confusion between overall and conditional accuracy."
    )
    para(
        "These are descriptive Monte Carlo diagnostics with known causal truth. They are not a truth-free validation test available in observational data. Conditional accuracy is defined only when at least one contrast meets tolerance; a zero denominator is saved as missing, never as zero recovery. Wilson intervals quantify binomial Monte Carlo uncertainty conditional on the observed qualifying count."
    )
    figure(
        "02_conditional_recovery",
        "Left: accuracy of Mean PEACE overall and among datasets with an accurate contrast, at fixedρ=0.2. Conditional error bars are Wilson 95% intervals based on the qualifying contrast count. Right: the four paired accuracy states among all 500 datasets. Conditioning changes the evaluated subset; it is not a causal intervention.",
    )
    figure(
        "03_paired_errors",
        "Each point is one response fit at ρ=0.2. The dashed boundaries mark ±10% relative error for each target; the central square means both estimates are accurate. Symmetric-log axes retain all extreme observations and are linear within ±0.1. This displays joint estimation behavior rather than a ranking of metrics.",
    )
    heading("5. Why an accurate finite contrast need not certify the aggregation")
    equation(
        r"h(x)=c+ax+b(x^2-1),\qquad\Delta=a+b,\qquad P_d=\int|a+2bx|w_d(x)\,\mathrm dx,",
        "For a fitted quadratic h(x)=c+a x+b(x²−1):\nΔ=a+b; P_d=∫|a+2bx|w_d(x)dx; c cancels from both summaries",
    )
    equation(
        r"\widehat\Delta-\Delta=e_a+e_b,\qquad\widehat P-P\simeq0.50124e_a+1.22287e_b,",
        "For small coefficient errors at a=1,b=0.75,d=1/2:\nContrast error = e_a+e_b\nMean PEACE error ≈0.50124e_a+1.22287e_b",
    )
    para(
        "Errors in a and b can cancel in the finite contrast while changing local slopes and their magnitudes. For example e_a=−e_b makes the contrast exact, but the first-order PEACE error is about 0.72163e_b. This is the mathematical reason that the conditional recovery frequency can be below 100%. Both targets depend on curvature, and its precision depends on instrument information. A single contrast does not generally determine the full response or its local variation."
    )
    para(
        "The local sensitivity calculation was independently checked by adaptive quadrature and finite differences. At n=64,000/ρ=0.2, empirical coefficient SDs are about 0.0228 for a and 0.0761 for b. The first-order calculation predicts PEACE SD 0.0936, close to the observed 0.0930. This explains sampling behavior in this setting, not a universal variance ordering or weak-IV confidence theorem."
    )
    heading("6. Visible limitations and sensitivity checks")
    primary(summary, 0.05)
    table(recovery_table(summary, 0.05))
    para(
        "Lower correlation can support good recovery when sample size provides adequate effective information, as at ρ=0.2/n=256,000. Atρ=0.05 the curvature direction remains weak even at that n: the illustrative nρ⁴ scale is 1.6, compared with 409.6 at ρ=0.2. The marginal contrast accuracy is 22.4% and Mean PEACE accuracy 15.8% there. These are design-specific component scales, not universal diagnostic thresholds. Completeness establishes uniqueness, not finite-sample precision."
    )
    figure(
        "04_weak_information",
        "Mean PEACE recovery across sample size and instrument strength. Curves show medians; shaded bands are 5th–95th percentiles. Both axes are logarithmic. The same response and fixed penalty are used along each curve, including unsuccessful cells.",
    )
    para(
        "Fixed regularization can suppress real curvature and stay biased as n grows. The expanded response class adds another weak direction even though the true extra coefficient is zero. Both are displayed. Known marginal weights isolate response error; a paired fitted-normal-density calculation remains in the numerical outputs and can matter once response estimates become precise. At d=0 density fitting has no effect on the unit weights. No general nonparametric density guarantee is asserted."
    )
    figure(
        "05_estimation_sensitivity",
        "Fixed penalty and response-class sensitivity of estimated Mean PEACE at n=4,000/d=1/2. Curves show medians; full sampling distributions and error summaries remain available. Estimation choices are evaluated for the same target.",
    )
    para(
        "The original cubic response is h_c(x)=0.4x³+0.5x²−0.2x−0.5. At n=4,000/ρ=0.2 its Mean PEACE is within 10% in 13.2% of datasets, versus 29.2% in the quadratic setting. Both known truths, all fitted coefficients and the original PEACE values are retained. The new paired-recovery files also cover this stress analysis. This is a comparison of specified estimation problems, not a claim that one causal functional is generally preferable."
    )
    heading("7. Degree interpretation and scope")
    para(
        "For the quadratic response, true Mean PEACE decreases from 2.0083 at d=0 to 1.0636 at d=2 over the declared grid. The degree-derivative identity specializes to ∂P_d/∂d=−Cov_wd(|h′(X)|,X²): higher d emphasizes central high-density treatment values. The finite 0→1 contrast stays 1.75 because its endpoints are fixed. Degrees define different local-aggregation priorities, and were not selected by ground-truth estimation error. They do not create missing IV information."
    )
    figure(
        "06_degree_recovery",
        "Recovery of the Mean PEACE degree curve at n=4,000. Dashed curves are exact targets and solid curves are medians across 500 response fits. Each d defines its own aggregation target.",
    )
    para(
        "Suggested interpretation: the IV example studies how a learned causal response supports estimation of a finite intervention contrast and aggregated local causal movement. Mean PEACE recovery improves with sample size, while uncertainty about curvature can remain consequential even when the finite contrast is accurately estimated. The conditional recovery analysis quantifies that relationship and the full simulations retain its limits."
    )
    heading("8. Reproduction and verification")
    audit = json.loads((OUT / "paired_recovery_audit.json").read_text())
    para(
        f"Paired diagnostics cover {audit['paired_response_fits']:,} saved response fits in {audit['simulation_cells']} cells. All four accuracy-state counts were independently checked, and marginal rates agree with the original summaries. Zero qualifying-count cases are retained as undefined conditional rates. The existing numerical audit verifies targets, coefficients, quadrature, nested samples and unchanged original PEACE values; full simulation reruns produced identical outputs. The original cubic study is checked against the hashes recorded before the follow-up; manuscript hash checks are optional when manuscript sources are present."
    )
    para(
        "Run python report.py after the saved simulations exist. It rebuilds recovery.py diagnostics, checks local sensitivity, and generates this report, the compact summary and six figures. README.md documents the full reproduction commands and simulation restrictions. The sources and full reference results are distributed in the GitHub reproducibility package, and the quadratic IV results support the manuscript example."
    )
    for label, path in [
        ("Compact summary", "COMPACT_SUMMARY.md"),
        ("Mathematical argument", "Mathematical_argument.md"),
        ("Original cubic study and linear robust-inference check", "../REPORT.html"),
    ]:
        md.append(f"[{label}]({path})")
        web.append(f'<p><a href="{path}">{label}</a></p>')
    (ROOT / "REPORT.md").write_text("\n\n".join(md) + "\n")
    style = "body{max-width:1200px;margin:40px auto;padding:0 24px;color:#1c2933;font:17px/1.65 Georgia,serif}h1,h2{font-family:Arial,sans-serif;line-height:1.25}h1{font-size:32px}h2{font-size:24px;margin-top:42px}.equation{background:#f2f5f7;padding:18px 22px;border-left:4px solid #2463a6;font:18px/1.8 Georgia,serif;overflow:auto}.tablewrap{overflow:auto;margin:22px 0}table{border-collapse:collapse;font:14px/1.5 Arial,sans-serif;width:100%}th,td{padding:8px 10px;border-bottom:1px solid #d9e0e5;text-align:right;white-space:nowrap}th{background:#eef2f5}figure{margin:28px 0}img{width:100%;height:auto}figcaption{font:14px/1.5 Arial,sans-serif;color:#4d5c67;margin-top:8px}a{color:#1b5d9e}"
    (ROOT / "REPORT.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'
        + title
        + "</title><style>"
        + style
        + "</style></head><body>"
        + "\n".join(web)
        + "</body></html>"
    )


def main():
    raw, summary = recovery.analyze()
    sensitivity_analysis()
    figures(raw, summary)
    compact(summary)
    report(summary)
    print("Paired recovery, two-quantity reports and six figures saved")


if __name__ == "__main__":
    main()
