"""Additional figures separating estimands, sampling error and curve meaning."""

from experiment import (
    Grid,
    OUT,
    SCENARIOS,
    Scenario,
    aggregation,
    comparator_truth,
    json,
    ndtr,
    norm,
    normalize_log,
    np,
    pd,
    savefig,
    threadpool_limits,
    true_derivatives,
    truth_logs,
)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.integrate import quad as integrate

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = ["#2458a6", "#c26520", "#2a8b64"]
    g = Grid(128)
    ds = np.linspace(0, 3, 121)
    logs = truth_logs(g)
    total = np.exp(0.1 * (g.a - 65)) + (2.5 + 2.5 * ndtr(g.a / 18 - 3.5)) / 18
    vals = [float(total @ normalize_log(2 * d * logs[2], g.wa, 0)) for d in ds]
    pd.DataFrame(dict(d=ds, age_total=vals)).to_csv(
        OUT / "total_age_peace.csv", index=False
    )
    density_curves = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for sc, color in [(SCENARIOS[0], colors[0]), (SCENARIOS[-1], colors[1])]:
        v = aggregation(
            g,
            truth_logs(g, sc),
            *true_derivatives(g.s, g.a, sc),
            [(d, 1.0) for d in ds],
        )
        ax.plot(ds, v[:, 0], color=color, label=sc.name.replace("_", " "))
        for d, value in zip(ds, v[:, 0]):
            density_curves.append(dict(scenario=sc.name, d=d, r=1.0, sodium=value))
    ax.set(
        xlabel="d (r = 1)",
        ylabel="Sodium Mean PEACE",
        title="Same causal response, different treatment distributions",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "15_distribution_curve")
    plt.close(fig)
    pd.DataFrame(density_curves).to_csv(
        OUT / "distribution_oracle_curves.csv", index=False
    )

    # Exact finite-contrast and observational derivative targets.
    def eh(mu, sd=1.0):
        z = (mu - 3.5) / sd
        return (mu - 3.5) * ndtr(z) + sd * norm.pdf(z)

    def total_age(a):
        return 90 + 2.5 * a / 18 + 2.5 * eh(a / 18) + 10 * np.exp(0.1 * (a - 65))

    acontrast = 10 * (np.exp(0.5) - np.exp(-0.5))
    atotal = total_age(70) - total_age(60)
    slopes = comparator_truth(Scenario())
    table = [
        dict(quantity="Sodium contrast: do(5) minus do(2)", value=11.25, units="mmHg"),
        dict(
            quantity="Sodium contrast divided by 3-unit change",
            value=3.75,
            units="mmHg per sodium unit",
        ),
        dict(
            quantity="Sodium QTE: do(5) versus do(2), every quantile",
            value=11.25,
            units="mmHg",
        ),
        dict(
            quantity="Sodium full-population average derivative",
            value=slopes["sodium"],
            units="mmHg per sodium unit",
        ),
        dict(
            quantity="Age controlled contrast: do(70,s) minus do(60,s)",
            value=acontrast,
            units="mmHg",
        ),
        dict(
            quantity="Age controlled contrast divided by 10 years",
            value=acontrast / 10,
            units="mmHg per year",
        ),
        dict(
            quantity="Age total contrast: do(70) minus do(60)",
            value=atotal,
            units="mmHg",
        ),
        dict(
            quantity="Age full-population average controlled derivative",
            value=slopes["age_controlled"],
            units="mmHg per year",
        ),
        dict(
            quantity="Age full-population average total derivative",
            value=slopes["age_controlled"] + slopes["sodium"] / 18,
            units="mmHg per year",
        ),
    ]
    pd.DataFrame(table).to_csv(OUT / "estimand_comparison.csv", index=False)
    # Empirical sampling distributions; ribbons are not confidence intervals.
    raw = pd.read_csv(OUT / "simulation_raw.csv")
    curves = pd.read_csv(OUT / "oracle_curves.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, e in enumerate(["sodium", "age_controlled"]):
        truth = curves[curves.r == 1]
        ax[j].plot(truth.d, truth[e], color="black", lw=2, label="Exact target")
        for method, color in zip(["structural", "spline"], colors):
            z = (
                raw[
                    (raw.scenario == "baseline")
                    & (raw.n == 2000)
                    & (raw.density == "estimated_gaussian")
                    & (raw.r == 1)
                    & (raw.exposure == e)
                    & (raw.method == method)
                ]
                .groupby("d")
                .estimate
            )
            mean = z.mean()
            lo = z.quantile(0.05)
            hi = z.quantile(0.95)
            ax[j].plot(mean.index, mean, marker="o", color=color, label=method)
            ax[j].fill_between(mean.index, lo, hi, color=color, alpha=0.13)
        ax[j].set(
            xlabel="d (r = 1)",
            ylabel="Mean PEACE (exposure-specific units)",
            title=e.replace("_", " ").title(),
            xlim=(0, 2),
        )
        ax[j].legend(fontsize=8)
    fig.suptitle(
        "Estimated curves at n = 2,000\nRibbons: middle 90% of estimates across repeated datasets",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, "11_estimated_curves")
    plt.close(fig)
    # A flat curve is a sufficient consequence of constant response magnitude.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    z = curves[curves.r == 1]
    ax[0].plot(z.d, z.sodium, color=colors[0], label="Threshold response")
    ax[0].axhline(3.75, color=colors[1], ls="--", label="Constant slope 3.75")
    ax[1].plot(z.d, z.age_controlled, color=colors[0], label="Exponential response")
    ax[1].axhline(1.0, color=colors[1], ls="--", label="Constant slope 1")
    for a, title in zip(ax, ["Sodium", "Age controlled"]):
        a.set(xlabel="d (r = 1)", ylabel="Mean PEACE", title=title)
        a.legend(fontsize=8)
    fig.suptitle(
        "Changing weights cannot change an everywhere constant local magnitude"
    )
    fig.tight_layout()
    savefig(fig, "12_flat_reference")
    plt.close(fig)
    # Explicit auxiliary example: positive and negative responses cancel in a
    # finite contrast, but do not cancel in magnitude-based PEACE.
    gc = Grid(128, sdomain=(2.0, 5.0))
    s = np.linspace(2, 5, 301)
    h = 2.5 * (s - 3.5) ** 2
    L = 5 * (gc.s - 3.5)
    zero_uniform = np.sum(gc.ws * L) / 3
    absolute_uniform = np.sum(gc.ws * abs(L)) / 3
    assert abs(zero_uniform) < 1e-10 and abs(absolute_uniform - 3.75) < 1e-10
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(s, h, color=colors[0])
    ax[0].scatter([2, 5], [h[0], h[-1]], color=colors[1])
    ax[0].set(
        xlabel="Sodium",
        ylabel="Sodium component of response",
        title="Auxiliary sign-reversing response",
    )
    ax[1].plot(s, 5 * (s - 3.5), color=colors[0], label="Signed local response")
    ax[1].plot(s, abs(5 * (s - 3.5)), color=colors[1], label="Local magnitude")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set(
        xlabel="Sodium",
        ylabel="Response per sodium unit",
        title="Zero contrast can conceal local effects",
    )
    ax[1].legend(fontsize=8)
    fig.suptitle(
        "Auxiliary illustration only: contrast do(5) - do(2) = 0; Mean PEACE at d = 0 is 3.75",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, "13_cancellation_illustration")
    plt.close(fig)
    (OUT / "cancellation_check.json").write_text(
        json.dumps(
            dict(
                domain=[2, 5],
                contrast=0.0,
                average_signed_derivative=float(zero_uniform),
                mean_peace_d0=float(absolute_uniform),
                note="Auxiliary oracle illustration; not the original sodium response and not an estimator benchmark.",
            ),
            indent=2,
        )
    )
    print("interpretation figures and exact comparisons passed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
