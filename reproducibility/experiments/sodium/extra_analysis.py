"""Joint-effect, curve-shape and support diagnostics for the sodium study."""

from experiment import (
    BASE_SEED,
    Grid,
    LinearRegression,
    OUT,
    ResponseFit,
    SCENARIOS,
    TARGETS,
    aggregation,
    density,
    fitted_logs,
    generate,
    logsumexp,
    ndtr,
    normalize_log,
    np,
    pd,
    savefig,
    summarize,
    threadpool_limits,
    true_derivatives,
    truth_logs,
)


def joint(grid, logs, gs, ga, ds):
    lj, ls, la = logs
    L = np.hypot(gs[:, None], ga[None, :])
    logarea = np.log(grid.ws)[:, None] + np.log(grid.wa)[None, :]
    return np.array(
        [
            np.sum(np.exp(2 * d * lj + logarea - logsumexp(2 * d * lj + logarea)) * L)
            for d in ds
        ]
    )


def main():
    rows = []
    ds = np.linspace(0, 3, 121)
    for n in [24, 48, 96, 192]:
        grid = Grid(n)
        gs, ga = true_derivatives(grid.s, grid.a)
        vals = joint(grid, truth_logs(grid), gs, ga, ds)
        for d, v in zip(ds, vals):
            rows.append(dict(nodes=n, d=d, value=v))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "joint_oracle.csv", index=False)
    assert (
        np.max(
            np.abs(
                df[df.nodes == 96].value.to_numpy()
                - df[df.nodes == 192].value.to_numpy()
            )
        )
        < 1e-5
    )
    grid = Grid(96)
    ds = [0.0, 0.5, 1.0, 2.0]
    truth = joint(grid, truth_logs(grid), *true_derivatives(grid.s, grid.a), ds)
    rows = []
    for ni, n in enumerate([500, 2000, 8000]):
        for rep in range(100):
            x, y, p = generate(n, BASE_SEED + ni * 100000 + rep)
            for method in ["structural", "spline"]:
                f = ResponseFit(x, y, method)
                gs, ga = f.derivatives(grid)
                for density, logs in [
                    ("estimated_gaussian", fitted_logs(grid, x)),
                    ("oracle", truth_logs(grid)),
                ]:
                    v = joint(grid, logs, gs, ga, ds)
                    for i, d in enumerate(ds):
                        rows.append(
                            dict(
                                n=n,
                                rep=rep,
                                method=method,
                                density=density,
                                d=d,
                                truth=truth[i],
                                estimate=v[i],
                            )
                        )
        print("joint", n, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "joint_raw.csv", index=False)
    summarize(df, ["n", "method", "density", "d"], OUT / "joint_summary.csv")
    # Independent closed-form age integral checks for Gaussian conditionals.
    g = Grid(128)
    lj, ls, la = truth_logs(g)
    vs = 1 + (5 / 18) ** 2
    c = 25 / 18
    mu = 65 + c / vs * (g.s - 65 / 18)
    v = 25 - c * c / vs
    checks = []
    for d, r in TARGETS:
        if d == 0:
            inner = np.full_like(g.s, (np.exp(1.5) - np.exp(-1.5)) / 3)
        else:
            sd = np.sqrt(v / (2 * d))
            ratio = (
                ndtr((80 - mu - 0.1 * sd**2) / sd) - ndtr((50 - mu - 0.1 * sd**2) / sd)
            ) / (ndtr((80 - mu) / sd) - ndtr((50 - mu) / sd))
            inner = np.exp(0.1 * (mu - 65) + 0.005 * sd**2) * ratio
        exact = inner @ normalize_log(r * ls, g.ws, 0)
        numeric = aggregation(g, (lj, ls, la), *true_derivatives(g.s, g.a), [(d, r)])[
            0, 1
        ]
        checks.append(dict(d=d, r=r, error=abs(exact - numeric)))
    pd.DataFrame(checks).to_csv(OUT / "age_closed_form_checks.csv", index=False)
    assert max(z["error"] for z in checks) < 1e-6
    # Prespecified observational-support diagnostics.
    rows = []
    for si, sc in enumerate(SCENARIOS):
        x, y, p = generate(200000, BASE_SEED + 2500000 + si, sc)
        rows.append(
            dict(
                scenario=sc.name,
                sodium_below_zero=float(np.mean(x[:, 0] < 0)),
                sodium_above_eight=float(np.mean(x[:, 0] > 8)),
                age_outside_domain=float(np.mean((x[:, 1] < 50) | (x[:, 1] > 80))),
                sodium_above_threshold=float(np.mean(x[:, 0] > 3.5)),
                residual_sodium_sd=sc.sigma
                if not sc.mixture
                else sc.sigma * np.sqrt(1.17),
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "support_diagnostics.csv", index=False)
    # Deliberate collider adjustment: same simple regression in both arms.
    # A misspecified linear response is avoided with the structural feature basis.
    rows = []
    for rep in range(100):
        x, y, p = generate(2000, BASE_SEED + 2700000 + rep)
        s, a = x.T
        features = np.column_stack(
            [s, np.maximum(s - 3.5, 0), np.exp(0.1 * (a - 65)), a - 65]
        )
        for adjust in [False, True]:
            f = np.column_stack([features, p]) if adjust else features
            model = LinearRegression().fit(f, y)
            b = model.coef_
            gs = b[0] + b[1] * (grid.s > 3.5)
            ga = 0.1 * b[2] * np.exp(0.1 * (grid.a - 65)) + b[3]
            val = aggregation(grid, fitted_logs(grid, x), gs, ga, [(0.5, 1)])[0]
            tr = aggregation(
                grid, truth_logs(grid), *true_derivatives(grid.s, grid.a), [(0.5, 1)]
            )[0]
            for j, e in enumerate(["sodium", "age_controlled"]):
                rows.append(
                    dict(
                        rep=rep,
                        adjustment="includes_proteinuria"
                        if adjust
                        else "correct_inputs",
                        exposure=e,
                        truth=tr[j],
                        estimate=val[j],
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "collider_raw.csv", index=False)
    summarize(df, ["adjustment", "exposure"], OUT / "collider_summary.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = pd.read_csv(OUT / "joint_oracle.csv")
    z = curves[curves.nodes == 192]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z.d, z.value, color="#2458a6")
    ax.set(
        xlabel="d",
        ylabel="Joint gradient magnitude (mixed raw units)",
        title="Joint sodium-age PEACE | no covariates, hence no r",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "09_joint_curve")
    plt.close(fig)
    z = pd.read_csv(OUT / "collider_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    exposures = ["sodium", "age_controlled"]
    for off, (adj, color) in zip(
        [-0.15, 0.15],
        [("correct_inputs", "#2458a6"), ("includes_proteinuria", "#c26520")],
    ):
        q = z[z.adjustment == adj].set_index("exposure").loc[exposures]
        ax.bar(
            np.arange(2) + off,
            q["mean"],
            width=0.3,
            color=color,
            label=adj.replace("_", " "),
        )
        ax.errorbar(
            np.arange(2) + off,
            q["mean"],
            yerr=1.96 * q.bias_mcse,
            fmt="none",
            color="black",
            capsize=3,
        )
    ax.scatter(
        np.arange(2),
        z[z.adjustment == "correct_inputs"].set_index("exposure").loc[exposures].truth,
        color="black",
        marker="x",
        label="True causal PEACE",
        zorder=5,
    )
    ax.set_xticks(range(2), ["Sodium", "Age controlled"])
    ax.set(
        ylabel="Mean PEACE (exposure-specific units)",
        title="Conditioning on proteinuria changes the target response",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "10_collider_adjustment")
    plt.close(fig)
    print("extra checks passed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
