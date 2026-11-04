"""Distinguish mean-response PEACE from mean individual-response magnitude."""

from experiment import (
    BASE_SEED,
    Grid,
    OUT,
    ResponseFit,
    aggregation,
    fitted_logs,
    generate,
    ndtr,
    norm,
    np,
    pd,
    quad,
    savefig,
    summarize,
    threadpool_limits,
    true_derivatives,
    truth_logs,
)


def folded_normal_mean(mu, sigma):
    if sigma == 0:
        return abs(mu)
    return sigma * np.sqrt(2 / np.pi) * np.exp(-(mu**2) / (2 * sigma**2)) + mu * (
        2 * ndtr(mu / sigma) - 1
    )


def main():
    g = Grid(128)
    logs = truth_logs(g)
    gs, ga = true_derivatives(g.s, g.a)
    rows = []
    for kappa in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for d, r in [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]:
            mean = aggregation(g, logs, gs, ga, [(d, r)])[0, 0]
            individual = aggregation(
                g, logs, folded_normal_mean(gs, kappa), ga, [(d, r)]
            )[0, 0]
            rows.append(
                dict(
                    kappa=kappa,
                    d=d,
                    r=r,
                    mean_response_peace=mean,
                    mean_individual_peace=individual,
                    gap=individual - mean,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "separability_sensitivity.csv", index=False)
    # Numerical integration independently verifies the folded-normal identity.
    z, w = quad(-10, 10, 1000)
    w = w * norm.pdf(z)
    for mu in [2.5, 5.0]:
        numeric = np.sum(w * abs(mu + 3 * z))
        exact = folded_normal_mean(mu, 3.0)
        assert abs(numeric - exact) < 0.0003
    rows = []
    grid = Grid(96)
    meantrue = aggregation(
        grid, truth_logs(grid), *true_derivatives(grid.s, grid.a), [(0.5, 1)]
    )[0, 0]
    individualtrue = float(
        df[(df.kappa == 3) & (df.d == 0.5)].mean_individual_peace.iloc[0]
    )
    for rep in range(100):
        seed = BASE_SEED + 2900000 + rep
        x, y, p = generate(2000, seed)
        u = np.random.default_rng(seed + 10000).normal(size=len(y))
        y = y + 3 * (x[:, 0] - 3.5) * u
        for method in ["structural", "spline"]:
            f = ResponseFit(x, y, method)
            est = aggregation(
                grid, fitted_logs(grid, x), *f.derivatives(grid), [(0.5, 1)]
            )[0, 0]
            for target, truth in [
                ("mean_response", meantrue),
                ("mean_individual_magnitude", individualtrue),
            ]:
                rows.append(
                    dict(
                        rep=rep, method=method, target=target, truth=truth, estimate=est
                    )
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "separability_raw.csv", index=False)
    summarize(raw, ["method", "target"], OUT / "separability_summary.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    z = df[df.d == 0.5]
    ax.plot(
        z.kappa,
        z.mean_response_peace,
        "o-",
        label="Magnitude of mean-response derivative",
        color="#2458a6",
    )
    ax.plot(
        z.kappa,
        z.mean_individual_peace,
        "o-",
        label="Mean individual-response magnitude",
        color="#c26520",
    )
    ax.set(
        xlabel="Unobserved response heterogeneity strength",
        ylabel="Sodium Mean PEACE (d = 0.5, r = 1)",
        title="Without separability, averaging and magnitude do not commute",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "14_separability_sensitivity")
    plt.close(fig)
    print("separability checks passed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
