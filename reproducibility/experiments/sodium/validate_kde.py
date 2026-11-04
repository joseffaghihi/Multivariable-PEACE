"""Bandwidth and quadrature check using the density-study seeds."""

from experiment import (
    BASE_SEED,
    Grid,
    OUT,
    SCENARIOS,
    aggregation,
    gaussian_kde,
    generate,
    np,
    pd,
    threadpool_limits,
    true_derivatives,
)


def main():
    rows = []
    for si, sc in enumerate([SCENARIOS[0], SCENARIOS[-1]]):
        for rep in range(2):
            x, y, p = generate(2000, BASE_SEED + 1800000 + si * 10000 + rep, sc)
            for fac in [0.5, 1.0, 2.0]:
                k = gaussian_kde(x.T, bw_method=lambda kde: kde.scotts_factor() * fac)
                ks = gaussian_kde(
                    x[:, 0], bw_method=lambda kde: kde.scotts_factor() * fac
                )
                ka = gaussian_kde(
                    x[:, 1], bw_method=lambda kde: kde.scotts_factor() * fac
                )
                vals = []
                for n in [24, 48, 96]:
                    g = Grid(n)
                    logs = (
                        k.logpdf(np.vstack([g.ss.ravel(), g.aa.ravel()])).reshape(
                            g.ss.shape
                        ),
                        ks.logpdf(g.s),
                        ka.logpdf(g.a),
                    )
                    vals.append(
                        aggregation(
                            g,
                            logs,
                            *true_derivatives(g.s, g.a, sc),
                            [(0.25, 1), (0.5, 1), (1, 1)],
                        )
                    )
                e48 = float(np.max(abs(vals[1] - vals[-1])))
                rows.append(
                    dict(
                        scenario=sc.name,
                        rep=rep,
                        bandwidth_factor=fac,
                        error_24_vs_96=float(np.max(abs(vals[0] - vals[-1]))),
                        error_48_vs_96=e48,
                    )
                )
                assert e48 < 0.0002
    pd.DataFrame(rows).to_csv(OUT / "kde_quadrature_checks.csv", index=False)
    print("KDE quadrature checks passed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
