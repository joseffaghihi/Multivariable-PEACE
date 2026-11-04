"""Independent numerical and implementation checks; exits on failure."""

from experiment import (
    BASE_SEED,
    Grid,
    OUT,
    ResponseFit,
    SCENARIOS,
    aggregation,
    comparator_truth,
    fitted_logs,
    generate,
    np,
    pd,
    response,
    threadpool_limits,
    true_derivatives,
    truth_logs,
)


def main():
    checks = []
    for sc in SCENARIOS:
        results = []
        for n in [24, 48, 96, 192]:
            g = Grid(n)
            results.append(
                aggregation(g, truth_logs(g, sc), *true_derivatives(g.s, g.a, sc))
            )
        for n, v in zip([24, 48, 96], results[:-1]):
            checks.append(
                dict(
                    scenario=sc.name,
                    nodes=n,
                    max_abs_error=float(np.max(abs(v - results[-1]))),
                )
            )
        assert np.max(abs(results[1] - results[-1])) < 1e-5
    pd.DataFrame(checks).to_csv(OUT / "scenario_quadrature_checks.csv", index=False)
    # Estimated spline integration at coarse versus fine grids.
    checks = []
    for sc in [SCENARIOS[0], SCENARIOS[1], SCENARIOS[-1]]:
        for seed in [BASE_SEED + 31, BASE_SEED + 68, BASE_SEED + 81]:
            x, y, p = generate(2000, seed, sc)
            f = ResponseFit(x, y)
            vals = []
            for n in [96, 192, 384]:
                g = Grid(n)
                vals.append(aggregation(g, fitted_logs(g, x), *f.derivatives(g)))
            err = np.max(abs(vals[0] - vals[-1]))
            checks.append(dict(scenario=sc.name, seed=seed, max_abs_error=float(err)))
            assert err < 0.002
    pd.DataFrame(checks).to_csv(OUT / "estimated_quadrature_checks.csv", index=False)
    # Verify stable normalized weights even for large d and narrow overlap.
    g = Grid()
    logs = truth_logs(g, SCENARIOS[1])
    value = aggregation(g, logs, *true_derivatives(g.s, g.a), [(20.0, 20.0)])
    assert np.all(np.isfinite(value))
    assert 2.5 <= value[0, 0] <= 5
    # A counterfactual outcome shift has no effect on derivative-based PEACE.
    x, y, p = generate(2000, BASE_SEED)
    a = ResponseFit(x, y)
    b = ResponseFit(x, y + 1000)
    assert (
        max(np.max(abs(u - v)) for u, v in zip(a.derivatives(g), b.derivatives(g)))
        < 1e-8
    )
    assert np.array_equal(generate(2000, BASE_SEED)[0], x)
    # Verify Gaussian conditional moment targets without fitting an estimator.
    rows = []
    for sc in [SCENARIOS[0], SCENARIOS[1]]:
        x, y, p = generate(500000, BASE_SEED + 3123, sc)
        es = x[:, 0] - x[:, 1] / 18
        sodium_projection = np.cov(
            es,
            response(x[:, 0], x[:, 1], sc) - 10 * np.exp(0.1 * (x[:, 1] - 65)),
            ddof=0,
        )[0, 1] / np.var(es)
        truth = comparator_truth(sc)["sodium"]
        rows.append(
            dict(
                scenario=sc.name,
                empirical_sodium_projection=sodium_projection,
                truth=truth,
                error=sodium_projection - truth,
            )
        )
        # Monte Carlo check only: 500,000 observations, not an exact tolerance.
        assert abs(sodium_projection - truth) < 0.04
    pd.DataFrame(rows).to_csv(OUT / "slope_target_checks.csv", index=False)
    print("independent validation passed", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
