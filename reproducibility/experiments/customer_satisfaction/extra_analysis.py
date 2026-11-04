"""Joint response estimation, bootstrap, model/domain sensitivity and validation."""

from experiment import (
    A,
    B,
    C,
    D,
    DOM,
    E,
    I,
    L,
    LinearRegression,
    M,
    N,
    NAMES,
    PAIRS,
    RES,
    ROOT,
    SplineTransformer,
    coordinate_derivative,
    density,
    derivative,
    expected_feature_derivative,
    fit_response,
    fitted_derivative,
    generate,
    json,
    kde_values,
    logit,
    norm,
    np,
    pair_grid,
    pairs,
    pd,
    plt,
    quad,
    savefig,
    spline_model,
    spline_quadrature,
    weighted,
)
from scipy.integrate import quad as adaptive_quad


def joint_estimation():
    rows = []
    pairs = [(N, M), (C, N), (E, L)]
    for n in [1000, 5000]:
        for rep in range(60):
            x, y = generate(n, seed=121000 + rep, noise=0.5)
            for pair in pairs:
                cols = list(pair) + ([C] if pair == (N, M) else [])
                model = spline_model()
                model.fit(x[cols].to_numpy(), y)
                grids = [
                    spline_quadrature(model, cols, x, coordinate=j) for j in [0, 1]
                ]
                U, V, W, lf, g1, g2 = pair_grid(pair, grids=grids)
                # Additivity makes the standardized derivative independent of
                # nuisance covariates; evaluate one coordinate at a time.
                deriv = []
                for j, grid in enumerate([U[:, 0], V[0, :]]):
                    deriv.append(coordinate_derivative(model, grid, j))
                gh = np.hypot(deriv[0][:, None], deriv[1][None, :])
                if pair == (N, M):
                    reg = LinearRegression().fit(x[[N]], logit(x[M]))
                    sd = np.std(logit(x[M]) - reg.predict(x[[N]]), ddof=2)
                    lhat = (
                        np.log(kde_values(N, x[N].to_numpy(), U[:, 0]))[:, None]
                        + norm.logpdf(logit(V), reg.intercept_ + reg.coef_[0] * U, sd)
                        - np.log(V * (1 - V))
                    )
                elif pair == (C, N):
                    reg = LinearRegression().fit(x[[C]], x[N])
                    sd = np.std(x[N] - reg.predict(x[[C]]), ddof=2)
                    lhat = np.log(kde_values(C, x[C].to_numpy(), U[:, 0]))[
                        :, None
                    ] + norm.logpdf(V, reg.intercept_ + reg.coef_[0] * U, sd)
                else:
                    lhat = (
                        np.log(kde_values(E, x[E].to_numpy(), U[:, 0]))[:, None]
                        + np.log(kde_values(L, x[L].to_numpy(), V[0, :]))[None, :]
                    )
                for d in [0, 0.5, 1, 2]:
                    truth = weighted(
                        np.hypot(g1, g2).ravel(), lf.ravel(), W.ravel(), d
                    )[0]
                    for method, weight in [
                        ("Oracle density", lf),
                        ("Fitted joint density", lhat),
                    ]:
                        est = weighted(gh.ravel(), weight.ravel(), W.ravel(), d)[0]
                        rows.append(
                            dict(
                                n=n,
                                rep=rep,
                                pair=" + ".join(pair),
                                d=d,
                                method=method,
                                estimate=est,
                                truth=truth,
                            )
                        )
        print("completed joint estimation", n, flush=True)
    df = pd.DataFrame(rows)
    df["error"] = df.estimate - df.truth
    df.to_csv(RES / "joint_replicates.csv", index=False)
    s = (
        df.groupby(["n", "pair", "d", "method"])
        .agg(
            truth=("truth", "first"),
            bias=("error", "mean"),
            rmse=("error", lambda a: np.sqrt(np.mean(a * a))),
            estimate_sd=("estimate", "std"),
        )
        .reset_index()
    )
    s.to_csv(RES / "joint_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pair in zip(axes, pairs):
        for n in [1000, 5000]:
            z = s[
                (s.n == n)
                & (s.pair == " + ".join(pair))
                & (s.method == "Fitted joint density")
            ]
            ax.plot(z.d, z.rmse, "o-", label=f"n={n}")
        ax.set(title=" + ".join(pair), xlabel="d", ylabel="RMSE of joint Mean PEACE")
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    savefig("11_joint_estimation")


def bootstrap():
    # Percentile intervals are empirically audited, not presumed valid at nulls.
    rows = []
    rng = np.random.default_rng(22511)
    names = [N, L, A]
    for rep in range(40):
        x, y = generate(3000, seed=211000 + rep, noise=0.5)
        estimates = {name: [] for name in names}
        for draw in range(120):
            idx = rng.integers(0, len(x), len(x))
            xb = x.iloc[idx]
            yb = y[idx]
            for name in names:
                model, cols = fit_response(xb, yb, name)
                t, w = spline_quadrature(model, cols, xb)
                lf = np.log(density(name, t))
                g = fitted_derivative(model, cols, xb, t)
                # Hold density at truth to audit response-only intervals.
                estimates[name].append(weighted(g, lf, w, 0.5)[0])
        for name in names:
            m, cols = fit_response(x, y, name)
            t, w = spline_quadrature(m, cols, x)
            lf = np.log(density(name, t))
            truth = weighted(derivative(name, t), lf, w, 0.5)[0]
            lo, hi = np.quantile(estimates[name], [0.025, 0.975])
            point = weighted(fitted_derivative(m, cols, x, t), lf, w, 0.5)[0]
            rows.append(
                dict(
                    rep=rep,
                    n=3000,
                    bootstrap_draws=120,
                    variable=name,
                    truth=truth,
                    estimate=point,
                    lower=lo,
                    upper=hi,
                    covered=bool(lo <= truth <= hi),
                )
            )
        if (rep + 1) % 10 == 0:
            print("completed bootstrap datasets", rep + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "bootstrap_intervals.csv", index=False)
    s = (
        df.groupby("variable")
        .agg(
            coverage=("covered", "mean"),
            mean_lower=("lower", "mean"),
            mean_upper=("upper", "mean"),
        )
        .reset_index()
    )
    s.to_csv(RES / "bootstrap_summary.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(s.variable, s.coverage)
    plt.axhline(0.95, color="black", ls="--", label="Nominal 95%")
    plt.ylim(0, 1.05)
    plt.ylabel("Empirical coverage (40 independent datasets)")
    plt.legend()
    savefig("12_bootstrap_audit")


def robustness():
    rows = []
    for rep in range(40):
        x, y = generate(5000, seed=231000 + rep, noise=0.5)
        for name in [N, L, D, A]:
            for knots in [3, 5, 7]:
                model, cols = fit_response(x, y, name, knots)
                t, w = spline_quadrature(model, cols, x)
                lf = np.log(density(name, t))
                g = fitted_derivative(model, cols, x, t)
                for d in [0.5, 1]:
                    rows.append(
                        dict(
                            rep=rep,
                            variable=name,
                            knots=knots,
                            d=d,
                            estimate=weighted(g, lf, w, d)[0],
                            truth=weighted(derivative(name, t), lf, w, d)[0],
                        )
                    )
    df = pd.DataFrame(rows)
    df["error"] = df.estimate - df.truth
    df.to_csv(RES / "spline_sensitivity_replicates.csv", index=False)
    df.groupby(["variable", "knots", "d"]).agg(
        bias=("error", "mean"), rmse=("error", lambda a: np.sqrt(np.mean(a * a)))
    ).reset_index().to_csv(RES / "spline_sensitivity_summary.csv", index=False)
    # Exact domain dependence: d=0 over a declared interval, not sample extremes.
    rows = []
    for lo, hi in [(0, 10), (0.5, 9.5), (1, 9), (2, 8)]:
        rows.append(
            dict(lower=lo, upper=hi, upgrade_peace_d0=8 / ((lo + 1) * (hi + 1)))
        )
    pd.DataFrame(rows).to_csv(RES / "domain_sensitivity.csv", index=False)
    # Unit conversion: a change of coordinates rescales each component of a
    # joint gradient differently, so raw ranks are not scale-invariant.
    ref, _ = generate(200000, seed=231999)
    scales = ref.std().to_dict()
    rows = []
    for pair in PAIRS:
        U, V, W, lf, g1, g2 = pair_grid(pair, 120)
        for d in [0, 0.5, 1]:
            rows.append(
                dict(
                    pair=" + ".join(pair),
                    d=d,
                    native=weighted(np.hypot(g1, g2).ravel(), lf.ravel(), W.ravel(), d)[
                        0
                    ],
                    per_reference_SD=weighted(
                        np.hypot(scales[pair[0]] * g1, scales[pair[1]] * g2).ravel(),
                        lf.ravel(),
                        W.ravel(),
                        d,
                    )[0],
                )
            )
    pd.DataFrame(rows).to_csv(RES / "coordinate_scale_sensitivity.csv", index=False)
    print("completed robustness checks", flush=True)


def null_intervals():
    rows = []
    for rep in range(400):
        x, y = generate(3000, seed=251000 + rep, noise=0.5)
        parents = SplineTransformer(
            n_knots=5, degree=3, include_bias=False
        ).fit_transform(x[[M, L]])
        z = np.column_stack([np.ones(len(x)), x[A], parents])
        inv = np.linalg.pinv(z.T @ z)
        coef = np.linalg.lstsq(z, y, rcond=None)[0]
        resid = y - z @ coef
        leverage = np.sum((z @ inv) * z, axis=1)
        meat = z.T @ (((resid / (1 - leverage)) ** 2)[:, None] * z)
        se = np.sqrt((inv @ meat @ inv)[1, 1])
        signedlo = coef[1] - 1.96 * se
        signedhi = coef[1] + 1.96 * se
        lower = 0.0 if signedlo <= 0 <= signedhi else min(abs(signedlo), abs(signedhi))
        upper = max(abs(signedlo), abs(signedhi))
        rows.append(
            dict(
                rep=rep,
                variable=A,
                signed_slope=coef[1],
                se_HC3=se,
                signed_lower=signedlo,
                signed_upper=signedhi,
                magnitude_lower=lower,
                magnitude_upper=upper,
                covered=bool(lower <= 0 <= upper),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "null_linear_interval_replicates.csv", index=False)
    (RES / "null_linear_interval_summary.json").write_text(
        json.dumps(
            {
                "reps": 400,
                "n": 3000,
                "coverage": float(df.covered.mean()),
                "coverage_MCSE": float(
                    np.sqrt(df.covered.mean() * (1 - df.covered.mean()) / 400)
                ),
                "model": "Ad-spend-linear response plus additive parent splines; HC3 signed-slope interval mapped through absolute value",
                "mean_signed_slope": float(df.signed_slope.mean()),
                "mean_absolute_slope": float(np.abs(df.signed_slope).mean()),
            },
            indent=2,
        )
    )
    print("completed null interval reference", flush=True)


def validate():
    checks = {}
    # Check the archived simulator cell without running the notebook analyses.
    source = ROOT.parent.parent / "Customer_Satisfaction_Example.ipynb"
    notebook = json.loads(source.read_text())
    cell = "".join(notebook["cells"][1]["source"])
    cell = cell.replace("import xgboost as xgb", "")
    scope = {}
    exec(compile(cell, "original simulator cell", "exec"), scope)
    original = scope["generator"](1000, seed=184)
    explicit = (
        5 * original[N]
        + 3 * original[M]
        + 2 * original[E]
        + original[D]
        - 0.2 * original[B]
        + 0.2 * original[C]
        + 0.3 * original[I]
        + 8 / (original[L] + 1)
    )
    checks["original_outcome_algebra_max_error"] = float(
        np.max(np.abs(explicit - original["Customer Satisfaction"]))
    )
    o1 = scope["generator"](1000, fixed={M: 0.4}, seed=184)
    o2 = scope["generator"](1000, fixed={M: 0.6}, seed=184)
    checks["usage_intervention_changes_bugs"] = bool(np.any(o1[B] != o2[B]))
    # Common-random-number simulator finite differences versus analytic means.
    errors = {}
    for name in NAMES:
        t = sum(DOM[name]) / 2
        eps = 1e-4 * (DOM[name][1] - DOM[name][0])
        _, yp = generate(150000, 301001, fixed={name: t + eps})
        _, ym = generate(150000, 301001, fixed={name: t - eps})
        errors[name] = abs(
            float(np.mean((yp - ym) / (2 * eps)))
            - float(derivative(name, np.array([t]))[0])
        )
    checks["simulator_derivative_errors"] = errors
    coeff = np.array([5, 3, 2, 1, -0.2, 0.2, 0.3, 8, 0, 0])
    feature_errors = {}
    for name in NAMES:
        t = np.linspace(*DOM[name], 25)
        feature_errors[name] = float(
            np.max(
                np.abs(
                    expected_feature_derivative(name, t) @ coeff - derivative(name, t)
                )
            )
        )
    checks["propagated_feature_derivative_errors"] = feature_errors
    errors = {}
    for pair in PAIRS:
        a = pair_grid(pair, 100)
        b = pair_grid(pair, 180)
        errors[" + ".join(pair)] = max(
            abs(
                weighted(np.hypot(a[4], a[5]).ravel(), a[3].ravel(), a[2].ravel(), d)[0]
                - weighted(np.hypot(b[4], b[5]).ravel(), b[3].ravel(), b[2].ravel(), d)[
                    0
                ]
            )
            for d in [0, 0.5, 1, 3]
        )
    checks["pair_quadrature_refinement_errors"] = errors
    mass = {}
    for name, lo, hi in [(D, 0.01, 0.26), (M, 0.1, 0.9), (A, 0.1, 1.4), (N, -1, 1.4)]:
        t, w = quad(lo, hi, 700)
        mass[name] = float(w @ density(name, t))
    checks["marginal_density_mass_checks"] = mass
    signs = pd.read_csv(RES / "oracle_single.csv")
    checks["signed_decomposition_max_error"] = float(
        np.max(np.abs(signs.peace - signs.positive - signs.negative))
    )
    original_metrics = json.loads((RES / "validation_oracle.json").read_text())
    checks["max_covariance_identity_error"] = max(
        v["covariance_identity_error"]
        for v in original_metrics.values()
        if isinstance(v, dict)
    )
    checks["uniform_upgrade_error"] = original_metrics["uniform_upgrade_error"]
    integration_errors = []
    fitted_derivative_errors = []
    for seed, n in [(61001, 1000), (61002, 5000), (99001, 5000)]:
        x, y = generate(n, seed=seed, noise=0.5)
        for name in NAMES[:-1]:
            model, cols = fit_response(x, y, name)
            values = {}
            probe = np.linspace(*DOM[name], 37)
            points = np.tile(x[cols].mean().to_numpy(), (len(probe), 1))
            points[:, 0] = probe
            eps = 1e-6 * (DOM[name][1] - DOM[name][0])
            plus = points.copy()
            minus = points.copy()
            plus[:, 0] += eps
            minus[:, 0] -= eps
            fd = (model.predict(plus) - model.predict(minus)) / (2 * eps)
            fitted_derivative_errors.append(
                float(np.max(np.abs(fd - fitted_derivative(model, cols, x, probe))))
            )
            for nodes in [24, 48]:
                t, w = spline_quadrature(model, cols, x, nodes=nodes)
                g = fitted_derivative(model, cols, x, t)
                lf = np.log(kde_values(name, x[name].to_numpy(), t))
                values[nodes] = [weighted(g, lf, w, d)[0] for d in [0, 0.5, 1, 2]]
            integration_errors.append(
                dict(
                    seed=seed,
                    n=n,
                    variable=name,
                    max_absolute_difference=float(
                        np.max(np.abs(np.array(values[24]) - values[48]))
                    ),
                )
            )
    pd.DataFrame(integration_errors).to_csv(
        RES / "estimated_quadrature_checks.csv", index=False
    )
    checks["estimated_quadrature_max_difference"] = max(
        v["max_absolute_difference"] for v in integration_errors
    )
    checks["fitted_derivative_max_finite_difference_error"] = max(
        fitted_derivative_errors
    )
    joint_errors = []
    x, y = generate(1000, seed=121001, noise=0.5)
    for pair in [(N, M), (C, N), (E, L)]:
        cols = list(pair) + ([C] if pair == (N, M) else [])
        model = spline_model()
        model.fit(x[cols].to_numpy(), y)
        vals = []
        for nodes in [24, 48]:
            grids = [
                spline_quadrature(model, cols, x, coordinate=j, nodes=nodes)
                for j in [0, 1]
            ]
            U, V, W, lf, g1, g2 = pair_grid(pair, grids=grids)
            gh = np.hypot(
                coordinate_derivative(model, U[:, 0], 0)[:, None],
                coordinate_derivative(model, V[0, :], 1)[None, :],
            )
            vals.append(
                [
                    weighted(gh.ravel(), lf.ravel(), W.ravel(), d)[0]
                    for d in [0, 0.5, 1, 2]
                ]
            )
        joint_errors.append(float(np.max(np.abs(np.array(vals[0]) - vals[1]))))
    checks["estimated_joint_quadrature_max_difference"] = max(joint_errors)
    checks["passed"] = bool(
        checks["original_outcome_algebra_max_error"] < 1e-12
        and not checks["usage_intervention_changes_bugs"]
        and max(feature_errors.values()) < 1e-10
        and max(errors.values()) < 2e-5
        and max(abs(a - 1) for a in mass.values()) < 0.002
        and max(checks["simulator_derivative_errors"].values()) < 0.001
        and checks["max_covariance_identity_error"] < 1e-7
        and checks["estimated_quadrature_max_difference"] < 1e-4
        and checks["fitted_derivative_max_finite_difference_error"] < 1e-5
        and max(joint_errors) < 1e-4
    )
    (RES / "validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if not checks["passed"]:
        raise RuntimeError("Validation did not pass")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "part", choices=["joint", "bootstrap", "robustness", "null", "validate", "all"]
    )
    args = p.parse_args()
    for name, fn in [
        ("joint", joint_estimation),
        ("bootstrap", bootstrap),
        ("robustness", robustness),
        ("null", null_intervals),
        ("validate", validate),
    ]:
        if args.part in [name, "all"]:
            fn()
