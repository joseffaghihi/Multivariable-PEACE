from study import (
    DESIGNS,
    DOM,
    KFold,
    M,
    N,
    RES,
    STATUS,
    TARGETS,
    generate,
    json,
    kde_values,
    np,
    pd,
    truth_conditional,
    truth_total,
    weighted,
)
from models import selected_additive


def run_selected(reps=40):
    rows = []
    conditional = []
    settings = []
    prediction = []
    checks = []
    for scenario, noise, sizes in [
        ("original", 0, [1000, 5000, 20000]),
        ("outcome_noise", 0.5, [1000, 5000, 20000]),
        ("interaction", 0.5, [5000, 20000]),
    ]:
        for n in sizes:
            for rep in range(reps):
                x, y = generate(n, seed=910000 + rep + int(noise * 100), noise=noise)
                if scenario == "interaction":
                    y = y + 20 * (x[N].to_numpy() - 0.2) * (x[M].to_numpy() - 0.515)
                for fold, (train, test) in enumerate(
                    KFold(2, shuffle=True, random_state=29001 + rep).split(x)
                ):
                    for design, cols in DESIGNS.items():
                        xt = x.iloc[train][cols].to_numpy()
                        xe = x.iloc[test][cols].to_numpy()
                        m = selected_additive(xt, y[train])
                        method = "selected_additive"
                        settings.append(
                            dict(
                                scenario=scenario,
                                n=n,
                                rep=rep,
                                fold=fold,
                                design=design,
                                method=method,
                                knots=m.knots,
                                df=m.df,
                                bic=m.bic,
                                nonlinear_components="; ".join(
                                    cols[j] for j in m.selected_components
                                ),
                            )
                        )
                        prediction.append(
                            dict(
                                scenario=scenario,
                                n=n,
                                rep=rep,
                                fold=fold,
                                design=design,
                                method=method,
                                heldout_mse=float(
                                    np.mean((m.predict(xe) - y[test]) ** 2)
                                ),
                            )
                        )
                        for name in TARGETS[design]:
                            j = cols.index(name)
                            g = m.derivative(xe, j)
                            conditional.append(
                                dict(
                                    scenario=scenario,
                                    n=n,
                                    rep=rep,
                                    fold=fold,
                                    variable=name,
                                    method=method,
                                    estimate=float(np.mean(np.abs(g))),
                                    signed=float(np.mean(g)),
                                    truth=truth_conditional(name, scenario),
                                )
                            )
                            ref = m.prepare_reference(xe, j)
                            inside = (xe[:, j] >= DOM[name][0]) & (
                                xe[:, j] <= DOM[name][1]
                            )
                            rows.append(
                                dict(
                                    scenario=scenario,
                                    n=n,
                                    rep=rep,
                                    fold=fold,
                                    variable=name,
                                    method=method,
                                    d=0.5,
                                    estimate=float(
                                        np.mean(
                                            np.abs(
                                                m.standardized_derivative(
                                                    xe[inside, j], ref
                                                )
                                            )
                                        )
                                    ),
                                    truth=truth_total(name, 0.5, scenario),
                                    support=STATUS[name],
                                )
                            )
                            t, w = m.integration_grid(*DOM[name], j, nodes=24)
                            tg = m.standardized_derivative(t, ref)
                            lf = np.log(
                                kde_values(name, x.iloc[train][name].to_numpy(), t)
                            )
                            for d in [0, 1, 2]:
                                rows.append(
                                    dict(
                                        scenario=scenario,
                                        n=n,
                                        rep=rep,
                                        fold=fold,
                                        variable=name,
                                        method=method,
                                        d=d,
                                        estimate=weighted(tg, lf, w, d)[0],
                                        truth=truth_total(name, d, scenario),
                                        support=STATUS[name],
                                    )
                                )
                        if rep == 0 and fold == 0:
                            probe = xe[:25]
                            j = 0
                            eps = 1e-5 * m.sd[j]
                            plus = probe.copy()
                            minus = probe.copy()
                            plus[:, j] += eps
                            minus[:, j] -= eps
                            checks.append(
                                float(
                                    np.max(
                                        np.abs(
                                            (m.predict(plus) - m.predict(minus))
                                            / (2 * eps)
                                            - m.derivative(probe, j)
                                        )
                                    )
                                )
                            )
                if (rep + 1) % 10 == 0:
                    print(
                        "completed selected additive", scenario, n, rep + 1, flush=True
                    )
            pd.DataFrame(rows).to_csv(
                RES / "selected_global_fold_results.csv", index=False
            )
            pd.DataFrame(conditional).to_csv(
                RES / "selected_conditional_fold_results.csv", index=False
            )
            pd.DataFrame(settings).to_csv(
                RES / "nonlinear_component_selection.csv", index=False
            )
            pd.DataFrame(prediction).to_csv(
                RES / "selected_prediction_fold_results.csv", index=False
            )
    out = {
        "derivative_max_finite_difference_error": max(checks),
        "passed": max(checks) < 1e-5,
    }
    (RES / "selected_validation.json").write_text(json.dumps(out, indent=2))
    assert out["passed"]


if __name__ == "__main__":
    run_selected()
