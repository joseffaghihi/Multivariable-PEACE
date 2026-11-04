from study import (
    A,
    D,
    DESIGNS,
    DOM,
    KFold,
    RES,
    TARGETS,
    fit_models,
    generate,
    json,
    kde_values,
    np,
    pd,
    weighted,
)


def main():
    rows = []
    for n in [1000, 5000, 20000]:
        x, y = generate(n, seed=910051, noise=0.5)
        train, test = next(KFold(2, shuffle=True, random_state=29002).split(x))
        for design, cols in DESIGNS.items():
            xt = x.iloc[train][cols].to_numpy()
            xe = x.iloc[test][cols].to_numpy()
            models = fit_models(xt, y[train])
            for name in TARGETS[design]:
                j = cols.index(name)
                for kind, m in models.items():
                    if name in [D, A] and kind == "interaction":
                        continue
                    ref = m.prepare_reference(xe, j)
                    values = {}
                    for nodes in [24, 96]:
                        t, w = m.integration_grid(*DOM[name], j, nodes=nodes)
                        g = m.standardized_derivative(t, ref)
                        lf = np.log(kde_values(name, x.iloc[train][name].to_numpy(), t))
                        values[nodes] = np.array(
                            [weighted(g, lf, w, d)[0] for d in [0, 1, 2]]
                        )
                    error = np.abs(values[24] - values[96])
                    rel = error / (1 + np.abs(values[96]))
                    rows.append(
                        dict(
                            n=n,
                            design=design,
                            variable=name,
                            method=kind,
                            max_absolute_difference=float(error.max()),
                            max_scaled_difference=float(rel.max()),
                        )
                    )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "all_variable_quadrature_audit.csv", index=False)
    check = {
        "max_absolute_difference": float(df.max_absolute_difference.max()),
        "max_scaled_difference": float(df.max_scaled_difference.max()),
        "passed": bool(
            df.max_scaled_difference.max() < 0.001
            and df.max_absolute_difference.max() < 0.01
        ),
    }
    (RES / "integration_audit.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))
    if not check["passed"]:
        raise RuntimeError("Further integration refinement is required")


if __name__ == "__main__":
    main()
