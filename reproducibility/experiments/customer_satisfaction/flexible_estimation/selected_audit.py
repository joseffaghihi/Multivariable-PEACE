"""Independent quadrature refinement for the added selected-additive method."""

from study import (
    DESIGNS,
    DOM,
    KFold,
    RES,
    TARGETS,
    generate,
    json,
    kde_values,
    np,
    pd,
    weighted,
)
from models import selected_additive


def main():
    rows = []
    for n in [1000, 5000, 20000]:
        x, y = generate(n, seed=910051, noise=0.5)
        train, test = next(KFold(2, shuffle=True, random_state=29002).split(x))
        for design, cols in DESIGNS.items():
            m = selected_additive(x.iloc[train][cols].to_numpy(), y[train])
            xe = x.iloc[test][cols].to_numpy()
            for name in TARGETS[design]:
                j = cols.index(name)
                ref = m.prepare_reference(xe, j)
                values = {}
                for nodes in [24, 96]:
                    t, w = m.integration_grid(*DOM[name], j, nodes=nodes)
                    g = m.standardized_derivative(t, ref)
                    lf = np.log(kde_values(name, x.iloc[train][name].to_numpy(), t))
                    values[nodes] = np.array(
                        [weighted(g, lf, w, d)[0] for d in [0, 1, 2]]
                    )
                err = np.abs(values[24] - values[96])
                scaled = err / (1 + np.abs(values[96]))
                rows.append(
                    dict(
                        n=n,
                        variable=name,
                        max_absolute_difference=float(err.max()),
                        max_scaled_difference=float(scaled.max()),
                    )
                )
    d = pd.DataFrame(rows)
    d.to_csv(RES / "selected_quadrature_audit.csv", index=False)
    check = {
        "max_absolute_difference": float(d.max_absolute_difference.max()),
        "max_scaled_difference": float(d.max_scaled_difference.max()),
        "passed": bool(
            d.max_absolute_difference.max() < 0.01
            and d.max_scaled_difference.max() < 0.001
        ),
    }
    (RES / "selected_integration_audit.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check))
    assert check["passed"]


if __name__ == "__main__":
    main()
