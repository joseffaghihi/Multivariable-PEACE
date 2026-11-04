from study import BASE, E, KFold, L, RES, fit_models, generate, json, np, pd, weighted
from experiment import pair_grid


def run_joint(reps=40):
    U, V, W, lf, g1, g2 = pair_grid((E, L), 180)
    truth = weighted(np.hypot(g1, g2).ravel(), lf.ravel(), W.ravel(), 0.5)[0]
    rows = []
    checks = []
    for scenario, noise, sizes in [
        ("original", 0, [5000, 20000]),
        ("outcome_noise", 0.5, [1000, 5000, 20000]),
    ]:
        for n in sizes:
            for rep in range(reps):
                x, y = generate(n, seed=920000 + rep + int(noise * 100), noise=noise)
                for fold, (train, test) in enumerate(
                    KFold(2, shuffle=True, random_state=rep + 1901).split(x)
                ):
                    xt = x.iloc[train][BASE].to_numpy()
                    xe = x.iloc[test][BASE].to_numpy()
                    fits = fit_models(xt, y[train])
                    indices = [BASE.index(E), BASE.index(L)]
                    for method, model in fits.items():
                        ref = model.joint_reference(xe, indices)
                        g = model.joint_standardized_derivatives(xe[:, indices], ref)
                        rows.append(
                            dict(
                                scenario=scenario,
                                n=n,
                                rep=rep,
                                fold=fold,
                                method=method,
                                pair="Economy + Last upgrade",
                                d=0.5,
                                estimate=float(np.mean(np.linalg.norm(g, axis=1))),
                                truth=truth,
                            )
                        )
                        if rep == 0 and fold == 0:
                            smallref = xe[50:80]
                            r = model.joint_reference(smallref, indices)
                            vals = xe[:5, indices]
                            direct = []
                            for val in vals:
                                z = smallref.copy()
                                z[:, indices] = val
                                direct.append(
                                    [np.mean(model.derivative(z, j)) for j in indices]
                                )
                            checks.append(
                                float(
                                    np.max(
                                        np.abs(
                                            model.joint_standardized_derivatives(
                                                vals, r
                                            )
                                            - direct
                                        )
                                    )
                                )
                            )
                if (rep + 1) % 10 == 0:
                    print("completed flexible joint", scenario, n, rep + 1, flush=True)
            pd.DataFrame(rows).to_csv(RES / "joint_fold_results.csv", index=False)
    (RES / "joint_validation.json").write_text(
        json.dumps(
            {
                "max_standardization_identity_error": max(checks),
                "passed": max(checks) < 1e-8,
            },
            indent=2,
        )
    )
    assert max(checks) < 1e-8


if __name__ == "__main__":
    run_joint()
