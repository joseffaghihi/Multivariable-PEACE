from study import BASE, KFold, L, RES, fit_models, generate, json, np, pd, quad
from models import selected_additive
from scipy.optimize import brentq


def main(reps=40):
    rows = []
    root = brentq(lambda t: -8 / (t + 1) ** 2 + 0.06 * t, 0, 10)
    t1, w1 = quad(0, root, 250)
    t2, w2 = quad(root, 10, 250)
    negative = float(w1 @ (8 / (t1 + 1) ** 2 - 0.06 * t1) / 10)
    positive = float(w2 @ (-8 / (t2 + 1) ** 2 + 0.06 * t2) / 10)
    truth = positive + negative
    for n in [5000, 20000]:
        for rep in range(reps):
            x, y = generate(n, seed=930000 + rep, noise=0.5)
            y = y + 0.03 * x[L].to_numpy() ** 2
            for fold, (train, test) in enumerate(
                KFold(2, shuffle=True, random_state=rep + 1911).split(x)
            ):
                xt = x.iloc[train][BASE].to_numpy()
                xe = x.iloc[test][BASE].to_numpy()
                models = fit_models(xt, y[train])
                models["selected_additive"] = selected_additive(xt, y[train])
                j = BASE.index(L)
                for method, m in models.items():
                    ref = m.prepare_reference(xe, j)
                    g = m.standardized_derivative(xe[:, j], ref)
                    rows.append(
                        dict(
                            n=n,
                            rep=rep,
                            fold=fold,
                            variable=L,
                            method=method,
                            d=0.5,
                            estimate=float(np.mean(np.abs(g))),
                            positive=float(np.mean(np.maximum(g, 0))),
                            negative=float(np.mean(np.maximum(-g, 0))),
                            truth=truth,
                            true_positive=positive,
                            true_negative=negative,
                        )
                    )
            if (rep + 1) % 10 == 0:
                print("completed flexible shape challenge", n, rep + 1, flush=True)
        pd.DataFrame(rows).to_csv(RES / "shape_challenge_fold_results.csv", index=False)
    (RES / "shape_challenge_truth.json").write_text(
        json.dumps(
            {
                "peace": truth,
                "positive": positive,
                "negative": negative,
                "derivative_zero": root,
                "signed_average": positive - negative,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
