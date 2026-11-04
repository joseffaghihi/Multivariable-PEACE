"""Matched linear, additive and interaction-capable observational estimates.
Oracle functions are confined to generation/scoring and numerical checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment import (
    np,
    pd,
    plt,
    generate,
    quad,
    weighted,
    kde_values,
    NAMES,
    DOM,
    C,
    I,
    E,
    L,
    N,
    D,
    M,
    A,
    B,
    R,
    derivative,
    density,
    need_density,
    need_deriv,
    calls_deriv,
    sigmoid_mean,
    gx,
    gw,
)
from models import fit_models
from sklearn.model_selection import KFold
from scipy.special import expit
import json, argparse, hashlib
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
BASE = [C, I, E, L, N, D, M, B]
DESIGNS = {
    "calls": [C, E, L, B],
    "need": [N, C, I, E, L, B],
    "outcome": BASE,
    "ad": BASE + [A],
}
TARGETS = {"calls": [C], "need": [N], "outcome": [I, E, L, D, M, B], "ad": [A]}
STATUS = {
    C: "root overlap",
    E: "root overlap",
    L: "root overlap",
    B: "root overlap",
    N: "weak conditional overlap",
    I: "weak conditional overlap",
    M: "weak conditional overlap",
    D: "global support requires model extrapolation",
    A: "global support requires model extrapolation",
}
DEGREES = [0, 0.5, 1, 2]


def true_total_derivative(name, t, scenario):
    g = derivative(name, np.asarray(t))
    if scenario == "interaction":
        if name == N:
            n = np.asarray(t)
            g = g + 20 * (
                sigmoid_mean(0.3 * n)
                - 0.515
                + (n - 0.2) * 0.3 * sigmoid_mean(0.3 * n, deriv=True)
            )
        if name == C:
            n = 0.1 * np.asarray(t)[..., None] + 0.1 * gx
            extra = (
                sigmoid_mean(0.3 * n)
                - 0.515
                + (n - 0.2) * 0.3 * sigmoid_mean(0.3 * n, deriv=True)
            )
            g = g + 2 * (extra @ gw)
    return g


@lru_cache(maxsize=128)
def truth_total(name, d, scenario):
    t, w = quad(*DOM[name], 300)
    return weighted(
        true_total_derivative(name, t, scenario), np.log(density(name, t)), w, d
    )[0]


@lru_cache(maxsize=64)
def truth_conditional(name, scenario):
    if name == N:
        t, w = quad(-1.4, 1.8, 400)
        return float(
            w @ (np.abs(true_total_derivative(name, t, scenario)) * need_density(t))
        )
    if name == M and scenario == "interaction":
        split = 0.2 - 3 / 20
        out = 0
        for lo, hi in [(-1.4, split), (split, 1.8)]:
            t, w = quad(lo, hi, 200)
            out += float(w @ (np.abs(3 + 20 * (t - 0.2)) * need_density(t)))
        return out
    if name == C:
        t, w = quad(0, 4, 200)
        return float(w @ np.abs(true_total_derivative(name, t, scenario)) / 4)
    return {I: 0.3, E: 2, L: 8 / 11, D: 1, M: 3, A: 0, B: 0.2}[name]


def run(reps=40):
    rows = []
    conditional = []
    predict = []
    settings = []
    numerical = []
    scenarios = [
        ("original", 0, [1000, 5000, 20000]),
        ("outcome_noise", 0.5, [1000, 5000, 20000]),
        ("interaction", 0.5, [5000, 20000]),
    ]
    for scenario, noise, sizes in scenarios:
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
                        fitted = fit_models(xt, y[train], seed=1701)
                        for method, model in fitted.items():
                            mse = float(np.mean((model.predict(xe) - y[test]) ** 2))
                            predict.append(
                                dict(
                                    scenario=scenario,
                                    n=n,
                                    rep=rep,
                                    fold=fold,
                                    design=design,
                                    method=method,
                                    heldout_mse=mse,
                                )
                            )
                            settings.append(
                                dict(
                                    scenario=scenario,
                                    n=n,
                                    rep=rep,
                                    fold=fold,
                                    design=design,
                                    method=method,
                                    knots=0 if method == "linear" else model.knots,
                                    df=model.df,
                                    penalty=model.penalty,
                                    gcv=model.gcv,
                                )
                            )
                            for name in TARGETS[design]:
                                j = cols.index(name)
                                g = model.derivative(xe, j)
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
                                # For an unrestricted interaction-capable model,
                                # full fixed interventions D/A are not identified
                                # by data support. Never silently report them.
                                if name in [D, A] and method == "interaction":
                                    continue
                                ref = model.prepare_reference(xe, j)
                                inside = (xe[:, j] >= DOM[name][0]) & (
                                    xe[:, j] <= DOM[name][1]
                                )
                                assert inside.any()
                                totalg = model.standardized_derivative(
                                    xe[inside, j], ref
                                )
                                val = float(np.mean(np.abs(totalg)))
                                rows.append(
                                    dict(
                                        scenario=scenario,
                                        n=n,
                                        rep=rep,
                                        fold=fold,
                                        variable=name,
                                        method=method,
                                        d=0.5,
                                        estimate=val,
                                        truth=truth_total(name, 0.5, scenario),
                                        support=STATUS[name],
                                    )
                                )
                                t, w = model.integration_grid(*DOM[name], j, nodes=24)
                                tg = model.standardized_derivative(t, ref)
                                logf = np.log(
                                    kde_values(name, x.iloc[train][name].to_numpy(), t)
                                )
                                for d in [0, 1, 2]:
                                    val = weighted(tg, logf, w, d)[0]
                                    rows.append(
                                        dict(
                                            scenario=scenario,
                                            n=n,
                                            rep=rep,
                                            fold=fold,
                                            variable=name,
                                            method=method,
                                            d=d,
                                            estimate=val,
                                            truth=truth_total(name, d, scenario),
                                            support=STATUS[name],
                                        )
                                    )
                                if rep == 0 and fold == 0 and name in [C, N, L]:
                                    t2, w2 = model.integration_grid(
                                        *DOM[name], j, nodes=48
                                    )
                                    g2 = model.standardized_derivative(t2, ref)
                                    lf2 = np.log(
                                        kde_values(
                                            name, x.iloc[train][name].to_numpy(), t2
                                        )
                                    )
                                    err = max(
                                        abs(
                                            weighted(tg, logf, w, d)[0]
                                            - weighted(g2, lf2, w2, d)[0]
                                        )
                                        for d in [0, 1, 2]
                                    )
                                    numerical.append(
                                        dict(
                                            scenario=scenario,
                                            n=n,
                                            variable=name,
                                            method=method,
                                            absolute_difference=err,
                                        )
                                    )
                if (rep + 1) % 10 == 0:
                    print("completed flexible", scenario, n, rep + 1, flush=True)
            # Save intermediate complete scenarios so progress is reviewable.
            pd.DataFrame(rows).to_csv(RES / "global_fold_results.csv", index=False)
            pd.DataFrame(conditional).to_csv(
                RES / "conditional_fold_results.csv", index=False
            )
            pd.DataFrame(predict).to_csv(
                RES / "prediction_fold_results.csv", index=False
            )
            pd.DataFrame(settings).to_csv(RES / "selected_models.csv", index=False)
            pd.DataFrame(numerical).to_csv(
                RES / "quadrature_refinement.csv", index=False
            )
    print("all flexible scenarios complete", flush=True)


def validate():
    x, y = generate(1200, seed=810123, noise=0.5)
    cols = BASE
    X = x[cols].to_numpy()
    fitted = fit_models(X, y)
    errors = {}
    standardized = {}
    for kind, m in fitted.items():
        for j in [0, 3, 4, 6]:
            probe = X[100:127].copy()
            eps = 1e-5 * m.sd[j]
            plus = probe.copy()
            minus = probe.copy()
            plus[:, j] += eps
            minus[:, j] -= eps
            fd = (m.predict(plus) - m.predict(minus)) / (2 * eps)
            errors[kind + ":" + cols[j]] = float(
                np.max(np.abs(fd - m.derivative(probe, j)))
            )
            ref = X[200:237]
            r = m.prepare_reference(ref, j)
            t = np.quantile(X[:, j], [0.2, 0.5, 0.8])
            direct = []
            for value in t:
                z = ref.copy()
                z[:, j] = value
                direct.append(np.mean(m.derivative(z, j)))
            standardized[kind + ":" + cols[j]] = float(
                np.max(np.abs(m.standardized_derivative(t, r) - direct))
            )
    checks = {
        "analytic_derivative_max_error": max(errors.values()),
        "standardization_identity_max_error": max(standardized.values()),
        "derivative_details": errors,
        "standardization_details": standardized,
        "interaction_usage_total_truth": truth_total(M, 0.5, "interaction"),
        "interaction_usage_conditional_truth": truth_conditional(M, "interaction"),
    }
    checks["passed"] = bool(
        max(errors.values()) < 1e-5
        and max(standardized.values()) < 1e-8
        and checks["interaction_usage_conditional_truth"]
        > checks["interaction_usage_total_truth"]
    )
    (RES / "validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if not checks["passed"]:
        raise RuntimeError("Flexible estimator validation did not pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["run", "validate"])
    parser.add_argument("--reps", type=int, default=40)
    args = parser.parse_args()
    if args.part == "run":
        run(args.reps)
    else:
        validate()
