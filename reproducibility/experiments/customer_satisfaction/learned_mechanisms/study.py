"""DAG-informed, observationally fitted mechanisms and PEACE.

Truth is imported solely for data generation and scoring. Estimator functions
fit() and gradients() never call the oracle derivative/feature functions.
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment import (
    np,
    pd,
    plt,
    quad,
    gx,
    gw,
    expit,
    logit,
    norm,
    logsumexp,
    generate,
    structural_features,
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
    kde_values,
    weighted,
)
from scipy.optimize import brentq
import json, hashlib, argparse
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
DS = [0, 0.5, 1, 2]


def ols(z, y, robust=False):
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    mean = z.mean(0)
    scale = z.std(0)
    assert np.all(scale > 1e-12)
    design = np.column_stack([np.ones(len(z)), (z - mean) / scale])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    slope = coef[1:] / scale
    out = {
        "intercept": float(coef[0] - mean @ slope),
        "coef": slope,
        "resid": y - design @ coef,
    }
    if robust:
        inv = np.linalg.inv(design.T @ design)
        h = np.sum((design @ inv) * design, axis=1)
        meat = design.T @ (((out["resid"] / (1 - h)) ** 2)[:, None] * design)
        out["se"] = np.sqrt(np.diag(inv @ meat @ inv)[1:]) / scale
    return out


def fit(x, y, mode="graph"):
    """Model families/graph are supplied; numerical coefficients are learned."""
    models = {
        N: ols(x[C], x[N]),
        I: ols(x[C], x[I]),
        M: ols(x[N], logit(x[M])),
        D: ols(expit(x[N]), x[D]),
        A: ols(x[[M, L]], x[A]),
        R: ols(x[B] * expit(x[N]), x[R]),
    }
    for name in [N, M]:
        models[name]["sd"] = float(np.sqrt(np.mean(models[name]["resid"] ** 2)))
    basis = structural_features(x)
    columns = list(range(10)) if mode == "expanded" else list(range(8))
    if mode == "quadratic_upgrade":
        basis = np.column_stack([basis, x[L] ** 2])
        columns += [10]
    out = ols(basis[:, columns], y, robust=True)
    coeff = np.zeros(11)
    se = np.zeros(11)
    coeff[columns] = out["coef"]
    se[columns] = out["se"]
    return {
        "models": models,
        "outcome": coeff,
        "outcome_se": se,
        "mode": mode,
        "mean_bugs": float(x[B].mean()),
        "mean_sigmoid_need": float(expit(x[N]).mean()),
        "outcome_sd": float(np.sqrt(np.mean(out["resid"] ** 2))),
    }


def logistic_expect(t, sd, deriv=False):
    s = expit(np.asarray(t)[..., None] + sd * gx)
    return (s * (1 - s) if deriv else s) @ gw


def gradients(fitobj, name, t, blocked=()):
    m = fitobj["models"]
    beta = fitobj["outcome"]
    t = np.asarray(t)
    F = np.zeros(t.shape + (11,))
    if name == N:
        ds = expit(t) * (1 - expit(t))
        dm = m[M]["coef"][0] * logistic_expect(
            m[M]["intercept"] + m[M]["coef"][0] * t, m[M]["sd"], True
        )
        F[..., 0] = 1
        F[..., 1] = dm if M not in blocked else 0
        F[..., 3] = m[D]["coef"][0] * ds if D not in blocked else 0
        F[..., 8] = m[A]["coef"][0] * F[..., 1]
        F[..., 9] = m[R]["coef"][0] * fitobj["mean_bugs"] * ds
    elif name == C:
        n = m[N]["intercept"] + m[N]["coef"][0] * t[..., None] + m[N]["sd"] * gx
        ds = (expit(n) * (1 - expit(n))) @ gw
        dm = (
            m[N]["coef"][0]
            * m[M]["coef"][0]
            * (
                logistic_expect(
                    m[M]["intercept"] + m[M]["coef"][0] * n, m[M]["sd"], True
                )
                @ gw
            )
        )
        F[..., 0] = m[N]["coef"][0] if N not in blocked else 0
        F[..., 1] = dm if N not in blocked and M not in blocked else 0
        F[..., 3] = (
            m[N]["coef"][0] * m[D]["coef"][0] * ds
            if N not in blocked and D not in blocked
            else 0
        )
        F[..., 5] = 1
        F[..., 6] = m[I]["coef"][0] if I not in blocked else 0
        F[..., 8] = m[A]["coef"][0] * F[..., 1]
        F[..., 9] = (
            m[N]["coef"][0] * m[R]["coef"][0] * fitobj["mean_bugs"] * ds
            if N not in blocked
            else 0
        )
    elif name == L:
        F[..., 7] = -1 / (t + 1) ** 2
        F[..., 8] = m[A]["coef"][1]
        F[..., 10] = 2 * t
    elif name == M:
        F[..., 1] = 1
        F[..., 8] = m[A]["coef"][0]
    elif name == B:
        F[..., 4] = 1
        F[..., 9] = m[R]["coef"][0] * fitobj["mean_sigmoid_need"]
    else:
        F[..., {I: 6, E: 2, D: 3, A: 8, R: 9}[name]] = 1
    return F @ beta


def fitted_need_density(fitobj, t):
    m = fitobj["models"][N]
    a = m["intercept"]
    b = m["coef"][0]
    s = m["sd"]
    if abs(b) < 1e-8:
        return norm.pdf(t, a, s)
    low, high = sorted([a, a + 4 * b])
    return np.maximum(
        (norm.cdf((t - low) / s) - norm.cdf((t - high) / s)) / (high - low),
        np.finfo(float).tiny,
    )


def grid(name, k=128, grad=None):
    lo, hi = DOM[name]
    if grad is None:
        return quad(lo, hi, k)
    # Exact sign splits protect the upgrade sign-change stress scenario.
    probe = np.linspace(lo, hi, 101)
    vals = grad(probe)
    breaks = [lo, hi]
    for j in range(len(probe) - 1):
        if vals[j] * vals[j + 1] < 0:
            breaks.append(
                brentq(lambda z: float(grad(np.array([z]))[0]), probe[j], probe[j + 1])
            )
    breaks = sorted(breaks)
    ts = []
    ws = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        t, w = quad(left, right, k)
        ts.extend(t)
        ws.extend(w)
    return np.array(ts), np.array(ws)


def calculate(f, name, d, weights="fitted_model", sample=None, k=128):
    return calculate_curve(f, name, [d], weights, sample, k)[0]


def calculate_curve(f, name, ds, weights="fitted_model", sample=None, k=128):
    t, w = grid(name, k, lambda z: gradients(f, name, z))
    g = gradients(f, name, t)
    # Nonconstant gradients occur only for calls, need, upgrade. Other scalar
    # gradients are constants, so their reference density cancels exactly.
    if name == N:
        lf = np.log(fitted_need_density(f, t))
    else:
        lf = np.zeros_like(t)
    if weights == "kde" and name in [C, N, L]:
        lf = np.log(kde_values(name, sample[name].to_numpy(), t))
    return [weighted(g, lf, w, d) for d in ds]


@lru_cache(maxsize=256)
def true_value(name, d, scenario="original", k=256):
    grad = lambda t: (
        derivative(name, t)
        + (0.06 * t if name == L and scenario == "wrong_outcome_shape" else 0)
    )
    t, w = grid(name, k, grad)
    lf = np.log(density(name, t)) if name == N else np.zeros_like(t)
    return weighted(grad(t), lf, w, d)[0]


def simulate(reps=100):
    rows = []
    parameters = []
    for scenario, noise in [("original", 0), ("outcome_noise", 0.5)]:
        for n in [1000, 5000, 20000, 100000]:
            for rep in range(reps):
                x, y = generate(n, seed=410000 + rep + int(noise * 100), noise=noise)
                for mode in ["graph", "expanded"]:
                    f = fit(x, y, mode)
                    for name in NAMES:
                        for d, (est, pos, neg) in zip(DS, calculate_curve(f, name, DS)):
                            rows.append(
                                dict(
                                    scenario=scenario,
                                    n=n,
                                    rep=rep,
                                    model=mode,
                                    density="fitted_model",
                                    variable=name,
                                    d=d,
                                    estimate=est,
                                    truth=true_value(name, d),
                                    positive=pos,
                                    negative=neg,
                                )
                            )
                    if mode == "graph":
                        for name in [C, N, L]:
                            for d, (est, pos, neg) in zip(
                                [0.5, 1, 2],
                                calculate_curve(f, name, [0.5, 1, 2], "kde", x),
                            ):
                                rows.append(
                                    dict(
                                        scenario=scenario,
                                        n=n,
                                        rep=rep,
                                        model=mode,
                                        density="kde",
                                        variable=name,
                                        d=d,
                                        estimate=est,
                                        truth=true_value(name, d),
                                        positive=pos,
                                        negative=neg,
                                    )
                                )
                    if mode == "expanded":
                        for name, index in [(A, 8), (R, 9)]:
                            b = f["outcome"][index]
                            se = f["outcome_se"][index]
                            lo = b - 1.96 * se
                            hi = b + 1.96 * se
                            parameters.append(
                                dict(
                                    scenario=scenario,
                                    n=n,
                                    rep=rep,
                                    variable=name,
                                    signed_estimate=b,
                                    se=se,
                                    magnitude_lower=0
                                    if lo <= 0 <= hi
                                    else min(abs(lo), abs(hi)),
                                    magnitude_upper=max(abs(lo), abs(hi)),
                                    covered=bool(lo <= 0 <= hi),
                                )
                            )
            print("completed learned mechanisms", scenario, n, reps, flush=True)
    pd.DataFrame(rows).to_csv(RES / "replicates.csv", index=False)
    pd.DataFrame(parameters).to_csv(RES / "expanded_null_intervals.csv", index=False)


def stress(reps=100):
    rows = []
    for scenario in ["hidden_confounding", "wrong_outcome_shape"]:
        for rep in range(reps):
            x, y = generate(
                5000,
                seed=510000 + rep,
                noise=0.5,
                rho=0.5 if scenario == "hidden_confounding" else 0,
                gamma=0.1 if scenario == "hidden_confounding" else 0,
            )
            if scenario == "wrong_outcome_shape":
                y = y + 0.03 * x[L].to_numpy() ** 2
            for mode in (
                ["graph", "quadratic_upgrade"]
                if scenario == "wrong_outcome_shape"
                else ["graph"]
            ):
                f = fit(x, y, mode)
                for name in [C, N, L]:
                    for d in DS:
                        est = calculate(f, name, d)[0]
                        rows.append(
                            dict(
                                scenario=scenario,
                                n=5000,
                                rep=rep,
                                model=mode,
                                variable=name,
                                d=d,
                                estimate=est,
                                truth=true_value(name, d, scenario),
                            )
                        )
        print("completed learned stress", scenario, flush=True)
    pd.DataFrame(rows).to_csv(RES / "stress_replicates.csv", index=False)


def uncertainty(datasets=60, draws=150):
    rng = np.random.default_rng(611234)
    rows = []
    for n in [5000, 20000]:
        for rep in range(datasets):
            x, y = generate(n, seed=610000 + rep, noise=0.5)
            fitted = fit(x, y)
            values = {name: [] for name in NAMES if name not in [A, R]}
            for b in range(draws):
                idx = rng.integers(0, n, n)
                f = fit(x.iloc[idx], y[idx])
                for name in values:
                    if name in [C, N, L]:
                        v = calculate(f, name, 0.5)[0]
                    else:
                        v = float(gradients(f, name, np.array([sum(DOM[name]) / 2]))[0])
                    values[name].append(v)
            for name, v in values.items():
                lo, hi = np.quantile(v, [0.025, 0.975])
                if name not in [C, N, L]:
                    lo, hi = (
                        (0 if lo <= 0 <= hi else min(abs(lo), abs(hi))),
                        max(abs(lo), abs(hi)),
                    )
                truth = true_value(name, 0.5)
                point = calculate(fitted, name, 0.5)[0]
                rows.append(
                    dict(
                        n=n,
                        rep=rep,
                        variable=name,
                        draws=draws,
                        truth=truth,
                        estimate=point,
                        lower=lo,
                        upper=hi,
                        covered=bool(lo <= truth <= hi),
                    )
                )
            if (rep + 1) % 10 == 0:
                print("completed learned bootstrap", n, rep + 1, flush=True)
    pd.DataFrame(rows).to_csv(RES / "bootstrap_intervals.csv", index=False)


def pair_value(f, pair, d, k=64):
    u, wu = quad(*DOM[pair[0]], k)
    v, wv = quad(*DOM[pair[1]], k)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wu, wv)
    g1 = gradients(f, pair[0], U, blocked=pair[1:])
    g2 = gradients(f, pair[1], V, blocked=pair[:1])
    m = f["models"]
    if pair == (N, M):
        lf = (
            np.log(fitted_need_density(f, U))
            + norm.logpdf(logit(V), m[M]["intercept"] + m[M]["coef"][0] * U, m[M]["sd"])
            - np.log(V * (1 - V))
        )
    elif pair == (C, N):
        lf = norm.logpdf(V, m[N]["intercept"] + m[N]["coef"][0] * U, m[N]["sd"])
    else:
        lf = np.zeros_like(U)
    return weighted(np.hypot(g1, g2).ravel(), lf.ravel(), W.ravel(), d)[0]


def joint(reps=100):
    from experiment import pair_grid

    rows = []
    pairs = [(N, M), (C, N), (E, L)]
    truth = {}
    for pair in pairs:
        U, V, W, lf, g1, g2 = pair_grid(pair, 160)
        for d in DS:
            truth[(pair, d)] = weighted(
                np.hypot(g1, g2).ravel(), lf.ravel(), W.ravel(), d
            )[0]
    for n in [1000, 5000, 20000]:
        for rep in range(reps):
            x, y = generate(n, seed=710000 + rep, noise=0.5)
            f = fit(x, y)
            for pair in pairs:
                for d in DS:
                    rows.append(
                        dict(
                            n=n,
                            rep=rep,
                            pair=" + ".join(pair),
                            d=d,
                            estimate=pair_value(f, pair, d),
                            truth=truth[(pair, d)],
                        )
                    )
        print("completed learned joint", n, flush=True)
    pd.DataFrame(rows).to_csv(RES / "joint_replicates.csv", index=False)


def validate():
    checks = {}
    x, y = generate(5000, seed=810001, noise=0.5)
    f = fit(x, y, "expanded")
    # Independent propagation through the fitted equations, with shared random
    # draws, checks the pathwise derivative formulas without using true nodes.
    rng = np.random.default_rng(810002)
    size = 200000
    errorsN = rng.normal(size=size)
    errorsM = rng.normal(size=size)
    rootC = rng.uniform(0, 4, size)
    rootB = rng.uniform(0, 2, size)
    m = f["models"]

    def response(name, t):
        calls = np.full(size, t) if name == C else rootC
        need = (
            np.full(size, t)
            if name == N
            else m[N]["intercept"] + m[N]["coef"][0] * calls + m[N]["sd"] * errorsN
        )
        # The expanded bugs-faced response standardizes sigmoid need over its
        # empirical observed marginal; use that same reference in this check.
        if name == B:
            need = np.resize(x[N].to_numpy(), size)
        usage = (
            np.full(size, t)
            if name == M
            else expit(
                m[M]["intercept"] + m[M]["coef"][0] * need + m[M]["sd"] * errorsM
            )
        )
        discount = (
            np.full(size, t)
            if name == D
            else m[D]["intercept"] + m[D]["coef"][0] * expit(need)
        )
        interaction = (
            np.full(size, t)
            if name == I
            else m[I]["intercept"] + m[I]["coef"][0] * calls
        )
        upgrade = np.full(size, t) if name == L else np.full(size, 5.0)
        bugs = np.full(size, t) if name == B else np.full(size, f["mean_bugs"])
        ad = (
            np.full(size, t)
            if name == A
            else m[A]["intercept"] + m[A]["coef"][0] * usage + m[A]["coef"][1] * upgrade
        )
        reported = (
            np.full(size, t)
            if name == R
            else m[R]["intercept"] + m[R]["coef"][0] * bugs * expit(need)
        )
        economy = np.full(size, t) if name == E else np.full(size, 0.75)
        F = np.column_stack(
            [
                need,
                usage,
                economy,
                discount,
                bugs,
                calls,
                interaction,
                1 / (upgrade + 1),
                ad,
                reported,
                upgrade**2,
            ]
        )
        return F @ f["outcome"]

    deriverrors = {}
    quaderrors = {}
    for name in NAMES:
        t = sum(DOM[name]) / 2
        eps = 1e-4 * (DOM[name][1] - DOM[name][0])
        numerical = np.mean(
            (response(name, t + eps) - response(name, t - eps)) / (2 * eps)
        )
        analytic = gradients(f, name, np.array([t]))[0]
        deriverrors[name] = float(abs(numerical - analytic))
        quaderrors[name] = max(
            abs(calculate(f, name, d, k=128)[0] - calculate(f, name, d, k=256)[0])
            for d in DS
        )
    checks["fitted_SEM_derivative_common_random_errors"] = deriverrors
    checks["scalar_quadrature_errors"] = quaderrors
    checks["pair_quadrature_errors"] = {
        " + ".join(p): max(
            abs(pair_value(f, p, d, 64) - pair_value(f, p, d, 128)) for d in DS
        )
        for p in [(N, M), (C, N), (E, L)]
    }
    t, w = quad(-1.5, 1.8, 300)
    checks["fitted_need_density_integral"] = float(w @ fitted_need_density(f, t))
    # Check estimation parameters respond to changed observed data, and are not
    # silently supplied from the simulator.
    altered = x.copy()
    altered[N] = x[N] + 0.2 * x[C]
    f2 = fit(altered, y)
    checks["learned_need_slope_changes_with_data"] = bool(
        abs(f2["models"][N]["coef"][0] - f["models"][N]["coef"][0] - 0.2) < 1e-10
    )
    checks["passed"] = bool(
        max(deriverrors.values()) < 0.002
        and max(quaderrors.values()) < 1e-6
        and max(checks["pair_quadrature_errors"].values()) < 1e-6
        and abs(checks["fitted_need_density_integral"] - 1) < 1e-7
        and checks["learned_need_slope_changes_with_data"]
    )
    (RES / "validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    if not checks["passed"]:
        raise RuntimeError("learned-SEM validation failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "part",
        choices=["simulate", "stress", "uncertainty", "joint", "validate", "all"],
    )
    p.add_argument("--reps", type=int, default=100)
    args = p.parse_args()
    for name, fn in [
        ("simulate", lambda: simulate(args.reps)),
        ("stress", lambda: stress(args.reps)),
        ("uncertainty", uncertainty),
        ("joint", lambda: joint(args.reps)),
        ("validate", validate),
    ]:
        if args.part in [name, "all"]:
            fn()
