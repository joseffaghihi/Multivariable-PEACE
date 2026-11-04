"""Reproducible sodium PEACE validation; no modifications to original notebooks.

Run with --stage core, comparators, uncertainty, density, or plots.
All Monte Carlo seeds and settings are fixed in this file before simulation.
"""

import os
import tempfile

for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[_key] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "peace-sodium-matplotlib")
)
import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from scipy.stats import norm, gaussian_kde
from numpy.polynomial.legendre import leggauss
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
TARGETS = [
    (0.0, 1.0),
    (0.25, 1.0),
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (0.5, 0.0),
    (0.5, 2.0),
]
BASE_SEED = 20261013


@dataclass(frozen=True)
class Scenario:
    name: str = "baseline"
    sigma: float = 1.0
    age_sd: float = 5.0
    noise: float = 10.0
    rho: float = 0.0
    gamma: float = 0.0
    linear: bool = False
    mixture: bool = False


SCENARIOS = [
    Scenario(),
    Scenario("low_overlap", sigma=0.25),
    Scenario("high_noise", noise=20.0),
    Scenario("hidden_confounding", rho=0.5, gamma=10.0),
    Scenario("linear_response", linear=True),
    Scenario("skewed_treatment", mixture=True),
]


def generate(n, seed, sc=Scenario()):
    rng = np.random.default_rng(seed)
    a = rng.normal(65.0, sc.age_sd, n)
    u = rng.normal(size=n)
    e = rng.normal(size=n)
    if sc.mixture:
        e = rng.normal(0.0, 0.6, n) + np.where(rng.random(n) < 0.1, 2.7, -0.3)
    s = a / 18.0 + sc.sigma * (np.sqrt(1 - sc.rho**2) * e + sc.rho * u)
    y = response(s, a, sc) + sc.gamma * u + rng.normal(0.0, sc.noise, n)
    p = 0.4 * s + 0.3 * y + rng.normal(size=n)
    return np.column_stack([s, a]), y, p


def response(s, a, sc=Scenario()):
    if sc.linear:
        return 100.0 + 3.75 * s + (a - 65.0)
    return (
        90.0
        + 2.5 * s
        + 2.5 * np.maximum(s - 3.5, 0.0)
        + 10.0 * np.exp(0.1 * (a - 65.0))
    )


def true_derivatives(s, a, sc=Scenario()):
    if sc.linear:
        return np.full_like(s, 3.75), np.ones_like(a)
    return 2.5 + 2.5 * (s > 3.5), np.exp(0.1 * (a - 65.0))


def quad(lo, hi, n):
    x, w = leggauss(n)
    return lo + (x + 1) * (hi - lo) / 2, w * (hi - lo) / 2


class Grid:
    def __init__(self, n=48, sdomain=(0.0, 8.0), adomain=(50.0, 80.0)):
        # Splitting at the known kink makes oracle integration accurate.
        cuts = (
            [sdomain[0]]
            + ([3.5] if sdomain[0] < 3.5 < sdomain[1] else [])
            + [sdomain[1]]
        )
        parts = [quad(x, y, n) for x, y in zip(cuts[:-1], cuts[1:])]
        self.s = np.concatenate([p[0] for p in parts])
        self.ws = np.concatenate([p[1] for p in parts])
        self.a, self.wa = quad(*adomain, n)
        self.sdomain = sdomain
        self.adomain = adomain
        self.ss, self.aa = np.meshgrid(self.s, self.a, indexing="ij")


def gaussian_logs(grid, mean, cov):
    s, a = grid.ss, grid.aa
    vs, va = cov[0, 0], cov[1, 1]
    c = cov[0, 1]
    ma = mean[1]
    ms = mean[0]
    ls = norm.logpdf(grid.s, ms, np.sqrt(vs))
    la = norm.logpdf(grid.a, ma, np.sqrt(va))
    lsc = norm.logpdf(s, ms + c / va * (a - ma), np.sqrt(max(vs - c * c / va, 1e-10)))
    return lsc + la[None, :], ls, la


def truth_logs(grid, sc=Scenario()):
    if sc.mixture:
        la = norm.logpdf(grid.a, 65.0, sc.age_sd)
        terms = []
        margs = []
        for prob, shift in [(0.9, -0.3), (0.1, 2.7)]:
            terms.append(
                np.log(prob)
                + norm.logpdf(grid.ss, grid.aa / 18 + sc.sigma * shift, 0.6 * sc.sigma)
            )
            margs.append(
                np.log(prob)
                + norm.logpdf(
                    grid.s,
                    65 / 18 + sc.sigma * shift,
                    np.sqrt((sc.age_sd / 18) ** 2 + (0.6 * sc.sigma) ** 2),
                )
            )
        return logsumexp(terms, axis=0) + la[None, :], logsumexp(margs, axis=0), la
    cov = np.array(
        [
            [sc.sigma**2 + (sc.age_sd / 18) ** 2, sc.age_sd**2 / 18],
            [sc.age_sd**2 / 18, sc.age_sd**2],
        ]
    )
    return gaussian_logs(grid, np.array([65 / 18, 65.0]), cov)


def fitted_logs(grid, x):
    return gaussian_logs(grid, x.mean(axis=0), np.cov(x, rowvar=False, ddof=0))


def normalize_log(logw, weights, axis):
    shape = [1] * logw.ndim
    shape[axis] = len(weights)
    lw = logw + np.log(weights).reshape(shape)
    return np.exp(lw - logsumexp(lw, axis=axis, keepdims=True))


def aggregation(grid, logs, gs, ga, targets=TARGETS):
    lj, ls, la = logs
    vals = []
    for d, r in targets:
        ps = normalize_log(2 * d * lj, grid.ws, 0)
        pa = normalize_log(2 * d * lj, grid.wa, 1)
        qa = normalize_log(r * la, grid.wa, 0)
        qs = normalize_log(r * ls, grid.ws, 0)
        # Joint log density can replace conditional here: the conditioning
        # marginal cancels in each normalized inner integral.
        es = np.sum(ps * np.abs(gs)[:, None], axis=0)
        ea = np.sum(pa * np.abs(ga)[None, :], axis=1)
        vals.append((float(es @ qa), float(ea @ qs)))
    return np.array(vals)


def cov_sensitivity(grid, logs, gs, d, r):
    lj, ls, la = logs
    ps = normalize_log(2 * d * lj, grid.ws, 0)
    qa = normalize_log(r * la, grid.wa, 0)
    L = np.abs(gs)[:, None]
    lf = lj - la[None, :]
    el = np.sum(ps * L, axis=0)
    ef = np.sum(ps * lf, axis=0)
    vd = 2 * np.sum(qa * (np.sum(ps * L * lf, axis=0) - el * ef))
    vr = np.sum(qa * el * la) - np.sum(qa * el) * np.sum(qa * la)
    return vd, vr


class ResponseFit:
    """Additive regression; structural benchmark or flexible cubic splines.

    Splines use fixed knots in the target domain and a ridge penalty on the
    bounded, unscaled B-spline basis. The derivative uses centered differences
    of the basis, then linearly combined. This is verified below.
    """

    def __init__(self, x, y, kind="spline", knots=7):
        self.kind = kind
        if kind == "structural":
            self.model = LinearRegression().fit(self.features(x), y)
        else:
            self.ts = SplineTransformer(
                knots=np.linspace(0, 8, knots)[:, None],
                degree=3,
                include_bias=False,
                extrapolation="linear",
            ).fit(x[:, 0:1])
            self.ta = SplineTransformer(
                knots=np.linspace(50, 80, knots)[:, None],
                degree=3,
                include_bias=False,
                extrapolation="linear",
            ).fit(x[:, 1:2])
            self.model = Ridge(alpha=0.1).fit(self.features(x), y)

    def features(self, x):
        s, a = x[:, 0], x[:, 1]
        if self.kind == "structural":
            return np.column_stack(
                [s, np.maximum(s - 3.5, 0), np.exp(0.1 * (a - 65)), a - 65]
            )
        return np.column_stack(
            [self.ts.transform(s[:, None]), self.ta.transform(a[:, None])]
        )

    def derivatives(self, grid):
        if self.kind == "structural":
            b = self.model.coef_
            return b[0] + b[1] * (grid.s > 3.5), 0.1 * b[2] * np.exp(
                0.1 * (grid.a - 65)
            ) + b[3]
        b = self.model.coef_
        eps = 1e-4
        bs = (
            self.ts.transform((grid.s + eps)[:, None])
            - self.ts.transform((grid.s - eps)[:, None])
        ) / (2 * eps)
        ba = (
            self.ta.transform((grid.a + eps)[:, None])
            - self.ta.transform((grid.a - eps)[:, None])
        ) / (2 * eps)
        return bs @ b[: bs.shape[1]], ba @ b[bs.shape[1] :]


def summarize(df, keys, path):
    rows = []
    for key, z in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        err = z.estimate.to_numpy() - z.truth.to_numpy()
        m = len(err)
        rmse = np.sqrt(np.mean(err**2))
        rows.append(
            dict(zip(keys, key))
            | dict(
                repetitions=m,
                truth=z.truth.mean(),
                mean=z.estimate.mean(),
                bias=err.mean(),
                rmse=rmse,
                sd=z.estimate.std(ddof=1),
                bias_mcse=err.std(ddof=1) / np.sqrt(m),
                rmse_mcse=np.std(err**2, ddof=1) / (2 * np.sqrt(m) * rmse)
                if rmse > 0
                else 0,
            )
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def core(reps=100):
    grid = Grid(96)
    truth = {
        sc.name: aggregation(
            grid, truth_logs(grid, sc), *true_derivatives(grid.s, grid.a, sc)
        )
        for sc in SCENARIOS
    }
    rows = []
    designs = (
        [(SCENARIOS[0], n) for n in [500, 2000, 8000]]
        + [(s, 2000) for s in SCENARIOS[1:]]
        + [(SCENARIOS[3], 8000)]
    )
    for si, (sc, n) in enumerate(designs):
        for rep in range(reps):
            seed = BASE_SEED + si * 100000 + rep
            x, y, p = generate(n, seed, sc)
            estlog = fitted_logs(grid, x)
            truelog = truth_logs(grid, sc)
            for method in ["structural", "spline"]:
                fit = ResponseFit(x, y, method)
                gs, ga = fit.derivatives(grid)
                for density, logs in [
                    ("estimated_gaussian", estlog),
                    ("oracle", truelog),
                ]:
                    val = aggregation(grid, logs, gs, ga)
                    for ti, (d, r) in enumerate(TARGETS):
                        for j, exposure in enumerate(["sodium", "age_controlled"]):
                            rows.append(
                                dict(
                                    scenario=sc.name,
                                    n=n,
                                    rep=rep,
                                    seed=seed,
                                    method=method,
                                    density=density,
                                    exposure=exposure,
                                    d=d,
                                    r=r,
                                    truth=truth[sc.name][ti, j],
                                    estimate=val[ti, j],
                                )
                            )
            if (rep + 1) % 25 == 0:
                print("core", sc.name, n, rep + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation_raw.csv", index=False)
    summarize(
        df,
        ["scenario", "n", "method", "density", "exposure", "d", "r"],
        OUT / "simulation_summary.csv",
    )
    (OUT / "design.json").write_text(
        json.dumps(
            dict(
                seed=BASE_SEED,
                repetitions=reps,
                targets=TARGETS,
                scenarios=[asdict(s) for s in SCENARIOS],
                designs=[dict(scenario=s.name, n=n) for s, n in designs],
                sodium_domain=grid.sdomain,
                age_domain=grid.adomain,
                quadrature_nodes_per_sodium_segment=96,
                age_quadrature_nodes=96,
                spline_knots_per_input=7,
                ridge_alpha=0.1,
            ),
            indent=2,
        )
    )


def oracle():
    grid = Grid(128)
    logs = truth_logs(grid)
    gs, ga = true_derivatives(grid.s, grid.a)
    ds = np.linspace(0, 3, 121)
    rs = [0, 0.5, 1, 2, 3]
    rows = []
    for r in rs:
        v = aggregation(grid, logs, gs, ga, [(d, r) for d in ds])
        for i, d in enumerate(ds):
            rows.append(dict(d=d, r=r, sodium=v[i, 0], age_controlled=v[i, 1]))
    pd.DataFrame(rows).to_csv(OUT / "oracle_curves.csv", index=False)
    checks = []
    ref = aggregation(grid, logs, gs, ga)
    for n in [24, 48, 96, 192]:
        g = Grid(n)
        v = aggregation(g, truth_logs(g), *true_derivatives(g.s, g.a))
        checks.append(
            dict(
                check="quadrature",
                setting=str(n),
                max_abs_difference=float(np.max(abs(v - ref))),
            )
        )
    # Independent closed-form conditional-normal calculation for sodium.
    for d, r in TARGETS:
        q = normalize_log(r * logs[2], grid.wa, 0)
        if d == 0:
            p = np.full(len(grid.a), (8 - 3.5) / 8)
        else:
            sd = 1 / np.sqrt(2 * d)
            mu = grid.a / 18
            p = (ndtr((8 - mu) / sd) - ndtr((3.5 - mu) / sd)) / (
                ndtr((8 - mu) / sd) - ndtr((0 - mu) / sd)
            )
        closed = 2.5 + 2.5 * (p @ q)
        actual = aggregation(grid, logs, gs, ga, [(d, r)])[0, 0]
        checks.append(
            dict(
                check="closed_form_sodium",
                setting=f"{d},{r}",
                max_abs_difference=abs(closed - actual),
            )
        )
    d, r = 0.7, 1.2
    eps = 1e-4
    vd, vr = cov_sensitivity(grid, logs, gs, d, r)
    fd = (
        aggregation(grid, logs, gs, ga, [(d + eps, r)])[0, 0]
        - aggregation(grid, logs, gs, ga, [(d - eps, r)])[0, 0]
    ) / (2 * eps)
    fr = (
        aggregation(grid, logs, gs, ga, [(d, r + eps)])[0, 0]
        - aggregation(grid, logs, gs, ga, [(d, r - eps)])[0, 0]
    ) / (2 * eps)
    checks.extend(
        [
            dict(
                check="d_covariance_identity",
                setting="log_density",
                max_abs_difference=abs(vd - fd),
            ),
            dict(
                check="r_covariance_identity",
                setting="log_density",
                max_abs_difference=abs(vr - fr),
            ),
        ]
    )
    sc = Scenario("linear", linear=True)
    flat = aggregation(grid, logs, *true_derivatives(grid.s, grid.a, sc))
    checks.append(
        dict(
            check="constant_derivative_flatness",
            setting="all_targets",
            max_abs_difference=float(np.max(abs(flat - [3.75, 1]))),
        )
    )
    x, y, p = generate(2000, BASE_SEED)
    fit = ResponseFit(x, y)
    dsfit, dafit = fit.derivatives(grid)
    eps = 2e-4
    xplus = np.column_stack([grid.s + eps, np.full(len(grid.s), 65.0)])
    xminus = xplus.copy()
    xminus[:, 0] -= 2 * eps
    fds = (
        fit.model.predict(fit.features(xplus)) - fit.model.predict(fit.features(xminus))
    ) / (2 * eps)
    checks.append(
        dict(
            check="spline_derivative",
            setting="prediction_difference",
            max_abs_difference=float(np.max(abs(fds - dsfit))),
        )
    )
    # Independent reproduction of finite-difference simulator derivatives.
    eps = 1e-5
    for s, truth in [(3.0, 2.5), (4.0, 5.0)]:
        fd = (response(s + eps, 65.0) - response(s - eps, 65.0)) / (2 * eps)
        checks.append(
            dict(
                check="simulator_sodium_derivative",
                setting=str(s),
                max_abs_difference=abs(fd - truth),
            )
        )
    pd.DataFrame(checks).to_csv(OUT / "validation_checks.csv", index=False)
    assert max(z["max_abs_difference"] for z in checks) < 2e-5, checks
    # Domain sensitivity: distinct estimands, not numerical error.
    rows = []
    for sd in [(0.0, 7.0), (0.0, 8.0), (0.0, 9.0), (1.0, 7.0)]:
        g = Grid(96, sdomain=sd)
        v = aggregation(
            g, truth_logs(g), *true_derivatives(g.s, g.a), [(d, 1) for d in ds]
        )
        for i, d in enumerate(ds):
            rows.append(
                dict(domain=str(sd), d=d, sodium=v[i, 0], age_controlled=v[i, 1])
            )
    pd.DataFrame(rows).to_csv(OUT / "domain_sensitivity.csv", index=False)
    # Direct versus total age derivative and finite contrasts.
    a = np.linspace(50, 80, 301)
    direct = np.exp(0.1 * (a - 65))
    total = direct + (2.5 + 2.5 * ndtr(a / 18 - 3.5)) / 18
    pd.DataFrame(
        dict(age=a, controlled_derivative=direct, total_derivative=total)
    ).to_csv(OUT / "age_pathways.csv", index=False)
    vals = aggregation(grid, logs, gs, ga)
    pd.DataFrame(
        [
            dict(exposure=e, d=d, r=r, value=vals[i, j])
            for i, (d, r) in enumerate(TARGETS)
            for j, e in enumerate(["sodium", "age_controlled"])
        ]
    ).to_csv(OUT / "oracle_targets.csv", index=False)
    print("oracle checks passed", flush=True)


def comparator_truth(sc):
    # Conditional Gaussian residual projections: Stein's identity. These are
    # treatment-linear slope targets, not arbitrary nonlinear contrasts.
    z, w = quad(-8.0, 8.0, 256)
    w = w * norm.pdf(z)
    w = w / w.sum()
    a = 65 + sc.age_sd * z
    vs = sc.sigma**2 + (sc.age_sd / 18) ** 2
    c = sc.age_sd**2 / 18
    s = 65 / 18 + np.sqrt(vs) * z
    theta_s = (
        np.full_like(a, 3.75)
        if sc.linear
        else 2.5 + 2.5 * ndtr((a / 18 - 3.5) / sc.sigma)
    )
    ma = 65 + c / vs * (s - 65 / 18)
    va = sc.age_sd**2 - c * c / vs
    theta_a = np.ones_like(s) if sc.linear else np.exp(0.1 * (ma - 65) + 0.005 * va)
    return dict(sodium=float(theta_s @ w), age_controlled=float(theta_a @ w))


def comparators(reps=50):
    from econml.dml import CausalForestDML, LinearDML

    rows = []
    scs = [SCENARIOS[i] for i in [0, 1, 3]]
    z, w = quad(-7.0, 7.0, 1001)
    w = w * norm.pdf(z)
    w = w / w.sum()
    for si, sc in enumerate(scs):
        truths = comparator_truth(sc)
        for rep in range(reps):
            seed = BASE_SEED + 900000 + si * 10000 + rep
            x, y, p = generate(2000, seed, sc)
            for j, exposure in enumerate(["sodium", "age_controlled"]):
                t = x[:, j]
                cov = x[:, 1 - j : 2 - j] if j == 0 else x[:, 0:1]
                evalcov = (
                    (65 + sc.age_sd * z)[:, None]
                    if j == 0
                    else (65 / 18 + np.sqrt(sc.sigma**2 + (sc.age_sd / 18) ** 2) * z)[
                        :, None
                    ]
                )
                nuisance = make_pipeline(
                    SplineTransformer(n_knots=7, degree=3),
                    StandardScaler(),
                    Ridge(alpha=0.1),
                )
                for method in ["CausalForestDML", "LinearDML"]:
                    kw = dict(
                        model_y=nuisance,
                        model_t=LinearRegression(),
                        cv=3,
                        random_state=seed,
                    )
                    if method == "CausalForestDML":
                        model = CausalForestDML(
                            **kw,
                            n_estimators=120,
                            max_depth=8,
                            min_samples_leaf=10,
                            n_jobs=1,
                            inference=False,
                        )
                    else:
                        model = LinearDML(
                            **kw,
                            featurizer=SplineTransformer(
                                n_knots=5, degree=3, include_bias=False
                            ),
                        )
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        model.fit(Y=y, T=t, X=cov, inference=None)
                    estimates = np.asarray(
                        model.const_marginal_effect(evalcov)
                    ).reshape(-1)
                    assert np.all(np.isfinite(estimates))
                    rows.append(
                        dict(
                            scenario=sc.name,
                            n=2000,
                            rep=rep,
                            method=method,
                            exposure=exposure,
                            truth=truths[exposure],
                            estimate=float(estimates @ w),
                            warnings=" | ".join(
                                sorted(set(str(z.message) for z in captured))
                            ),
                        )
                    )
            if (rep + 1) % 10 == 0:
                print("comparators", sc.name, rep + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comparators_raw.csv", index=False)
    summarize(
        df, ["scenario", "n", "method", "exposure"], OUT / "comparators_summary.csv"
    )


def uncertainty(reps=50, boots=100):
    grid = Grid(64)
    targets = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    truth = aggregation(
        grid, truth_logs(grid), *true_derivatives(grid.s, grid.a), targets
    )
    rows = []
    for rep in range(reps):
        seed = BASE_SEED + 1500000 + rep
        x, y, p = generate(2000, seed)
        fit = ResponseFit(x, y)
        estimate = aggregation(
            grid, fitted_logs(grid, x), *fit.derivatives(grid), targets
        )
        rng = np.random.default_rng(seed + 10000)
        v = []
        for b in range(boots):
            ix = rng.integers(0, len(y), len(y))
            xb = x[ix]
            yb = y[ix]
            f = ResponseFit(xb, yb)
            v.append(
                aggregation(grid, fitted_logs(grid, xb), *f.derivatives(grid), targets)
            )
        low, high = np.quantile(v, [0.025, 0.975], axis=0)
        for ti, (d, r) in enumerate(targets):
            for j, e in enumerate(["sodium", "age_controlled"]):
                rows.append(
                    dict(
                        rep=rep,
                        d=d,
                        r=r,
                        exposure=e,
                        truth=truth[ti, j],
                        estimate=estimate[ti, j],
                        lower=low[ti, j],
                        upper=high[ti, j],
                        covered=low[ti, j] <= truth[ti, j] <= high[ti, j],
                        width=high[ti, j] - low[ti, j],
                    )
                )
        if (rep + 1) % 10 == 0:
            print("uncertainty", rep + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "uncertainty_raw.csv", index=False)
    summary = (
        df.groupby(["exposure", "d", "r"])
        .agg(
            repetitions=("covered", "size"),
            coverage=("covered", "mean"),
            mean_width=("width", "mean"),
        )
        .reset_index()
    )
    summary["coverage_mcse"] = np.sqrt(
        summary.coverage * (1 - summary.coverage) / summary.repetitions
    )
    summary.to_csv(OUT / "uncertainty_summary.csv", index=False)


def density(reps=30):
    grid = Grid(48)
    targets = [(0.25, 1.0), (0.5, 1.0), (1.0, 1.0)]
    rows = []
    for si, sc in enumerate([SCENARIOS[0], SCENARIOS[-1]]):
        truth = aggregation(
            grid, truth_logs(grid, sc), *true_derivatives(grid.s, grid.a, sc), targets
        )
        for rep in range(reps):
            seed = BASE_SEED + 1800000 + si * 10000 + rep
            x, y, p = generate(2000, seed, sc)
            # Hold response slopes at truth to isolate the density contribution.
            g = true_derivatives(grid.s, grid.a, sc)
            for method in ["Gaussian", "KDE_Scott", "KDE_half", "KDE_double"]:
                if method == "Gaussian":
                    logs = fitted_logs(grid, x)
                else:
                    factor = {"KDE_Scott": 1.0, "KDE_half": 0.5, "KDE_double": 2.0}[
                        method
                    ]
                    joint = gaussian_kde(
                        x.T, bw_method=lambda kde: kde.scotts_factor() * factor
                    )
                    ks = gaussian_kde(
                        x[:, 0], bw_method=lambda kde: kde.scotts_factor() * factor
                    )
                    ka = gaussian_kde(
                        x[:, 1], bw_method=lambda kde: kde.scotts_factor() * factor
                    )
                    logs = (
                        joint.logpdf(
                            np.vstack([grid.ss.ravel(), grid.aa.ravel()])
                        ).reshape(grid.ss.shape),
                        ks.logpdf(grid.s),
                        ka.logpdf(grid.a),
                    )
                v = aggregation(grid, logs, *g, targets)
                for ti, (d, r) in enumerate(targets):
                    for j, e in enumerate(["sodium", "age_controlled"]):
                        rows.append(
                            dict(
                                scenario=sc.name,
                                rep=rep,
                                method=method,
                                exposure=e,
                                d=d,
                                r=r,
                                truth=truth[ti, j],
                                estimate=v[ti, j],
                            )
                        )
            if (rep + 1) % 10 == 0:
                print("density", sc.name, rep + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "density_raw.csv", index=False)
    summarize(
        df, ["scenario", "method", "exposure", "d", "r"], OUT / "density_summary.csv"
    )


def savefig(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")


def plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    curves = pd.read_csv(OUT / "oracle_curves.csv")
    cols = ["#2458a6", "#c26520", "#2a8b64", "#9b4e9d", "#67727c"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (r, z) in enumerate(curves.groupby("r")):
        ax[0].plot(z.d, z.sodium, color=cols[k], label=f"r = {r:g}")
        ax[1].plot(z.d, z.age_controlled, color=cols[k], label=f"r = {r:g}")
    for a, title, yl in zip(
        ax,
        [
            "Sodium: weighted local responsiveness",
            "Age: controlled local responsiveness",
        ],
        ["mmHg per sodium unit", "mmHg per year"],
    ):
        a.set(xlabel="Treatment weighting d", ylabel=yl, title=title)
        a.grid(alpha=0.2)
    ax[1].legend(ncol=2, fontsize=9)
    fig.suptitle(
        "Exact PEACE curves | fixed domains S: [0, 8], A: [50, 80]", fontsize=13
    )
    fig.tight_layout()
    savefig(fig, "01_oracle_curves")
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    s = np.linspace(0, 8, 401)
    ax[0].plot(s, response(s, 65), color=cols[0])
    ax[0].axvline(3.5, ls=":", color="gray")
    ax[0].set(
        xlabel="Sodium",
        ylabel="Expected blood pressure (mmHg)",
        title="Interventional response at age 65",
    )
    ax[1].step(s, 2.5 + 2.5 * (s > 3.5), where="post", color=cols[0])
    ax[1].set(
        xlabel="Sodium", ylabel="mmHg per sodium unit", title="Local sodium response"
    )
    ax[1].set_ylim(0, 5.5)
    age = pd.read_csv(OUT / "age_pathways.csv")
    ax[2].plot(
        age.age,
        age.controlled_derivative,
        label="Controlled: sodium fixed",
        color=cols[0],
    )
    ax[2].plot(
        age.age, age.total_derivative, label="Total: sodium responds", color=cols[1]
    )
    ax[2].set(xlabel="Age", ylabel="mmHg per year", title="Which pathways remain open?")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "02_response_and_pathways")
    plt.close(fig)
    domain = pd.read_csv(OUT / "domain_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, z in domain.groupby("domain"):
        ax.plot(z.d, z.sodium, label="Sodium " + label)
    ax.set(
        xlabel="d (r = 1)",
        ylabel="Mean PEACE (mmHg per sodium unit)",
        title="Domain choice changes the estimand, especially near d = 0",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig(fig, "03_domain_sensitivity")
    plt.close(fig)
    sim = pd.read_csv(OUT / "simulation_summary.csv")
    z = sim[
        (sim.scenario == "baseline")
        & (sim.density == "estimated_gaussian")
        & (sim.r == 1)
        & (sim.d == 0.5)
    ]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for j, e in enumerate(["sodium", "age_controlled"]):
        for method, color in [("structural", cols[0]), ("spline", cols[1])]:
            zz = z[(z.exposure == e) & (z.method == method)].sort_values("n")
            ax[j].errorbar(
                zz.n,
                zz.rmse,
                yerr=1.96 * zz.rmse_mcse,
                marker="o",
                label=method,
                color=color,
                capsize=3,
            )
        ax[j].set(
            xscale="log",
            xlabel="Sample size",
            ylabel="RMSE",
            title=e.replace("_", " ").title(),
        )
        ax[j].set_xticks([500, 2000, 8000], ["500", "2,000", "8,000"])
        ax[j].minorticks_off()
        ax[j].legend()
        ax[j].grid(alpha=0.2)
    fig.suptitle(
        "Observational PEACE estimation | d = 0.5, r = 1\nBars show Monte Carlo uncertainty in RMSE, not individual estimate intervals",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, "04_estimation_sample_size")
    plt.close(fig)
    z = sim[
        (sim.n == 2000)
        & (sim.density == "estimated_gaussian")
        & (sim.r == 1)
        & (sim.d == 0.5)
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    order = [s.name for s in SCENARIOS]
    for j, e in enumerate(["sodium", "age_controlled"]):
        for off, (method, color) in zip(
            [-0.15, 0.15], [("structural", cols[0]), ("spline", cols[1])]
        ):
            zz = (
                z[(z.exposure == e) & (z.method == method)]
                .set_index("scenario")
                .loc[order]
            )
            ax[j].errorbar(
                np.arange(len(order)) + off,
                zz.bias,
                yerr=1.96 * zz.bias_mcse,
                fmt="o",
                color=color,
                capsize=3,
                label=method,
            )
        ax[j].axhline(0, color="gray", lw=1)
        ax[j].set_xticks(
            range(len(order)), [s.replace("_", "\n") for s in order], fontsize=8
        )
        ax[j].set(ylabel="Bias", title=e.replace("_", " ").title())
        ax[j].legend()
    fig.suptitle("Bias against true causal PEACE | n = 2,000, d = 0.5, r = 1")
    fig.tight_layout()
    savefig(fig, "05_scenario_bias")
    plt.close(fig)
    if (OUT / "comparators_summary.csv").exists():
        cmp = pd.read_csv(OUT / "comparators_summary.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        order = ["baseline", "low_overlap", "hidden_confounding"]
        for j, e in enumerate(["sodium", "age_controlled"]):
            for off, (method, color) in zip(
                [-0.12, 0.12], [("CausalForestDML", cols[0]), ("LinearDML", cols[1])]
            ):
                zz = (
                    cmp[(cmp.exposure == e) & (cmp.method == method)]
                    .set_index("scenario")
                    .loc[order]
                )
                ax[j].errorbar(
                    np.arange(3) + off,
                    zz.bias,
                    yerr=1.96 * zz.bias_mcse,
                    fmt="o",
                    label=method,
                    color=color,
                    capsize=3,
                )
            ax[j].axhline(0, color="gray")
            ax[j].set_xticks(range(3), [s.replace("_", "\n") for s in order])
            ax[j].set(
                ylabel="Bias against own slope target",
                title=e.replace("_", " ").title(),
            )
            ax[j].legend(fontsize=8)
        fig.suptitle("Causal Forest and DML: separately validated slope targets")
        fig.tight_layout()
        savefig(fig, "06_comparator_bias")
        plt.close(fig)
    if (OUT / "density_summary.csv").exists():
        den = pd.read_csv(OUT / "density_summary.csv")
        z = den[(den.d == 1) & (den.r == 1)]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        methods = ["Gaussian", "KDE_Scott", "KDE_half", "KDE_double"]
        for j, sc in enumerate(["baseline", "skewed_treatment"]):
            for off, (e, color) in zip(
                [-0.15, 0.15], [("sodium", cols[0]), ("age_controlled", cols[1])]
            ):
                zz = (
                    z[(z.scenario == sc) & (z.exposure == e)]
                    .set_index("method")
                    .loc[methods]
                )
                ax[j].bar(
                    np.arange(4) + off,
                    zz.rmse,
                    width=0.3,
                    label=e.replace("_", " "),
                    color=color,
                )
            ax[j].set_xticks(
                range(4),
                ["Gaussian", "KDE Scott", "KDE half", "KDE double"],
                rotation=15,
            )
            ax[j].set(
                ylabel="RMSE from density estimation alone",
                title=sc.replace("_", " ").title(),
            )
            ax[j].legend(fontsize=8)
        fig.suptitle("Density specification and bandwidth sensitivity | d = 1, r = 1")
        fig.tight_layout()
        savefig(fig, "07_density_sensitivity")
        plt.close(fig)
    if (OUT / "uncertainty_summary.csv").exists():
        z = pd.read_csv(OUT / "uncertainty_summary.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        for e, color in [("sodium", cols[0]), ("age_controlled", cols[1])]:
            zz = z[z.exposure == e]
            ax.errorbar(
                zz.d,
                zz.coverage,
                yerr=1.96 * zz.coverage_mcse,
                fmt="o-",
                capsize=3,
                label=e.replace("_", " "),
                color=color,
            )
        ax.axhline(0.95, color="gray", ls="--", label="Nominal 95%")
        ax.set(
            xlabel="d (r = 1)",
            ylabel="Empirical coverage",
            ylim=(0, 1.05),
            title="Percentile bootstrap: coverage is a diagnostic",
        )
        ax.legend()
        fig.tight_layout()
        savefig(fig, "08_bootstrap_coverage")
        plt.close(fig)
    print("plots saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["core", "comparators", "uncertainty", "density", "plots", "oracle"],
    )
    parser.add_argument("--reps", type=int)
    parser.add_argument("--boots", type=int, default=100)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "core":
            oracle()
            core(args.reps or 100)
        elif args.stage == "oracle":
            oracle()
        elif args.stage == "comparators":
            comparators(args.reps or 50)
        elif args.stage == "uncertainty":
            uncertainty(args.reps or 50, args.boots)
        elif args.stage == "density":
            density(args.reps or 30)
        else:
            plots()
