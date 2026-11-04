"""Standalone customer SEM experiment. Run with the accompanying locked environment."""

import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "peace-customer-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "peace-customer-cache")
)
for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[key] = "1"
from pathlib import Path
import json, itertools, warnings
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm, beta, gaussian_kde
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
NAMES = [
    "Sales calls",
    "Interactions",
    "Economy",
    "Last upgrade",
    "Product need",
    "Discount",
    "Monthly usage",
    "Ad spend",
    "Bugs faced",
    "Bugs reported",
]
C, I, E, L, N, D, M, A, B, R = NAMES
PARENTS = {
    C: [],
    I: [C],
    E: [],
    L: [],
    N: [C],
    D: [N],
    M: [N],
    A: [M, L],
    B: [],
    R: [B, N],
}
DOM = {
    C: (0, 4),
    I: (0.2, 4.8),
    E: (0.5, 1),
    L: (0, 10),
    N: (-0.3, 0.7),
    D: (0.07, 0.18),
    M: (0.35, 0.65),
    A: (0.3, 1.2),
    B: (0, 2),
    R: (0, 1.5),
}
BOUNDS = {
    C: (0, 4),
    E: (0.5, 1),
    L: (0, 10),
    B: (0, 2),
    M: (0, 1),
    D: (0, 0.275),
    R: (0, 2),
    A: (0, 1.49),
}
DEGREES = np.array([0, 0.25, 0.5, 1, 2, 3.0])
FEATURES = [N, M, E, D, B, C, I, "reciprocal upgrade", A, R]
gx, gw = hermgauss(24)
gx = gx * np.sqrt(2)
gw = gw / np.sqrt(np.pi)


@lru_cache(maxsize=4096)
def quad(lo, hi, k=120):
    x, w = leggauss(k)
    return (hi + lo) / 2 + (hi - lo) / 2 * x, w * (hi - lo) / 2


def generate(n, seed=0, fixed=None, b=3.0, noise=0.0, rho=0.0, gamma=0.0, skew=False):
    rng = np.random.default_rng(seed)
    fixed = {} if fixed is None else fixed
    x = {}

    def put(k, v):
        x[k] = np.broadcast_to(fixed.get(k, v), (n,)).copy()

    latent = rng.normal(size=n)
    put(C, rng.uniform(0, 4, n))
    put(I, x[C] + rng.normal(0.5, 0.05, n))
    put(E, rng.uniform(0.5, 1, n))
    put(L, 10 * rng.beta(2, 2, n) if skew else rng.uniform(0, 10, n))
    put(
        N,
        0.1 * x[C] + 0.1 * (rho * latent + np.sqrt(1 - rho * rho) * rng.normal(size=n)),
    )
    put(D, 0.25 * (1 - expit(x[N])) + rng.uniform(0, 0.025, n))
    put(M, expit(0.3 * x[N] + rng.normal(0, 0.1, n)))
    put(A, x[M] * rng.uniform(0.95, 0.99, n) + (10 - x[L]) * 0.05)
    put(B, rng.uniform(0, 2, n))
    put(R, x[B] * expit(x[N]))
    x = pd.DataFrame(x)
    y = (
        5 * x[N]
        + b * x[M]
        + 2 * x[E]
        + x[D]
        - 0.2 * x[B]
        + 0.2 * x[C]
        + 0.3 * x[I]
        + 8 / (x[L] + 1)
    )
    return x, y.to_numpy() + noise * rng.normal(size=n) + gamma * latent


def sigmoid_mean(t, sd=0.1, deriv=False):
    p = expit(np.asarray(t)[..., None] + sd * gx)
    return (p * (1 - p) if deriv else p) @ gw


def need_density(t):
    return (norm.cdf(np.asarray(t) / 0.1) - norm.cdf((np.asarray(t) - 0.4) / 0.1)) / 0.4


def need_cdf(t):
    z = np.asarray(t) / 0.1
    z2 = (np.asarray(t) - 0.4) / 0.1
    return np.clip(
        0.25 * (z * norm.cdf(z) + norm.pdf(z) - z2 * norm.cdf(z2) - norm.pdf(z2)), 0, 1
    )


def usage_density(t):
    n, nw = quad(-1, 1.4, 180)
    t = np.asarray(t)
    z = logit(t)[..., None]
    return (norm.pdf(z, 0.3 * n, 0.1) @ (nw * need_density(n))) / (t * (1 - t))


def discount_density(t):
    def qcdf(q):
        q = np.asarray(q)
        bounded = np.clip(4 * q, 1e-15, 1 - 1e-15)
        val = 1 - need_cdf(-logit(bounded))
        return np.where(q <= 0, 0, np.where(q >= 0.25, 1, val))

    return np.maximum(
        (qcdf(t) - qcdf(np.asarray(t) - 0.025)) / 0.025, np.finfo(float).tiny
    )


@lru_cache(maxsize=1)
def ad_density_quadrature():
    m, mw = quad(0.1, 0.9, 1200)
    return m, mw * usage_density(m)


def ad_density(t):
    m, mw = ad_density_quadrature()
    a = np.asarray(t)[..., None]
    # Convolution of Uniform(.95*m,.99*m) and Uniform(0,.5).
    start = a - 0.95 * m
    positive = lambda v: np.maximum(v, 0)
    conditional = (
        positive(start)
        - positive(start - 0.04 * m)
        - positive(start - 0.5)
        + positive(start - 0.5 - 0.04 * m)
    ) / (0.02 * m)
    return np.maximum(np.maximum(conditional, 0) @ mw, np.finfo(float).tiny)


def need_deriv(t, b=3.0):
    s = expit(t)
    return (
        5 + 0.3 * b * sigmoid_mean(0.3 * np.asarray(t), deriv=True) - 0.25 * s * (1 - s)
    )


def calls_deriv(t, b=3.0, blocked=()):
    t = np.asarray(t)
    n = 0.1 * t[..., None] + 0.1 * gx
    v = 5 + (0.3 * b * sigmoid_mean(0.3 * n, deriv=True) if M not in blocked else 0)
    v = v - (0.25 * expit(n) * (1 - expit(n)) if D not in blocked else 0)
    return (
        0.2 + (0.3 if I not in blocked else 0) + 0.1 * (v @ gw)
        if N not in blocked
        else np.full_like(t, 0.2 + (0.3 if I not in blocked else 0))
    )


def derivative(name, t, b=3.0):
    t = np.asarray(t)
    if name == N:
        return need_deriv(t, b)
    if name == C:
        return calls_deriv(t, b)
    if name == L:
        return -8 / (t + 1) ** 2
    return np.full_like(t, {I: 0.3, E: 2, D: 1, M: b, A: 0, B: -0.2, R: 0}[name])


def mean_response(name, t, b=3.0):
    t = np.asarray(t)
    c, cw = quad(0, 4, 48)
    ng = 0.1 * c[:, None] + 0.1 * gx
    mean_n = 0.2
    mean_m = np.sum(cw[:, None] * gw * sigmoid_mean(0.3 * ng)) / 4
    mean_d = 0.25 * (1 - np.sum(cw[:, None] * gw * expit(ng)) / 4) + 0.0125
    mean_c = 2.0
    mean_i = 2.5
    mean_e = 0.75
    mean_l = np.log(11) / 10
    mean_b = 1.0
    if name == C:
        ns = 0.1 * t[..., None] + 0.1 * gx
        mean_n = 0.1 * t
        mean_m = sigmoid_mean(0.3 * ns) @ gw
        mean_d = 0.25 * (1 - expit(ns) @ gw) + 0.0125
        mean_c = t
        mean_i = t + 0.5
    elif name == N:
        mean_n = t
        mean_m = sigmoid_mean(0.3 * t)
        mean_d = 0.25 * (1 - expit(t)) + 0.0125
    elif name == I:
        mean_i = t
    elif name == E:
        mean_e = t
    elif name == L:
        mean_l = 1 / (t + 1)
    elif name == D:
        mean_d = t
    elif name == M:
        mean_m = t
    elif name == B:
        mean_b = t
    return np.broadcast_to(
        5 * mean_n
        + b * mean_m
        + 2 * mean_e
        + mean_d
        - 0.2 * mean_b
        + 0.2 * mean_c
        + 0.3 * mean_i
        + 8 * mean_l,
        t.shape,
    )


def density(name, t):
    if name == N:
        return need_density(t)
    if name in [C, E, L, B]:
        return np.full_like(t, 1 / (DOM[name][1] - DOM[name][0]))
    if name == I:
        return (norm.cdf((t - 0.5) / 0.05) - norm.cdf((t - 4.5) / 0.05)) / 4
    if name == D:
        return discount_density(t)
    if name == M:
        return usage_density(t)
    if name == A:
        return ad_density(t)
    # Reported bugs is oracle-only, with zero gradient. No density needed.
    return np.ones_like(t)


def weighted(g, logf, w, d):
    logw = np.log(w) + 2 * d * logf
    p = np.exp(logw - logsumexp(logw))
    return (
        float(p @ np.abs(g)),
        float(p @ np.maximum(g, 0)),
        float(p @ np.maximum(-g, 0)),
    )


def savefig(name):
    plt.savefig(FIG / (name + ".png"), dpi=175, bbox_inches="tight")
    plt.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close()


def structural_features(x):
    return np.column_stack(
        [x[N], x[M], x[E], x[D], x[B], x[C], x[I], 1 / (x[L] + 1), x[A], x[R]]
    )


def expected_feature_derivative(name, t):
    t = np.asarray(t)
    a = np.zeros((len(t), 10))
    if name == N:
        ds = expit(t) * (1 - expit(t))
        dm = 0.3 * sigmoid_mean(0.3 * t, deriv=True)
        a[:, 0] = 1
        a[:, 1] = dm
        a[:, 3] = -0.25 * ds
        a[:, 8] = 0.97 * dm
        a[:, 9] = ds
    elif name == C:
        n = 0.1 * t[:, None] + 0.1 * gx
        ds = (expit(n) * (1 - expit(n))) @ gw
        dm = 0.03 * (sigmoid_mean(0.3 * n, deriv=True) @ gw)
        a[:, 0] = 0.1
        a[:, 1] = dm
        a[:, 3] = -0.025 * ds
        a[:, 5] = 1
        a[:, 6] = 1
        a[:, 8] = 0.97 * dm
        a[:, 9] = 0.1 * ds
    elif name == L:
        a[:, 7] = -1 / (t + 1) ** 2
        a[:, 8] = -0.05
    elif name == M:
        a[:, 1] = 1
        a[:, 8] = 0.97
    elif name == B:
        c, cw = quad(0, 4, 48)
        n = 0.1 * c[:, None] + 0.1 * gx
        a[:, 4] = 1
        a[:, 9] = np.sum(cw[:, None] * gw * expit(n)) / 4
    else:
        a[:, {I: 6, E: 2, D: 3, A: 8, R: 9}[name]] = 1
    return a


def spline_model(knots=5, alpha=0.01):
    return make_pipeline(
        SplineTransformer(
            n_knots=knots, degree=3, include_bias=False, extrapolation="linear"
        ),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def fit_response(x, y, name, knots=5):
    # Additive conditional mean is correct for this SEM's parent adjustment sets.
    cols = [name] + PARENTS[name]
    model = spline_model(knots)
    model.fit(x[cols].to_numpy(), y)
    return model, cols


def fitted_derivative(model, cols, x, t):
    return coordinate_derivative(model, np.asarray(t), 0)


def coordinate_derivative(model, t, coordinate):
    spline = model[0].bsplines_[coordinate]
    degree = spline.k
    clipped = np.clip(t, spline.t[degree], spline.t[-degree - 1])
    basis = spline.derivative()(clipped)[:, :-1]
    width = basis.shape[1]
    start = coordinate * width
    return basis @ (
        model[2].coef_[start : start + width] / model[1].scale_[start : start + width]
    )


def spline_quadrature(model, cols, x, coordinate=0, nodes=24):
    """Resolve spline knots, linear-extrapolation boundaries and derivative zeros."""
    name = cols[coordinate]
    lo, hi = DOM[name]
    knots = model[0].bsplines_[coordinate].t
    breaks = np.unique(np.r_[lo, knots[(knots > lo) & (knots < hi)], hi])

    def grad(t):
        return coordinate_derivative(model, t, coordinate)

    roots = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        z = np.array([0.2, 0.5, 0.8])
        v = grad(left + (right - left) * z)
        if np.max(np.abs(v)) < 1e-7:
            continue
        coefficients = np.polynomial.polynomial.polyfit(z, v, 2)
        for root in np.polynomial.polynomial.polyroots(coefficients):
            if abs(root.imag) < 1e-6 and 1e-5 < root.real < 1 - 1e-5:
                roots.append(left + (right - left) * root.real)
    breaks = np.unique(np.r_[breaks, roots])
    points = []
    weights = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        t, w = quad(left, right, nodes)
        points.extend(t)
        weights.extend(w)
    return np.array(points), np.array(weights)


def kde_values(name, sample, t, mult=1.0):
    kde = gaussian_kde(sample, bw_method=lambda k: k.scotts_factor() * mult)
    f = kde(t)
    if name in BOUNDS:
        lo, hi = BOUNDS[name]
        f = f + kde(2 * lo - t) + kde(2 * hi - t)
    return f


def oracle():
    rows = []
    alternative = []
    validation = {}
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
    fig2, ax2 = plt.subplots(2, 5, figsize=(17, 6.5))
    for ax, axr, name in zip(axes.flat, ax2.flat, NAMES):
        t, w = quad(*DOM[name], 160)
        g = derivative(name, t)
        lf = np.log(density(name, t))
        curve = []
        for d in np.linspace(0, 3, 61):
            val, pos, neg = weighted(g, lf, w, d)
            curve.append(val)
            rows.append(dict(variable=name, d=d, peace=val, positive=pos, negative=neg))
        ax.plot(np.linspace(0, 3, 61), curve, label="Simulator: usage coefficient 3")
        other = [
            weighted(derivative(name, t, 30), lf, w, d)[0]
            for d in np.linspace(0, 3, 61)
        ]
        alternative.extend(
            dict(variable=name, d=d, peace=v, usage_coefficient=30)
            for d, v in zip(np.linspace(0, 3, 61), other)
        )
        ax.plot(
            np.linspace(0, 3, 61), other, "--", label="Printed equation: coefficient 30"
        )
        if np.ptp(np.r_[curve, other]) < 1e-8:
            center = float(curve[0])
            pad = max(0.05 * abs(center), 0.01)
            ax.set_ylim(center - pad, center + pad)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set(title=name, xlabel="d", ylabel="Mean PEACE (native units)")
        ax.grid(alpha=0.2)
        z = np.linspace(*DOM[name], 250)
        axr.plot(z, mean_response(name, z))
        axr.set(title=name, xlabel="Intervention value", ylabel="E[Y | do(value)]")
        axr.grid(alpha=0.2)
        # The log-density covariance identity is checked by a symmetric difference.
        p = np.exp(np.log(w) + lf - logsumexp(np.log(w) + lf))
        Lmag = np.abs(g)
        covariance = 2 * (p @ (Lmag * lf) - (p @ Lmag) * (p @ lf))
        fd = (weighted(g, lf, w, 0.50001)[0] - weighted(g, lf, w, 0.49999)[0]) / 0.00002
        validation[name] = {"covariance_identity_error": abs(covariance - fd)}
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Oracle total-response PEACE: separate equation specifications")
    fig.tight_layout()
    plt.figure(fig.number)
    savefig("01_oracle_single")
    fig2.suptitle("Oracle total intervention responses — original simulator")
    fig2.tight_layout()
    plt.figure(fig2.number)
    savefig("02_total_responses")
    pd.DataFrame(rows).to_csv(RES / "oracle_single.csv", index=False)
    pd.DataFrame(alternative).to_csv(
        RES / "oracle_single_printed_equation.csv", index=False
    )
    t, w = quad(0, 10, 200)
    g = derivative(L, t)
    ds = np.linspace(0, 20, 201)
    uniform = [weighted(g, np.zeros_like(t), w, d)[0] for d in ds]
    skew = [weighted(g, beta.logpdf(t / 10, 2, 2) - np.log(10), w, d)[0] for d in ds]
    plt.figure(figsize=(8, 4))
    plt.plot(ds, uniform, label="Original Uniform(0,10)")
    plt.plot(ds, skew, label="Sensitivity only: 10 × Beta(2,2)")
    plt.axhline(2 / 9, color="gray", ls=":", label="Beta modal limit: 2/9")
    plt.xlabel("d")
    plt.ylabel("Last upgrade: Mean PEACE")
    plt.legend()
    plt.grid(alpha=0.2)
    savefig("03_upgrade_distribution")
    pd.DataFrame({"d": ds, "uniform": uniform, "beta_2_2": skew}).to_csv(
        RES / "upgrade_distribution.csv", index=False
    )
    validation["uniform_upgrade_error"] = max(abs(np.array(uniform) - 8 / 11))
    validation["need_domain_mass"] = float(
        np.sum(quad(-0.3, 0.7, 250)[1] * need_density(quad(-0.3, 0.7, 250)[0]))
    )
    (RES / "validation_oracle.json").write_text(json.dumps(validation, indent=2))


PAIRS = [(N, M), (C, N), (N, D), (E, L), (M, L), (A, R)]


def pair_grid(pair, k=100, b=3.0, grids=None):
    if grids is None:
        u, wu = quad(*DOM[pair[0]], k)
        v, wv = quad(*DOM[pair[1]], k)
    else:
        (u, wu), (v, wv) = grids
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wu, wv)
    if pair == (N, M):
        gn = 5 - 0.25 * expit(U) * (1 - expit(U))
        gv = np.full_like(V, b)
        lf = (
            np.log(need_density(U))
            + norm.logpdf(logit(V), 0.3 * U, 0.1)
            - np.log(V * (1 - V))
        )
    elif pair == (C, N):
        gn = np.full_like(U, 0.5)
        gv = need_deriv(V, b)
        lf = -np.log(4) + norm.logpdf(V, 0.1 * U, 0.1)
    elif pair == (N, D):
        # Integrate over actual curved support: discount = lower(n)+.025*s.
        v, wv = quad(0, 1, k)
        U, S = np.meshgrid(u, v, indexing="ij")
        V = 0.25 * (1 - expit(U)) + 0.025 * S
        W = np.outer(wu, wv) * 0.025
        gn = 5 + 0.3 * b * sigmoid_mean(0.3 * U, deriv=True)
        gv = np.ones_like(U)
        lf = np.log(need_density(U)) - np.log(0.025)
    elif pair == (E, L):
        gn = np.full_like(U, 2)
        gv = -8 / (V + 1) ** 2
        lf = np.zeros_like(U)
    elif pair == (M, L):
        # Independence and constant usage derivative make its marginal density
        # cancel. This is a reduced integral, not a uniform-usage assertion.
        gn = np.full_like(U, b)
        gv = -8 / (V + 1) ** 2
        lf = np.zeros_like(U)
    elif pair == (A, R):
        gn = np.zeros_like(U)
        gv = np.zeros_like(U)
        lf = np.zeros_like(U)
    return U, V, W, lf, gn, gv


def pairs():
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, pair in zip(axes.flat, PAIRS):
        U, V, W, lf, g1, g2 = pair_grid(pair)
        for b, style in [(3, "-"), (30, "--")]:
            _, _, _, lf, g1, g2 = pair_grid(pair, b=b)
            g = np.hypot(g1, g2).ravel()
            ds = np.linspace(0, 3, 61)
            vals = []
            for d in ds:
                val = weighted(g, lf.ravel(), W.ravel(), d)[0]
                vals.append(val)
                rows.append(
                    {"pair": " + ".join(pair), "coefficient": b, "d": d, "peace": val}
                )
            ax.plot(ds, vals, style, label=f"Usage coefficient {b}")
        ax.ticklabel_format(useOffset=False, style="plain")
        if pair in [(E, L), (A, R)]:
            center = float(vals[0])
            pad = max(0.05 * abs(center), 0.01)
            ax.set_ylim(center - pad, center + pad)
        ax.set(title=" + ".join(pair), xlabel="d", ylabel="Joint Mean PEACE")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Joint interventions: gradient norm, not interaction strength")
    fig.tight_layout()
    savefig("04_joint_curves")
    pd.DataFrame(rows).to_csv(RES / "oracle_pairs.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, pair in zip(axes, PAIRS[:2] + [(E, L)]):
        U, V, W, lf, g1, g2 = pair_grid(pair, 75)
        im = ax.pcolormesh(U, V, np.hypot(g1, g2), shading="auto")
        fig.colorbar(im, ax=ax, label="Local gradient norm")
        ax.set(xlabel=pair[0], ylabel=pair[1])
    fig.tight_layout()
    savefig("05_local_joint_magnitudes")


def observational():
    rows = []
    overlap = []
    eligible = [name for name in NAMES if name != R]
    for scenario, n, reps, noise, rho, gamma in [
        ("original", 5000, 40, 0, 0, 0),
        ("outcome_noise", 1000, 80, 0.5, 0, 0),
        ("outcome_noise", 5000, 80, 0.5, 0, 0),
        ("hidden_confounding", 5000, 40, 0.5, 0.5, 0.1),
    ]:
        for rep in range(reps):
            x, y = generate(
                n,
                seed=61000 + rep + int(noise * 100),
                noise=noise,
                rho=rho,
                gamma=gamma,
            )
            coeff = LinearRegression().fit(structural_features(x), y).coef_
            names = eligible if scenario != "hidden_confounding" else [N, L, A]
            for name in names:
                fitted, cols = fit_response(x, y, name)
                t, w = spline_quadrature(fitted, cols, x)
                lf = np.log(density(name, t))
                trueg = derivative(name, t)
                g = fitted_derivative(fitted, cols, x, t)
                fhat = kde_values(name, x[name].to_numpy(), t)
                lhat = np.log(fhat)
                for method, grad, logf in [
                    ("Parent-adjusted additive spline + estimated density", g, lhat),
                    ("Parent-adjusted additive spline + oracle weight", g, lf),
                    (
                        "Fitted outcome coefficients + known SEM propagation",
                        expected_feature_derivative(name, t) @ coeff,
                        lf,
                    ),
                ]:
                    for d in [0, 0.5, 1, 2]:
                        est, pos, neg = weighted(grad, logf, w, d)
                        truth, tp, tn = weighted(trueg, lf, w, d)
                        rows.append(
                            dict(
                                scenario=scenario,
                                n=n,
                                rep=rep,
                                variable=name,
                                method=method,
                                d=d,
                                estimate=est,
                                truth=truth,
                                positive=pos,
                                negative=neg,
                                true_positive=tp,
                                true_negative=tn,
                            )
                        )
            if rep == 0:
                for name in NAMES:
                    if name == R:
                        variance = 0.0
                    elif name == I:
                        variance = 0.05**2
                    elif name == N:
                        variance = 0.1**2
                    elif name == D:
                        variance = 0.025**2 / 12
                    elif name == M:
                        variance = float(
                            np.mean((x[M] - sigmoid_mean(0.3 * x[N].to_numpy())) ** 2)
                        )
                    elif name == A:
                        variance = float(np.mean((x[M] * 0.04) ** 2 / 12))
                    else:
                        variance = float(np.var(x[name]))
                    overlap.append(
                        dict(
                            scenario=scenario,
                            variable=name,
                            conditional_variance=variance,
                            marginal_variance=float(np.var(x[name])),
                            variance_ratio=variance / np.var(x[name]),
                        )
                    )
        print("completed observational", scenario, n, reps, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "observational_replicates.csv", index=False)
    df["error"] = df.estimate - df.truth
    summary = (
        df.groupby(["scenario", "n", "variable", "method", "d"], sort=False)
        .agg(
            truth=("truth", "first"),
            bias=("error", "mean"),
            rmse=("error", lambda a: np.sqrt(np.mean(a * a))),
            estimate_sd=("estimate", "std"),
            mean_estimate=("estimate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RES / "observational_summary.csv", index=False)
    pd.DataFrame(overlap).to_csv(RES / "overlap.csv", index=False)
    subset = summary[
        (summary.scenario == "outcome_noise")
        & (summary.d == 0.5)
        & summary.method.str.startswith("Parent-adjusted additive spline + estimated")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True)
    for ax, n in zip(axes, [1000, 5000]):
        z = subset[subset.n == n].set_index("variable").loc[eligible]
        ax.barh(eligible, z.rmse)
        ax.set(
            title=f"Outcome-noise sensitivity: n={n}, 80 replicates",
            xlabel="RMSE of Mean PEACE at d=0.5",
        )
    fig.tight_layout()
    savefig("06_observational_rmse")
    # Dense estimated curves on a fixed independent illustration sample.
    x, y = generate(5000, 99001, noise=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dense = []
    for ax, name in zip(axes, [N, L, A]):
        m, cols = fit_response(x, y, name)
        t, w = spline_quadrature(m, cols, x)
        g = fitted_derivative(m, cols, x, t)
        for mult in [0.5, 1, 2]:
            lf = np.log(kde_values(name, x[name].to_numpy(), t, mult))
            values = [weighted(g, lf, w, d)[0] for d in np.linspace(0, 3, 61)]
            ax.plot(np.linspace(0, 3, 61), values, label=f"{mult:g} × Scott bandwidth")
            dense.extend(
                dict(variable=name, bandwidth=mult, d=d, estimate=v)
                for d, v in zip(np.linspace(0, 3, 61), values)
            )
        ax.plot(
            np.linspace(0, 3, 61),
            [
                weighted(derivative(name, t), np.log(density(name, t)), w, d)[0]
                for d in np.linspace(0, 3, 61)
            ],
            "k--",
            label="Oracle",
        )
        ax.set(title=name, xlabel="d", ylabel="Mean PEACE")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    savefig("07_estimated_curves_bandwidth")
    pd.DataFrame(dense).to_csv(RES / "illustrative_curves.csv", index=False)


def projection_truth():
    c, cw = quad(0, 4, 160)
    response = mean_response(C, c)
    calls = np.sum(cw * (c - 2) * response) / 4 / (4 / 3)
    n = 0.1 * c[:, None] + 0.1 * gx
    need = np.sum(cw[:, None] * gw * need_deriv(n)) / 4
    upgrade = (8 - 4.8 * np.log(11)) / (100 / 12)
    return {C: calls, I: 0.3, E: 2, L: upgrade, N: need, D: 1, M: 3, A: 0, B: -0.2}


def comparators(reps=30):
    from econml.dml import CausalForestDML, LinearDML

    rows = []
    truth = projection_truth()
    for scenario, noise, rho, gamma in [
        ("original", 0, 0, 0),
        ("outcome_noise", 0.5, 0, 0),
        ("hidden_confounding", 0.5, 0.5, 0.1),
    ]:
        names = list(truth) if scenario != "hidden_confounding" else [N, L, A]
        for rep in range(reps):
            x, y = generate(3000, seed=81000 + rep, noise=noise, rho=rho, gamma=gamma)
            for name in names:
                parents = PARENTS[name]
                z = x[parents].to_numpy() if parents else np.zeros((len(x), 1))
                my = spline_model() if parents else DummyRegressor()
                mt = (
                    LinearRegression()
                    if name in [N, I, A]
                    else (spline_model() if parents else DummyRegressor())
                )
                cf = CausalForestDML(
                    model_y=my,
                    model_t=mt,
                    n_estimators=120,
                    min_samples_leaf=30,
                    max_depth=8,
                    cv=3,
                    random_state=rep,
                    n_jobs=1,
                    inference=False,
                )
                dm = LinearDML(model_y=my, model_t=mt, cv=3, random_state=rep)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cf.fit(y, x[name].to_numpy(), X=z)
                    dm.fit(y, x[name].to_numpy(), W=z if parents else None)
                # Average over a fixed population reference, not each fitted sample.
                if parents:
                    ref, _ = generate(4096, seed=19001)
                    zref = ref[parents].to_numpy()
                else:
                    zref = np.zeros((1, 1))
                for method, estimate in [
                    ("Causal forest", np.mean(cf.const_marginal_effect(zref))),
                    (
                        "Cross-fitted linear DML",
                        np.ravel(dm.const_marginal_effect())[0],
                    ),
                ]:
                    rows.append(
                        dict(
                            scenario=scenario,
                            n=3000,
                            rep=rep,
                            variable=name,
                            method=method,
                            estimate=float(estimate),
                            truth=truth[name],
                        )
                    )
            if (rep + 1) % 10 == 0:
                print("completed comparators", scenario, rep + 1, flush=True)
    df = pd.DataFrame(rows)
    df["error"] = df.estimate - df.truth
    df.to_csv(RES / "comparator_replicates.csv", index=False)
    s = (
        df.groupby(["scenario", "variable", "method"])
        .agg(
            truth=("truth", "first"),
            mean_estimate=("estimate", "mean"),
            bias=("error", "mean"),
            rmse=("error", lambda a: np.sqrt(np.mean(a * a))),
            estimate_sd=("estimate", "std"),
        )
        .reset_index()
    )
    s.to_csv(RES / "comparator_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, method in zip(axes, ["Causal forest", "Cross-fitted linear DML"]):
        a = (
            s[(s.scenario == "original") & (s.method == method)]
            .set_index("variable")
            .loc[list(truth)]
        )
        ax.errorbar(
            a.mean_estimate,
            np.arange(len(a)),
            xerr=a.estimate_sd,
            fmt="o",
            label="Mean ± replicate SD",
        )
        ax.scatter(
            a.truth,
            np.arange(len(a)),
            marker="x",
            color="black",
            label="Own projection truth",
        )
        ax.set_yticks(np.arange(len(a)), list(truth))
        ax.set(title=method, xlabel="Signed projection slope")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig("08_comparator_targets")
    (RES / "projection_truth.json").write_text(json.dumps(truth, indent=2))


def prediction():
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split

    x, y = generate(10000, seed=17001)
    train, rest = train_test_split(np.arange(len(x)), test_size=0.4, random_state=42)
    valid, test = train_test_split(rest, test_size=0.5, random_state=43)
    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=1,
        n_jobs=1,
        random_state=42,
        early_stopping_rounds=40,
    )
    model.fit(
        x.iloc[train], y[train], eval_set=[(x.iloc[valid], y[valid])], verbose=False
    )
    pred = model.predict(x.iloc[test])
    attributions = shap.TreeExplainer(model)(x.iloc[test])
    means = np.mean(np.abs(attributions.values), axis=0)
    scores = dict(
        test_r2=float(r2_score(y[test], pred)),
        test_rmse=float(np.sqrt(np.mean((y[test] - pred) ** 2))),
        best_iteration=int(model.best_iteration),
        train_n=len(train),
        validation_n=len(valid),
        test_n=len(test),
        shap_semantics="Tree-path-dependent predictive attribution on held-out test observations",
    )
    out = pd.DataFrame({"variable": NAMES, "mean_absolute_shap": means})
    # Explicit population intervention contrasts: same outcome units as SHAP,
    # but distinct estimands. Reference quantiles are fixed before prediction.
    ref, _ = generate(200000, seed=17002)
    qs = []
    for name in NAMES:
        lo, hi = np.quantile(ref[name], [0.25, 0.75])
        contrast = float(np.diff(mean_response(name, np.array([lo, hi])))[0])
        qs.append((lo, hi, contrast))
    out[["population_q25", "population_q75", "causal_interquartile_contrast"]] = qs
    out.to_csv(RES / "prediction_and_causal_targets.csv", index=False)
    (RES / "prediction_metrics.json").write_text(json.dumps(scores, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(NAMES, means)
    axes[0].set(
        title=f"Predictive SHAP: held-out R²={scores['test_r2']:.4f}",
        xlabel="Mean absolute predictive attribution (Y units)",
    )
    axes[1].barh(NAMES, out.causal_interquartile_contrast)
    axes[1].set(
        title="Known SEM: do(Q75) − do(Q25)",
        xlabel="Signed total intervention contrast (Y units)",
    )
    fig.suptitle("Different questions: predictive attribution and causal intervention")
    fig.tight_layout()
    savefig("09_prediction_vs_intervention")


def sensitivity_plot():
    obs = pd.read_csv(RES / "observational_summary.csv")
    comp = pd.read_csv(RES / "comparator_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, [N, L, A]):
        vals = []
        labels = []
        for scenario in ["outcome_noise", "hidden_confounding"]:
            z = obs[
                (obs.scenario == scenario)
                & (obs.n == 5000)
                & (obs.variable == name)
                & (obs.d == 0.5)
                & obs.method.str.startswith("Parent-adjusted additive spline + oracle")
            ].iloc[0]
            vals.append(z.bias)
            labels.append("PEACE\n" + scenario.replace("_", " "))
            for method in ["Causal forest", "Cross-fitted linear DML"]:
                z = comp[
                    (comp.scenario == scenario)
                    & (comp.variable == name)
                    & (comp.method == method)
                ].iloc[0]
                vals.append(z.bias)
                labels.append(
                    ("CF" if method == "Causal forest" else "DML")
                    + "\n"
                    + scenario.replace("_", " ")
                )
        ax.bar(np.arange(6), vals)
        ax.set_xticks(np.arange(6), labels, rotation=65, ha="right", fontsize=7)
        ax.axhline(0, color="black", lw=0.7)
        ax.set(title=name, ylabel="Bias against each method’s own target")
    fig.suptitle(
        "Sensitivity to unobserved need–outcome confounding; different sample sizes"
    )
    fig.tight_layout()
    savefig("10_identification_sensitivity")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "part",
        choices=[
            "oracle",
            "observational",
            "comparators",
            "prediction",
            "sensitivity",
            "all",
        ],
    )
    args = p.parse_args()
    for name, fn in [
        ("oracle", lambda: (oracle(), pairs())),
        ("observational", observational),
        ("comparators", comparators),
        ("prediction", prediction),
        ("sensitivity", sensitivity_plot),
    ]:
        if args.part in [name, "all"]:
            fn()
