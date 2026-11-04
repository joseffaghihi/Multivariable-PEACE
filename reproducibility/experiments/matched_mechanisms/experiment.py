"""Matched scalar summaries, distinct probability-weighted causal geometry."""

import os
import tempfile

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[key] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "peace-matched-matplotlib")
)
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.special import logsumexp
from numpy.polynomial import Chebyshev as C
from numpy.polynomial.legendre import leggauss
import json, hashlib, argparse

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
RHO = np.array([0.5, 1, 1.5])
PI = np.array([0.9, 0.1])
WIDTH = 1 / 3
NOISE = 0.02
FREQ = np.arange(1, 21)
K = {"A": np.array([[3, 3, 1], [16, 4, 8]]), "B": np.array([[4, 1, 2], [12, 12, 4]])}
GRID = [(d, r) for d in [0, 0.25, 0.5, 1, 2] for r in [0, 0.5, 1, 2]]


def response(t, z, model, deriv=False):
    t = np.asarray(t)
    z = np.broadcast_to(np.asarray(z, dtype=int), t.shape)
    j = np.minimum((3 * t).astype(int), 2)
    u = 3 * t - j
    k = K[model][z, j]
    if deriv:
        return 3 * np.pi * k * np.sin(2 * np.pi * k * u)
    return np.sin(np.pi * k * u) ** 2


def score(tv, d, r, rho=RHO, pi=PI):
    p = rho ** (2 * d)
    q = pi**r
    q /= q.sum()
    return float(q @ (tv @ p) / (WIDTH * p.sum()))


def squared_score(energy, d, r, rho=RHO, pi=PI):
    return score(energy, d, r, rho, pi)


def oracle_arrays(model):
    k = K[model]
    return 2 * k, np.pi**2 * k * k / (2 * WIDTH)


def basis(u):
    return np.sin(np.pi * np.asarray(u)[:, None] * FREQ) ** 2


def fitted_tv(beta):
    # sin(k theta)=sin(theta) U_(k-1)(cos(theta)). Extrema are roots of this polynomial.
    U0 = C([1])
    U1 = C([0, 2])
    poly = FREQ[0] * beta[0] * U0
    if len(beta) > 1:
        poly = poly + FREQ[1] * beta[1] * U1
    for k in range(2, len(beta)):
        U2 = C([0, 2]) * U1 - U0
        poly = poly + FREQ[k] * beta[k] * U2
        U0, U1 = U1, U2
    roots = poly.roots()
    real = roots.real[
        (np.abs(roots.imag) < 1e-7) & (roots.real > -1) & (roots.real < 1)
    ]
    angles = np.arccos(real) / (2 * np.pi)
    points = np.unique(np.r_[0, 0.5, 1, angles, 1 - angles])
    values = basis(points) @ beta
    return float(np.abs(np.diff(values)).sum())


def fit_arrays(t, z, y):
    j = np.minimum((3 * t).astype(int), 2)
    tv = np.zeros((2, 3, 2))
    en = np.zeros_like(tv)
    for profile in [0, 1]:
        for cell in range(3):
            mask = (z == profile) & (j == cell)
            u = 3 * t[mask] - cell
            X = basis(u)
            assert len(u) > len(FREQ) + 5
            beta = lstsq(X, y[mask], lapack_driver="gelsy")[0]
            for m in range(2):
                tv[profile, cell, m] = fitted_tv(beta[:, m])
                en[profile, cell, m] = (
                    np.pi**2 / (2 * WIDTH) * np.sum(FREQ**2 * beta[:, m] ** 2)
                )
    return tv, en


def sample_t(n, rng):
    j = rng.choice(3, n, p=RHO / 3)
    return (j + rng.random(n)) / 3


def info_value(nodes=600):
    x, w = leggauss(nodes)
    theta = (x + 1) * np.pi / 2
    phase = np.sin(theta) ** 2
    phasew = w / 2
    yy = (x + 1) * (1 + 16 * NOISE) / 2 - 8 * NOISE
    wy = w * (1 + 16 * NOISE) / 2
    normal = np.exp(-0.5 * ((yy[:, None] - phase) / NOISE) ** 2) / (
        np.sqrt(2 * np.pi) * NOISE
    )
    density = normal @ phasew
    entropy = -float(wy @ (density * np.log(np.maximum(density, 1e-300))))
    noise_entropy = 0.5 * np.log(2 * np.pi * np.e * NOISE**2)
    return float(entropy - noise_entropy), float(wy @ density)


def oracle():
    rows = []
    for model in K:
        tv, en = oracle_arrays(model)
        for d, r in GRID:
            rows.append(
                dict(
                    model=model,
                    d=d,
                    r=r,
                    peace=score(tv, d, r),
                    positive=score(tv, d, r) / 2,
                    negative=score(tv, d, r) / 2,
                    signed_ade=0.0,
                    weighted_squared_derivative=squared_score(en, d, r),
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "oracle_scores.csv", index=False)
    baseline = []
    mi, mass = info_value()
    for model in K:
        tv, en = oracle_arrays(model)
        baseline.append(
            dict(
                model=model,
                endpoint_contrast=0.0,
                signed_ADE=0.0,
                uniform_TV_natural_profiles=score(tv, 0, 1),
                natural_absolute_derivative=score(tv, 0.5, 1),
                natural_squared_derivative=score(en, 0.5, 1),
                output_mean=0.5,
                output_variance=0.125 + NOISE**2,
                T_first_sobol=0.1025 / (0.125 + NOISE**2),
                T_total_sobol=0.125 / (0.125 + NOISE**2),
                Z_first_sobol=0.0,
                Z_total_sobol=0.0225 / (0.125 + NOISE**2),
                Janzing_T_edge_strength=mi,
            )
        )
    pd.DataFrame(baseline).to_csv(RES / "matched_baselines.csv", index=False)
    dense = []
    for d in np.linspace(0, 2, 101):
        for r in np.linspace(0, 2, 101):
            a, e = oracle_arrays("A")
            b, f = oracle_arrays("B")
            dense.append(
                dict(
                    d=d,
                    r=r,
                    A=score(a, d, r),
                    B=score(b, d, r),
                    difference=score(a, d, r) - score(b, d, r),
                    squared_difference=score(e, d, r) - score(f, d, r),
                )
            )
    pd.DataFrame(dense).to_csv(RES / "oracle_surfaces.csv", index=False)
    return df


def finite_steps():
    rows = []
    n = 262144
    for delta in [1 / 50, 1 / 200, 1 / 800, 1 / 3200]:
        t = delta / 2 + (np.arange(n) + 0.5) / n * (1 - delta)
        j = np.minimum((3 * t).astype(int), 2)
        for model in K:
            for d, r in [(1, 0), (1, 1), (0.25, 0), (0.25, 1)]:
                wt = RHO[j] ** (2 * d)
                wt = wt / wt.sum()
                q = PI**r
                q = q / q.sum()
                value = 0
                for profile in [0, 1]:
                    g = (
                        np.abs(
                            response(t + delta / 2, profile, model)
                            - response(t - delta / 2, profile, model)
                        )
                        / delta
                    )
                    value += q[profile] * (wt @ g)
                truth = score(oracle_arrays(model)[0], d, r)
                rows.append(
                    dict(
                        model=model,
                        d=d,
                        r=r,
                        step=delta,
                        local_contrast_average=value,
                        peace_limit=truth,
                        error=value - truth,
                    )
                )
    pd.DataFrame(rows).to_csv(RES / "finite_intervention_steps.csv", index=False)


def validate():
    a = np.array([4, 1, 2])
    b = np.array([3, 3, 1])
    assert a.sum() == b.sum() and RHO @ a == RHO @ b and RHO @ (a * a) == RHO @ (b * b)
    df = oracle()
    mi, mass = info_value(600)
    mi2, mass2 = info_value(1000)
    checks = {}
    checks["MI_quadrature_difference"] = abs(mi - mi2)
    checks["output_density_mass_error"] = abs(mass - 1)
    probe = np.linspace(0.001, 0.999, 901)
    eps = 1e-7
    checks["response_derivative_error"] = max(
        float(
            np.max(
                np.abs(
                    (response(probe + eps, z, m) - response(probe - eps, z, m))
                    / (2 * eps)
                    - response(probe, z, m, True)
                )
            )
        )
        for z in [0, 1]
        for m in K
    )
    errors = []
    rng = np.random.default_rng(552)
    for k in [1, 2, 3, 4, 8, 12, 16, 20]:
        beta = np.zeros(20)
        beta[k - 1] = 1
        errors.append(abs(fitted_tv(beta) - 2 * k))
    beta = rng.normal(0, 0.01, 20)
    beta[2] += 1
    exact = fitted_tv(beta)
    u = np.linspace(0, 1, 65537)
    dense = float(np.abs(np.diff(basis(u) @ beta)).sum())
    checks["spectral_TV_root_error"] = max(errors)
    checks["mixed_spectral_TV_vs_dense_error"] = abs(exact - dense)
    # A measure-preserving phase coupling verifies equal response laws without relying on histograms.
    phase = rng.random(10000)
    target = np.sin(np.pi * phase) ** 2
    lawerr = 0
    for k in np.unique(np.r_[K["A"].ravel(), K["B"].ravel()]):
        cycles = rng.integers(0, k, len(phase))
        u = (cycles + phase) / k
        lawerr = max(lawerr, float(np.max(np.abs(np.sin(np.pi * k * u) ** 2 - target))))
    checks["output_law_coupling_error"] = lawerr

    def val(m, d, r):
        return float(df[(df.model == m) & (df.d == d) & (df.r == r)].peace.iloc[0])

    checks["known_fraction_errors"] = [
        abs(val("A", 1, 1) - 96 / 7),
        abs(val("B", 1, 1) - 99 / 7),
        abs(val("A", 1, 0) - 192 / 7),
        abs(val("B", 1, 0) - 183 / 7),
    ]
    rcross = np.log(4) / np.log(9)
    checks["profile_crossover"] = rcross
    checks["crossover_error"] = abs(
        score(oracle_arrays("A")[0], 1, rcross)
        - score(oracle_arrays("B")[0], 1, rcross)
    )
    checks["passed"] = bool(
        checks["MI_quadrature_difference"] < 1e-6
        and checks["output_density_mass_error"] < 1e-8
        and checks["response_derivative_error"] < 1e-5
        and checks["spectral_TV_root_error"] < 1e-8
        and checks["mixed_spectral_TV_vs_dense_error"] < 1e-4
        and lawerr < 1e-10
        and max(checks["known_fraction_errors"]) < 1e-10
        and checks["crossover_error"] < 1e-10
    )
    (RES / "validation.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
    assert checks["passed"]
    finite_steps()


def run(reps=40):
    rows = []
    baselines = []
    for n in [2000, 8000, 32000]:
        for rep in range(reps):
            rng = np.random.default_rng(640000 + rep + n)
            t = sample_t(n, rng)
            z = np.r_[np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)]
            rng.shuffle(z)
            y = np.column_stack([response(t, z, m) for m in K]) + rng.normal(
                0, NOISE, (n, 2)
            )
            tv, en = fit_arrays(t, z, y)
            ref_t = sample_t(20000, rng)
            ref_z = rng.choice(2, 20000, p=PI)
            rho = (
                (np.bincount(np.minimum((3 * ref_t).astype(int), 2), minlength=3) + 0.5)
                / (20000 + 1.5)
                * 3
            )
            pi = (np.bincount(ref_z, minlength=2) + 0.5) / (20000 + 1.0)
            for index, model in enumerate(K):
                truth_tv, truth_en = oracle_arrays(model)
                for d, r in GRID:
                    estimate = score(tv[:, :, index], d, r, rho, pi)
                    oracle_value = score(truth_tv, d, r)
                    rows.append(
                        dict(
                            n=n,
                            rep=rep,
                            model=model,
                            d=d,
                            r=r,
                            estimate=estimate,
                            truth=oracle_value,
                            squared_estimate=score(en[:, :, index], d, r, rho, pi),
                            squared_truth=score(truth_en, d, r),
                        )
                    )
                baselines.append(
                    dict(
                        n=n,
                        rep=rep,
                        model=model,
                        natural_absolute_derivative=score(
                            tv[:, :, index], 0.5, 1, rho, pi
                        ),
                        natural_squared_derivative=score(
                            en[:, :, index], 0.5, 1, rho, pi
                        ),
                        uniform_TV=score(tv[:, :, index], 0, 1, rho, pi),
                    )
                )
            if (rep + 1) % 10 == 0:
                print(
                    "completed matched-mechanism calibration",
                    n,
                    rep + 1,
                    "/",
                    reps,
                    flush=True,
                )
        pd.DataFrame(rows).to_csv(RES / "calibration_results.csv", index=False)
        pd.DataFrame(baselines).to_csv(RES / "calibration_baselines.csv", index=False)
    print("all matched-mechanism calibration cases complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=["validate", "run"])
    p.add_argument("--reps", type=int, default=40)
    args = p.parse_args()
    if args.part == "validate":
        validate()
    else:
        run(args.reps)
