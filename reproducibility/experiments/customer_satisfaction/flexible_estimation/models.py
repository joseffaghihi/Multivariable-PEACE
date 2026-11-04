"""Generic raw-coordinate regressions; no simulator features or coefficients."""

import os
import tempfile

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[key] = "1"
import numpy as np
from sklearn.preprocessing import SplineTransformer


class SmoothRegression:
    def __init__(self, kind="additive", knots=12, seed=1701):
        self.kind = kind
        self.knots = knots
        self.seed = seed

    def features(self, z):
        if self.kind == "linear":
            return np.zeros((len(z), 0))
        pieces = [self.spline.transform(z)]
        if self.kind == "interaction":
            pieces.append(np.column_stack([z[:, j] * z[:, k] for j, k in self.pairs]))
            pieces.append(np.sqrt(2 / self.m) * np.cos(z @ self.omega + self.phase))
        return np.column_stack(pieces)

    def fit(self, x, y):
        x = np.asarray(x)
        self.mean = x.mean(0)
        self.sd = x.std(0)
        assert np.all(self.sd > 1e-10)
        z = (x - self.mean) / self.sd
        self.p = z.shape[1]
        linear = np.column_stack([np.ones(len(x)), z])
        base = np.linalg.lstsq(linear, y, rcond=None)[0]
        resid = y - linear @ base
        if self.kind == "linear":
            self.beta = base
            self.coef = np.zeros(0)
            self.gcv = np.mean(resid**2) / (1 - linear.shape[1] / len(x)) ** 2
            self.df = float(linear.shape[1])
            self.penalty = 0
            return self
        self.spline = SplineTransformer(
            n_knots=self.knots, degree=3, include_bias=False, extrapolation="linear"
        ).fit(z)
        self.width = self.spline.bsplines_[0].c.shape[0] - 1
        if self.kind == "interaction":
            self.pairs = [(j, k) for j in range(self.p) for k in range(j + 1, self.p)]
            rng = np.random.default_rng(self.seed)
            self.m = 96
            self.omega = rng.normal(size=(self.p, self.m)) * np.sqrt(2 / self.p)
            self.phase = rng.uniform(0, 2 * np.pi, self.m)
        F = self.features(z)
        projection = np.linalg.lstsq(linear, F, rcond=None)[0]
        U = F - linear @ projection
        scale = np.sqrt(np.mean(U**2, axis=0))
        scale = np.maximum(scale, 1e-8)
        U = U / scale
        gram = U.T @ U
        rhs = U.T @ resid
        eig, V = np.linalg.eigh(gram)
        eig = np.maximum(eig, 0)
        a = V.T @ rhs
        best = None
        for ratio in np.logspace(-7, 1, 9):
            penalty = len(x) * ratio
            c = V @ (a / (eig + penalty))
            df = linear.shape[1] + np.sum(eig / (eig + penalty))
            if df >= len(x) - 2:
                continue
            rss = max(float(resid @ resid - 2 * c @ rhs + c @ gram @ c), 0.0)
            gcv = (rss / len(x)) / (1 - df / len(x)) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, c, df, penalty)
        assert best is not None
        self.gcv, c, self.df, self.penalty = best
        self.coef = c / scale
        self.beta = base - projection @ self.coef
        return self

    def predict(self, x):
        z = (np.asarray(x) - self.mean) / self.sd
        return self.beta[0] + z @ self.beta[1:] + self.features(z) @ self.coef

    def spline_derivative(self, z, j):
        s = self.spline.bsplines_[j]
        clipped = np.clip(z, s.t[s.k], s.t[-s.k - 1])
        return s.derivative()(clipped)[:, :-1] / self.sd[j]

    def derivative(self, x, j):
        x = np.asarray(x)
        z = (x - self.mean) / self.sd
        out = np.full(len(x), self.beta[j + 1] / self.sd[j])
        if self.kind == "linear":
            return out
        out += (
            self.spline_derivative(z[:, j], j)
            @ self.coef[j * self.width : (j + 1) * self.width]
        )
        if self.kind == "interaction":
            start = self.p * self.width
            for k, (a, b) in enumerate(self.pairs):
                if j == a:
                    out += self.coef[start + k] * z[:, b] / self.sd[j]
                if j == b:
                    out += self.coef[start + k] * z[:, a] / self.sd[j]
            c = self.coef[start + len(self.pairs) :]
            out -= (
                np.sqrt(2 / self.m)
                * (np.sin(z @ self.omega + self.phase) @ (c * self.omega[j]))
                / self.sd[j]
            )
        return out

    def prepare_reference(self, reference, j):
        z = (np.asarray(reference) - self.mean) / self.sd
        result = {"mean_z": z.mean(0), "j": j}
        if self.kind == "interaction":
            angle = z @ self.omega + self.phase - z[:, j, None] * self.omega[j]
            result["characteristic"] = np.mean(np.exp(1j * angle), axis=0)
        return result

    def standardized_derivative(self, t, reference):
        j = reference["j"]
        z = (np.asarray(t) - self.mean[j]) / self.sd[j]
        out = np.full(len(z), self.beta[j + 1] / self.sd[j])
        if self.kind == "linear":
            return out
        out += (
            self.spline_derivative(z, j)
            @ self.coef[j * self.width : (j + 1) * self.width]
        )
        if self.kind == "interaction":
            start = self.p * self.width
            for k, (a, b) in enumerate(self.pairs):
                if j == a:
                    out += self.coef[start + k] * reference["mean_z"][b] / self.sd[j]
                if j == b:
                    out += self.coef[start + k] * reference["mean_z"][a] / self.sd[j]
            c = self.coef[start + len(self.pairs) :]
            sine = np.imag(
                np.exp(1j * np.outer(z, self.omega[j])) * reference["characteristic"]
            )
            out -= np.sqrt(2 / self.m) * (sine @ (c * self.omega[j])) / self.sd[j]
        return out

    def integration_grid(self, lo, hi, j, nodes=16):
        from numpy.polynomial.legendre import leggauss

        base = [lo, hi]
        if self.kind != "linear":
            knots = self.spline.bsplines_[j].t * self.sd[j] + self.mean[j]
            base.extend(knots[(knots > lo) & (knots < hi)])
        breaks = np.unique(base)
        g, w = leggauss(nodes)
        ts = []
        ws = []
        for a, b in zip(breaks[:-1], breaks[1:]):
            ts.extend((a + b) / 2 + (b - a) / 2 * g)
            ws.extend((b - a) / 2 * w)
        return np.array(ts), np.array(ws)

    def joint_reference(self, reference, indices):
        z = (np.asarray(reference) - self.mean) / self.sd
        out = {"mean_z": z.mean(0), "indices": indices}
        if self.kind == "interaction":
            angle = z @ self.omega + self.phase
            for j in indices:
                angle = angle - z[:, j, None] * self.omega[j]
            out["characteristic"] = np.mean(np.exp(1j * angle), axis=0)
        return out

    def joint_standardized_derivatives(self, values, reference):
        values = np.asarray(values)
        indices = reference["indices"]
        zs = (values - self.mean[indices]) / self.sd[indices]
        result = []
        if self.kind == "interaction":
            phase = sum(
                np.outer(zs[:, k], self.omega[j]) for k, j in enumerate(indices)
            )
            sine = np.imag(np.exp(1j * phase) * reference["characteristic"])
        for col, j in enumerate(indices):
            g = np.full(len(values), self.beta[j + 1] / self.sd[j])
            if self.kind != "linear":
                g += (
                    self.spline_derivative(zs[:, col], j)
                    @ self.coef[j * self.width : (j + 1) * self.width]
                )
            if self.kind == "interaction":
                start = self.p * self.width
                for k, (a, b) in enumerate(self.pairs):
                    other = b if j == a else (a if j == b else None)
                    if other is not None:
                        other_z = (
                            zs[:, indices.index(other)]
                            if other in indices
                            else reference["mean_z"][other]
                        )
                        g += self.coef[start + k] * other_z / self.sd[j]
                c = self.coef[start + len(self.pairs) :]
                g -= np.sqrt(2 / self.m) * (sine @ (c * self.omega[j])) / self.sd[j]
            result.append(g)
        return np.column_stack(result)


def fit_models(x, y, seed=1701):
    result = {"linear": SmoothRegression("linear").fit(x, y)}
    options = [k for k in [6, 12, 20] if k * x.shape[1] < len(x) / 2]
    candidates = [SmoothRegression("additive", knots=k).fit(x, y) for k in options]
    result["additive"] = min(candidates, key=lambda m: m.gcv)
    result["interaction"] = SmoothRegression(
        "interaction", knots=result["additive"].knots, seed=seed
    ).fit(x, y)
    return result


def selected_additive(x, y):
    """BIC selects nonlinear components and generic knot count from the data.

    Linear terms remain; no variable is predesignated as nonlinear. This is a
    finite additive model-selection approximation, not a generic inference rule.
    """
    x = np.asarray(x)
    n, p = x.shape
    best_model = None
    for knots in [6, 12, 20]:
        if knots * p >= n / 2:
            continue
        model = SmoothRegression("additive", knots=knots)
        model.mean = x.mean(0)
        model.sd = x.std(0)
        model.p = p
        z = (x - model.mean) / model.sd
        X = np.column_stack([np.ones(n), z])
        base = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ base
        model.spline = SplineTransformer(
            n_knots=knots, degree=3, include_bias=False, extrapolation="linear"
        ).fit(z)
        model.width = model.spline.bsplines_[0].c.shape[0] - 1
        F = model.spline.transform(z)
        H = np.linalg.lstsq(X, F, rcond=None)[0]
        U = F - X @ H
        Q = []
        transforms = []
        groups = []
        offset = 0
        for j in range(p):
            a = U[:, j * model.width : (j + 1) * model.width]
            uu, s, v = np.linalg.svd(a, full_matrices=False)
            rank = int(np.sum(s > 1e-8 * s[0]))
            transform = v[:rank].T / s[:rank] * np.sqrt(n)
            q = a @ transform
            Q.append(q)
            transforms.append(transform)
            groups.append(list(range(offset, offset + rank)))
            offset += rank
        Q = np.column_stack(Q)
        G = Q.T @ Q
        rhs = Q.T @ r
        yy = float(r @ r)

        def evaluate(indices):
            if not indices:
                return yy, np.zeros(0), p + 1
            gram = G[np.ix_(indices, indices)]
            co = np.linalg.solve(gram, rhs[indices])
            rss = max(yy - float(co @ rhs[indices]), 1e-20)
            return rss, co, p + 1 + len(indices)

        def bic(rss, df):
            return n * np.log(max(rss / n, 1e-20)) + df * np.log(n)

        chosen = []
        indices = []
        rss, co, df = evaluate(indices)
        criterion = bic(rss, df)
        while len(chosen) < p:
            candidates = []
            for j in range(p):
                if j in chosen:
                    continue
                ix = indices + groups[j]
                rr, cc, dd = evaluate(ix)
                candidates.append((bic(rr, dd), j, ix, rr, cc, dd))
            candidate = min(candidates, key=lambda q: q[0])
            if candidate[0] >= criterion:
                break
            criterion, j, indices, rss, co, df = candidate
            chosen.append(j)
        raw = np.zeros(F.shape[1])
        start = 0
        for j in chosen:
            count = len(groups[j])
            raw[j * model.width : (j + 1) * model.width] = (
                transforms[j] @ co[start : start + count]
            )
            start += count
        model.coef = raw
        model.beta = base - H @ raw
        model.bic = criterion
        model.selected_components = chosen
        model.df = float(df)
        model.penalty = 0.0
        model.gcv = (rss / n) / (1 - df / n) ** 2
        if best_model is None or model.bic < best_model.bic:
            best_model = model
    return best_model
