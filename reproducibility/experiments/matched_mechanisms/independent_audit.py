"""Numerical integration of responses; separate from closed-form baseline identities."""

from experiment import (
    GRID,
    K,
    NOISE,
    PI,
    RES,
    RHO,
    WIDTH,
    json,
    leggauss,
    np,
    oracle_arrays,
    pd,
    response,
    score,
)

x, w = leggauss(512)
u = (x + 1) / 2
weights = w / 2
xx, ww = leggauss(24)
rows = []
score_errors = []
energy_errors = []
signed_errors = []
baseline_errors = []
for model in K:
    tv = np.zeros((2, 3))
    energy = np.zeros_like(tv)
    for z in range(2):
        for j in range(3):
            k = K[model][z, j]
            edges = np.linspace(0, 1, 2 * k + 1)
            pts = (edges[:-1, None] + edges[1:, None]) / 2 + np.diff(edges)[
                :, None
            ] * xx / 2
            qw = np.diff(edges)[:, None] * ww / 2 * WIDTH
            grad = response((j + pts) / 3, z, model, True)
            tv[z, j] = np.sum(qw * np.abs(grad))
            energy[z, j] = np.sum(qw * grad**2)
            signed_errors.append(abs(np.sum(qw * grad)))
    for d, r in GRID:
        p = RHO ** (2 * d)
        p /= p.sum()
        q = PI**r
        q /= q.sum()
        values = np.array(
            [[response((j + u) / 3, z, model) for j in range(3)] for z in range(2)]
        )
        mean = np.einsum("z,j,i,zji->", q, p, weights, values)
        second = np.einsum("z,j,i,zji->", q, p, weights, values**2)
        mean_t = np.einsum("z,zji->ji", q, values)
        vt = np.einsum("j,i,ji->", p, weights, mean_t**2) - mean**2
        means_z = np.einsum("j,i,zji->z", p, weights, values)
        vz = q @ means_z**2 - mean**2
        variance = second - mean**2
        var_y = variance + NOISE**2
        found = {
            "output_mean": mean,
            "output_variance": var_y,
            "T_first_sobol": vt / var_y,
            "T_total_sobol": (variance - vz) / var_y,
            "Z_first_sobol": vz / var_y,
            "Z_total_sobol": (variance - vt) / var_y,
        }
        expected = {
            "output_mean": 0.5,
            "output_variance": 0.125 + NOISE**2,
            "T_first_sobol": (q @ q) / 8 / var_y,
            "T_total_sobol": 0.125 / var_y,
            "Z_first_sobol": 0.0,
            "Z_total_sobol": (1 - q @ q) / 8 / var_y,
        }
        baseline_errors.extend(abs(found[key] - expected[key]) for key in expected)
        score_errors.append(abs(score(tv, d, r) - score(oracle_arrays(model)[0], d, r)))
        energy_errors.append(
            abs(score(energy, d, r) - score(oracle_arrays(model)[1], d, r))
        )
        rows.append(dict(model=model, d=d, r=r, **found))
checks = {
    "response_moment_and_sobol_max_error": float(max(baseline_errors)),
    "peace_direct_quadrature_max_error": float(max(score_errors)),
    "energy_direct_quadrature_max_error": float(max(energy_errors)),
    "signed_integral_max_error": float(max(signed_errors)),
}
checks["passed"] = all(v < 1e-9 for v in checks.values())
assert checks["passed"]
pd.DataFrame(rows).to_csv(RES / "independent_moment_sobol_audit.csv", index=False)
(RES / "independent_audit_checks.json").write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
