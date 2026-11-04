"""Separate response-learning and fixed-reference-budget errors; no main-result replacement."""

from experiment import (
    K,
    NOISE,
    PI,
    RES,
    fit_arrays,
    json,
    np,
    oracle_arrays,
    pd,
    response,
    sample_t,
    score,
)


def main():
    original = pd.read_csv(RES / "calibration_results.csv").set_index(
        ["n", "rep", "model", "d", "r"]
    )
    rows = []
    errors = []
    for n in [2000, 8000, 32000]:
        for rep in range(40):
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
                oracle_tv, _ = oracle_arrays(model)
                for d, r in [(1, 0), (1, 1), (0.5, 1)]:
                    combined = score(tv[:, :, index], d, r, rho, pi)
                    errors.append(
                        abs(combined - original.loc[n, rep, model, d, r]["estimate"])
                    )
                    rows.append(
                        dict(
                            n=n,
                            rep=rep,
                            model=model,
                            d=d,
                            r=r,
                            truth=score(oracle_tv, d, r),
                            response_only=score(tv[:, :, index], d, r),
                            density_only=score(oracle_tv, d, r, rho, pi),
                            combined=combined,
                        )
                    )
        print("completed response/density error separation", n, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "response_density_audit.csv", index=False)
    check = {
        "max_replay_difference": max(errors),
        "passed": bool(max(errors) < 1e-9),
        "scope": "Oracle components isolate estimation errors for diagnosis only; they are not input to the original estimates.",
    }
    (RES / "density_audit_checks.json").write_text(json.dumps(check, indent=2))
    assert check["passed"]


if __name__ == "__main__":
    main()
