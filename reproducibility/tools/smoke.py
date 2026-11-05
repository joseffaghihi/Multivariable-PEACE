"""Small seeded fits and numerical checks for all four studies."""

import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def sodium():
    sys.path.insert(0, str(ROOT / "experiments/sodium"))
    e = importlib.import_module("experiment")
    e.oracle()
    x, y, _ = e.generate(800, seed=7139)
    grid = e.Grid(96)
    values = {}
    for kind in ("structural", "spline"):
        fit = e.ResponseFit(x, y, kind)
        values[kind] = e.aggregation(
            grid, e.fitted_logs(grid, x), *fit.derivatives(grid)
        ).tolist()
        assert e.np.isfinite(values[kind]).all()
    truth = e.aggregation(
        grid,
        e.truth_logs(grid),
        *e.true_derivatives(grid.s, grid.a),
        targets=[(0.5, 1)],
    )[0, 0]
    assert abs(truth - 3.85692) < 5e-6
    # The comparator targets a signed projection slope, not the PEACE functional.
    from econml.dml import CausalForestDML, LinearDML
    from sklearn.linear_model import LinearRegression

    features = x[:, [1]]
    for estimator in [
        LinearDML(
            model_y=LinearRegression(),
            model_t=LinearRegression(),
            cv=2,
            random_state=19,
        ),
        CausalForestDML(
            model_y=LinearRegression(),
            model_t=LinearRegression(),
            cv=2,
            n_estimators=16,
            min_samples_leaf=10,
            n_jobs=1,
            random_state=19,
        ),
    ]:
        estimator.fit(y, x[:, 0], X=features)
        assert e.np.isfinite(estimator.const_marginal_effect(features[:10])).all()
    return {"oracle_sodium": float(truth), "fits": values, "comparators": "passed"}


def customer():
    folder = ROOT / "experiments/customer_satisfaction"
    sys.path.insert(0, str(folder / "flexible_estimation"))
    sys.path.insert(0, str(folder))
    e = importlib.import_module("experiment")
    study = importlib.import_module("study")
    study.validate()
    x, y = e.generate(800, seed=8139, noise=0.5)
    # Exercise the prediction and attribution dependencies with a small fit.
    from xgboost import XGBRegressor
    import shap

    model = XGBRegressor(n_estimators=16, max_depth=2, n_jobs=1, random_state=19)
    model.fit(x, y)
    attribution = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent"
    ).shap_values(x.iloc[:10])
    assert e.np.isfinite(attribution).all()
    assert abs(study.truth_total(e.L, 0.5, "original") - 8 / 11) < 1e-10
    return {
        "analytic_derivatives_and_standardization": "passed",
        "xgboost_and_shap": "passed",
        "upgrade_truth": 8 / 11,
    }


def matched():
    sys.path.insert(0, str(ROOT / "experiments/matched_mechanisms"))
    e = importlib.import_module("experiment")
    e.validate()
    e.run(reps=1)
    raw = e.pd.read_csv(e.RES / "calibration_results.csv")
    assert len(raw) == 120 and e.np.isfinite(raw.select_dtypes("number")).all().all()
    a = e.score(e.oracle_arrays("A")[0], 0.5, 1)
    b = e.score(e.oracle_arrays("B")[0], 0.5, 1)
    assert abs(a - b) < 1e-12 and abs(a - 15.6) < 1e-12
    assert e.score(e.oracle_arrays("A")[0], 1, 0) > e.score(
        e.oracle_arrays("B")[0], 1, 0
    )
    assert e.score(e.oracle_arrays("A")[0], 1, 1) < e.score(
        e.oracle_arrays("B")[0], 1, 1
    )
    return {
        "matched_natural_magnitude": a,
        "degree_dependent_ordering": "passed",
        "calibration_rows": len(raw),
    }


def iv():
    folder = ROOT / "experiments/instrumental_variables/quadratic_comparison"
    sys.path.insert(0, str(folder))
    study = importlib.import_module("comparison")
    audit = importlib.import_module("audit")
    truth = study.functionals(study.true_theta())
    index = list(study.DEGREES).index(0.5)
    assert abs(truth["contrast"][index] - 1.75) < 1e-13
    assert abs(truth["peace"][index] - 1.41838949) < 5e-9
    independent = audit.independent_metrics(study.true_theta(), 0.5)
    assert abs(independent["peace"] - truth["peace"][index]) < 1e-11
    latent = study.core.generate_latent(2000, 17)
    errors = []
    for rho in (0.8, 0.05):
        w, x, y = study.observed(latent, rho)
        models = study.fit_models(x, w, y)
        assert len(models) == 7
        theta = next(theta for method, _, theta, _ in models if method == "IV degree 2")
        p, q = study.core.basis(x, 2), study.core.basis(w, 2)
        a, b, *_ = study.core.centered_moments(p, q, y)
        error = float(study.np.max(abs(a @ theta[1:] - b)))
        assert error < 1e-10
        values = study.functionals(theta)
        assert study.np.isfinite(values["peace"]).all()
        independent = audit.independent_metrics(theta, 0.5)
        assert abs(values["peace"][index] - independent["peace"]) < 1e-8
        errors.append(error)
    return {
        "contrast_truth": float(truth["contrast"][index]),
        "mean_peace_truth": float(truth["peace"][index]),
        "fresh_strong_and_weak_iv_fits": "passed",
        "moment_residuals": errors,
    }


if __name__ == "__main__":
    name = sys.argv[1]
    result = {
        "study": name,
        "checks": {
            "sodium": sodium,
            "customer": customer,
            "matched": matched,
            "iv": iv,
        }[name](),
        "passed": True,
    }
    (ROOT / f"smoke-{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
