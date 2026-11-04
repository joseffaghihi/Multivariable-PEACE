from experiment import (
    D,
    DOM,
    L,
    M,
    N,
    RES,
    derivative,
    gaussian_kde,
    generate,
    logsumexp,
    need_density,
    need_deriv,
    np,
    pair_grid,
    pd,
    plt,
    quad,
    savefig,
    weighted,
)


def main():
    ds = np.linspace(0, 3, 101)
    t, w = quad(*DOM[N], 180)
    g = need_deriv(t)
    lf = np.log(need_density(t))
    rows = []
    for d in ds:
        p = np.exp(np.log(w) + 2 * d * lf - logsumexp(np.log(w) + 2 * d * lf))
        m = float(p @ g)
        slope = 2 * float(p @ (g * lf) - m * (p @ lf))
        curvature = 4 * float(p @ ((g - m) * (lf - p @ lf) ** 2))
        rows.append(
            dict(d=d, peace=m, derivative_d=slope, second_derivative_d=curvature)
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "theorem_interpretation.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(df.d, df.peace)
    axes[0].axhline(
        need_deriv(np.array([0.2]))[0], ls=":", color="black", label="Unique-mode limit"
    )
    axes[0].set(
        title="Product need: small, real decrease", xlabel="d", ylabel="Mean PEACE"
    )
    axes[0].ticklabel_format(useOffset=False)
    axes[0].legend(fontsize=7)
    axes[1].plot(df.d, df.derivative_d)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set(
        title="Theorem 8.2: 2 Cov(magnitude, log f)",
        xlabel="d",
        ylabel="Derivative with respect to d",
    )
    for pair in [(N, M), (N, D)]:
        U, V, W, lf, g1, g2 = pair_grid(pair, 120)
        values = [
            weighted(np.hypot(g1, g2).ravel(), lf.ravel(), W.ravel(), d)[0] for d in ds
        ]
        axes[2].plot(ds, np.array(values) - values[0], label=" + ".join(pair))
    axes[2].set(
        title="Joint curves can move in opposite directions",
        xlabel="d",
        ylabel="Change from the d=0 value",
    )
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    savefig("13_theorem_interpretation")
    # Explicit interaction positive control, not part of the original simulator.
    U, V, W, lf, g1, g2 = pair_grid((N, M), 120)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kappa in [0, 10]:
        gn = g1 + kappa * (V - 0.515)
        gm = g2 + kappa * (U - 0.2)
        vals = [
            weighted(np.hypot(gn, gm).ravel(), lf.ravel(), W.ravel(), d)[0] for d in ds
        ]
        axes[0].plot(ds, vals, label=f"Added interaction coefficient κ={kappa}")
        # Four-intervention interaction contrast over fixed, supported values.
        contrast = kappa * (0.3 - 0.1) * (0.55 - 0.48)
        rows.extend(
            dict(
                kappa=kappa,
                d=d,
                joint_peace=v,
                mixed_partial=kappa,
                four_intervention_contrast=contrast,
            )
            for d, v in zip(ds, vals)
        )
    axes[0].set(xlabel="d", ylabel="Joint Mean PEACE")
    axes[0].legend(fontsize=7)
    axes[1].bar(["Original: κ=0", "Positive control: κ=10"], [0, 0.14])
    axes[1].set(ylabel="Four-intervention interaction contrast (Y units)")
    fig.suptitle("Joint responsiveness is positive even when interaction is zero")
    fig.tight_layout()
    savefig("14_interaction_positive_control")
    pd.DataFrame(rows).to_csv(RES / "interaction_positive_control.csv", index=False)
    # Explicit expected residual variation is more informative than labels.
    df = pd.read_csv(RES / "overlap.csv")
    df = df[df.scenario == "original"].drop_duplicates("variable")
    plt.figure(figsize=(10, 4))
    plt.bar(df.variable, np.maximum(df.variance_ratio, 1e-8))
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Residual / marginal variance (log scale)")
    plt.title(
        "Parent-adjusted treatment variation; reported-bugs bar is a display floor"
    )
    plt.tight_layout()
    savefig("15_overlap")
    # Separate density-only artifact, holding the true derivative fixed.
    x, _ = generate(1000, seed=999101)
    t, w = quad(0, 10, 180)
    g = derivative(L, t)
    rows = []
    plt.figure(figsize=(8, 4))
    for mult in [0.5, 1, 2]:
        for reflection in [False, True]:
            kde = gaussian_kde(x[L], bw_method=lambda k: k.scotts_factor() * mult)
            f = kde(t)
            if reflection:
                f = f + kde(-t) + kde(20 - t)
            vals = [weighted(g, np.log(f), w, d)[0] for d in ds]
            plt.plot(
                ds,
                vals,
                label=f"{mult:g} × Scott, "
                + ("reflected" if reflection else "unbounded"),
            )
            rows.extend(
                dict(d=d, bandwidth=mult, reflection=reflection, peace=v)
                for d, v in zip(ds, vals)
            )
    plt.axhline(8 / 11, color="black", ls="--", label="Exact uniform-density truth")
    plt.xlabel("d")
    plt.ylabel("Mean PEACE with oracle derivative")
    plt.legend(fontsize=7)
    plt.tight_layout()
    savefig("16_density_only_upgrade")
    pd.DataFrame(rows).to_csv(RES / "density_only_upgrade.csv", index=False)


if __name__ == "__main__":
    main()
