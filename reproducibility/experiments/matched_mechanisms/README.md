# Matched summaries and different weighted causal variation

Two scalar response mechanisms have matched endpoint contrasts, signed average derivatives, natural-weight absolute and squared derivatives, outcome distributions, information treatment-edge strength and the listed Sobol summaries. Their treatment-local variation is arranged differently across treatment regions and covariate profiles. Varying the declared PEACE weights can therefore distinguish them and change their ordering.

From the repository root:

    python reproduce.py --study matched --mode smoke
    python reproduce.py --study matched --mode validate
    python reproduce.py --study matched --mode figures
    python reproduce.py --study matched --mode full

DESIGN.md states the fixed equations, density/profile weights, sample sizes and calibration design. Mathematical_argument.md gives the matching argument. Calibration fits an explicit 20-mode bounded spectral family, with rare profiles oversampled and an independent natural reference sample supplying weights. This is model-conditional estimation, not unrestricted response learning.

A weighted squared-derivative comparator is retained. An absolute derivative with identical weights is PEACE's own gradient representation, so exclusive superiority over that same functional is not claimed. Larger PEACE is more local response movement under specified priorities, not a better controller, treatment or prediction model.
