# Model specification and analysis scope

The simulation mechanisms and their numerical settings are defined in `experiments/`. Each study includes a methods document, seeded simulation code and numerical checks.

The customer model uses a monthly-usage coefficient of 3 and independently generated bugs faced. A coefficient-30 model is included as an explicitly labelled sensitivity analysis. Graph-implied null effects are distinguished from effects estimated from observational data.

The studies are exploratory and are not claimed to be preregistered. The customer nonlinear-component selection was introduced after derivative instability was observed; the linear, additive and interaction-capable fits remain reported. The selected model is an exploratory extension.

The matched-mechanism design was recorded before its noisy calibration. `design-record.json` records the design and source hashes. Calibration assumes the stated spectral model family. It does not establish unrestricted response-learning guarantees.

The package contains synthetic experimental code, reference outputs and technical documentation. All commands run using the files supplied in this repository and the recorded dependency environment. Numerical checks, smaller fresh-fit checks and complete simulation runs serve different purposes; see [reproduction details](REPRODUCIBILITY.md).
