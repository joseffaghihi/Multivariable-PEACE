# Sodium intake and blood pressure

This study estimates controlled mean-response PEACE and Mean PEACE in a known synthetic sodium-age-blood-pressure model. It separates degree interpretation, response/density estimation, overlap, noise, confounding sensitivity and comparator estimands.

From the repository root:

    python reproduce.py --study sodium --mode smoke
    python reproduce.py --study sodium --mode validate
    python reproduce.py --study sodium --mode figures
    python reproduce.py --study sodium --mode full

METHODS.md gives the equations, domains, seeds and numerical settings. experiment.py contains simulation and estimation; extra_analysis.py covers joint targets and adjustment sensitivity; separability_analysis.py distinguishes individual from mean-response variation. validate.py and validate_kde.py check the numerical calculations. make_report.py builds summaries and readable reports.

The reference results include all final scenarios. Causal Forest and DML are scored against their own signed projection-slope targets, not arbitrary PEACE values. The KDE study isolates density error with a known response derivative. No instrument is present, and weak-instrument performance is not assessed.
