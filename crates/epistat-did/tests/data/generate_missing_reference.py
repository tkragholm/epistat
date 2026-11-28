# Reference values for epistat_did::missing::estimate, the difference-in-
# differences estimator of Testa, Kennedy and Reimherr for a pre-treatment
# outcome missing at random, from the authors' replication code.
#
# Written for epistat and licensed as epistat is (MIT OR Apache-2.0). The
# replication code is not part of epistat: this script loads `dgp.py` from a
# clone of github.com/testalorenzo/MissingDiD given on the command line and
# calls its functions unchanged.
#
# Run from the crate root, in an environment with numpy, pandas, statsmodels
# and tqdm (the versions are checked below):
#
#   git clone https://github.com/testalorenzo/MissingDiD /some/empty/dir/MissingDiD
#   python -I tests/data/generate_missing_reference.py /some/empty/dir/MissingDiD \
#       > tests/data/missing_ref.json
#
# For each assumption (their '2.3', outcome-independent, and '2.4',
# outcome-dependent) one data set of 500 units is drawn with their
# `generate_data` and their `fit_missingdid` is run for every combination of
# correctly specified (Z) and misspecified (X) nuisance models: 8 under 2.3,
# 32 under 2.4 (with and without the augmented pseudo-outcome of the nested
# regression). The script records, from the local variables of each call,
# the nuisance predictions the estimate used (after their clipping of
# probabilities to [0.01, 0.99]) and the per-unit terms whose sum is the
# estimate, with the estimate and the standard error the function returns.
#
# `fit_missingdid` regresses the nested outcome on the covariates of all
# units, where the paper regresses it among the controls. The script adds,
# for each 2.4 case, that regression among the controls (statsmodels OLS on
# the same pseudo-outcome), so the tests can check the estimator of the paper
# with the rest of the nuisance fits from the replication code.
#
# For the tolerances of the tests, each nuisance fit also gets:
#   gap      the largest difference between the fit's predictions and those of
#            the same model solved more accurately: three more Newton steps
#            from the solution for a logistic model, a QR least-squares solve
#            for a linear one. It measures the solver and rounding error of the
#            reference.
#   kappa    the 2-norm condition number of the fitting rows of the design
#            with each covariate centred and scaled by its mean and standard
#            deviation over all units, the design epistat solves.
#   clipped  the number of units whose probability was moved by the clipping,
#            among the units where it enters the estimator.
import importlib.util
import io
import json
import platform
import subprocess
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.api import OLS, Logit, add_constant

VERSIONS = {"numpy": "2.5.5", "pandas": "3.1.1", "statsmodels": "0.15.0"}
assert np.__version__ == VERSIONS["numpy"], np.__version__
assert pd.__version__ == VERSIONS["pandas"], pd.__version__
assert statsmodels.__version__ == VERSIONS["statsmodels"], statsmodels.__version__

src = sys.argv[1]
commit = subprocess.run(
    ["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
spec = importlib.util.spec_from_file_location("missingdid_dgp", f"{src}/dgp.py")
dgp = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    spec.loader.exec_module(dgp)
warnings.filterwarnings("ignore")

N = 500
SEEDS = {"2.3": 20261125, "2.4": 20261126}
Z = ["z1", "z2", "z3", "z4"]
X = ["x1", "x2", "x3", "x4"]


def captured_call(data, **kwargs):
    """Calls dgp.fit_missingdid and returns its result with its local variables at return."""
    seen = {}

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is dgp.fit_missingdid.__code__:
            seen.update(frame.f_locals)

    sys.setprofile(profile)
    try:
        result = dgp.fit_missingdid(data, **kwargs)
    finally:
        sys.setprofile(None)
    return result, seen


def arr(v):
    return np.asarray(v, dtype=float)


def standardised(data, cols, rows):
    m = data[cols].to_numpy(dtype=float)
    m = (m - m.mean(axis=0)) / m.std(axis=0)
    return np.column_stack([np.ones(len(m)), m])[rows]


def kappa(data, cols, rows):
    return float(np.linalg.cond(standardised(data, cols, rows)))


def logit_gap(y, design_fit, design_all):
    """Predictions of the replication code's fit, and the change three more Newton steps make."""
    fit = Logit(y, design_fit).fit(disp=0)
    more = Logit(y, design_fit).fit(disp=0, start_params=fit.params, method="newton", maxiter=3, tol=0.0)
    a = arr(fit.predict(design_all))
    b = arr(more.predict(design_all))
    return a, float(np.max(np.abs(a - b)))


def ols_gap(y, design_fit, design_all):
    fit = OLS(y, design_fit).fit()
    a = arr(fit.predict(design_all))
    beta, *_ = np.linalg.lstsq(np.asarray(design_fit, dtype=float), np.asarray(y, dtype=float), rcond=None)
    b = np.asarray(design_all, dtype=float) @ beta
    return a, float(np.max(np.abs(a - b)))


def nuisance_fits(data, ass):
    """The replication code's nuisance predictions, each with its gap, kappa and clipping count."""
    t = data["t"].to_numpy() == 1
    r = data["r"].to_numpy() == 1
    extra = [] if ass == "2.3" else ["y1"]
    out = {}
    for label, cols in (("Z", Z), ("X", X)):
        correct = label == "Z"
        # Propensity, all units, clipped where it enters: the controls.
        used = dgp.estimate_propensity_scores(data, use_correct_model=correct)
        raw, gap = logit_gap(data["t"], add_constant(data[cols]), add_constant(data[cols]))
        assert np.array_equal(np.clip(raw, 0.01, 0.99), arr(used))
        out[f"pi_{label}"] = {
            "values": arr(used).tolist(),
            "gap": gap,
            "kappa": kappa(data, cols, slice(None)),
            "clipped": int(np.sum(~t & (raw != np.clip(raw, 0.01, 0.99)))),
        }
        # Missingness, by arm, clipped where it enters: the arm's units with Y0.
        for arm, name in ((1, "gamma1"), (0, "gamma0")):
            rows = t if arm == 1 else ~t
            used = dgp.estimate_missingness_scores(data, ass=ass, t=arm, use_correct_model=correct)
            design = add_constant(data[cols + extra])
            raw, gap = logit_gap(data.loc[rows, "r"], design[rows], design)
            assert np.array_equal(np.clip(raw, 0.01, 0.99), arr(used))
            out[f"{name}_{label}"] = {
                "values": arr(used).tolist(),
                "gap": gap,
                "kappa": kappa(data, cols + extra, rows),
                "clipped": int(np.sum(rows & r & (raw != np.clip(raw, 0.01, 0.99)))),
            }
        # Outcome regressions: Y0 among the arm's units with Y0, Y1 among the controls.
        for arm, target, name, rows, xcols in (
            (1, "y0", "mu01", t & r, cols + extra),
            (0, "y1", "mu10", ~t, cols),
            (0, "y0", "mu00", ~t & r, cols + extra),
        ):
            used = dgp.get_y_model_predictions(data, ass=ass, t=arm, target=target, use_correct_model=correct)
            design = add_constant(data[xcols])
            raw, gap = ols_gap(data.loc[rows, target], design[rows], design)
            assert np.allclose(raw, arr(used), rtol=0, atol=1e-9)
            out[f"{name}_{label}"] = {
                "values": arr(used).tolist(),
                "gap": max(gap, float(np.max(np.abs(raw - arr(used))))),
                "kappa": kappa(data, xcols, rows),
                "clipped": 0,
            }
    return out


def nested_among_controls(data, m00, r_hat0, eta_correct, augmentation):
    """The nested regression of the paper: the pseudo-outcome regressed on the covariates among the controls."""
    t = data["t"].to_numpy() == 1
    filled = data.fillna(0)
    m00 = arr(m00)
    pseudo = m00 + filled["r"].to_numpy() * (filled["y0"].to_numpy() - m00) / arr(r_hat0) if augmentation else m00
    cols = Z if eta_correct else X
    design = add_constant(data[cols])
    raw, gap = ols_gap(pseudo[~t], design[~t], design)
    return raw, gap, kappa(data, cols, ~t)


cases = []
for ass in ("2.3", "2.4"):
    np.random.seed(SEEDS[ass])
    data = dgp.generate_data(n=N, att=5, ass=ass)
    t = data["t"].to_numpy()
    fits = nuisance_fits(data, ass)
    specs = []
    combos = (
        [(mu, g, p, False, False) for mu in (True, False) for g in (True, False) for p in (True, False)]
        if ass == "2.3"
        else [
            (mu, g, p, e, a)
            for mu in (True, False)
            for g in (True, False)
            for p in (True, False)
            for e in (True, False)
            for a in (True, False)
        ]
    )
    for mu, g, p, e, a in combos:
        (estimate, std), loc = captured_call(
            data,
            ass=ass,
            use_correct_model_mu=mu,
            use_correct_model_gamma=g,
            use_correct_model_pi=p,
            use_correct_model_eta=e,
            augmentation=a,
        )
        L = lambda b: "Z" if b else "X"  # noqa: E731
        keys = {
            "pi": f"pi_{L(p)}",
            "gamma1": f"gamma1_{L(g)}",
            "gamma0": f"gamma0_{L(g)}",
            "mu01": f"mu01_{L(mu)}",
            "mu10": f"mu10_{L(mu)}",
            "mu00": f"mu00_{L(mu)}",
        }
        for local, key in (
            ("pi_hat", "pi"),
            ("r_hat1", "gamma1"),
            ("r_hat0", "gamma0"),
            ("m_hat01", "mu01"),
            ("m_hat10", "mu10"),
            ("m_hat00", "mu00"),
        ):
            assert np.array_equal(arr(loc[local]), arr(fits[keys[key]]["values"])), (local, key)
        row = {
            "mu": mu,
            "gamma": g,
            "pi": p,
            "eta": e,
            "augmentation": a,
            "nuisance": keys,
            "estimate": float(estimate),
            "std": float(std),
            "terms": arr(loc["influence_function"]).tolist(),
        }
        if ass == "2.4":
            # The nested regression depends on the outcome, missingness and
            # nested models and on the augmentation, not on the propensity.
            tag = f"{L(mu)}{L(g)}{L(e)}{'A' if a else 'P'}"
            keys["eta_all_units"] = f"eta_all_units_{tag}"
            keys["eta_controls"] = f"eta_controls_{tag}"
            if keys["eta_all_units"] not in fits:
                fits[keys["eta_all_units"]] = {"values": arr(loc["eta_hat00"]).tolist()}
                raw, gap, kap = nested_among_controls(data, loc["m_hat00"], loc["r_hat0"], e, a)
                fits[keys["eta_controls"]] = {"values": raw.tolist(), "gap": gap, "kappa": kap, "clipped": 0}
            assert np.array_equal(arr(loc["eta_hat00"]), arr(fits[keys["eta_all_units"]]["values"]))
        specs.append(row)
    cases.append(
        {
            "assumption": "MarX" if ass == "2.3" else "MarXY1",
            "seed": SEEDS[ass],
            "n": N,
            "z": data[Z].to_numpy().ravel().tolist(),
            "x": data[X].to_numpy().ravel().tolist(),
            "y0": [None if np.isnan(v) else float(v) for v in data["y0"].to_numpy()],
            "y1": data["y1"].to_numpy(dtype=float).tolist(),
            "treated": [bool(v) for v in t],
            "observed": [bool(v) for v in data["r"].to_numpy()],
            "fits": fits,
            "specs": specs,
        }
    )

json.dump(
    {
        "generator": "tests/data/generate_missing_reference.py",
        "missingdid_repository": "https://github.com/testalorenzo/MissingDiD",
        "missingdid_commit": commit,
        "python": platform.python_version(),
        "versions": VERSIONS,
        "cases": cases,
    },
    sys.stdout,
)
