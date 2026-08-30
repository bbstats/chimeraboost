"""Scoring for a fitted regressor or classifier.

`chimeraboost.quantile_metrics` does this for a predicted quantile grid; this
is the same idea for the two point estimators, so that every model in the
library can be asked how it did without wiring up sklearn by hand.

Two things it does that a bare `mean_squared_error` call does not:

  * **Skill, not just error.** Every headline number is reported alongside a
    skill score against the no-skill forecast -- the training mean for
    regression, the class prior for classification. 1 is perfect, 0 is no
    better than ignoring every feature, negative is worse than that. An RMSE
    of 3.1 means nothing on its own; an R2 of 0.02 does.
  * **Calibration.** For classification, how much a monotone recalibration
    would improve the Brier score (the CORP miscalibration measure). This
    library temperature-scales `predict_proba`, and this is the number that
    says whether that worked.

Definitions match `benchmarks/run_benchmarks.py` exactly, so a number here is
comparable with the project's own benchmark tables.
"""

import numpy as np


def regression_report(y, pred, sample_weight=None, baseline=None):
    """Score point predictions: ``rmse``, ``mae``, ``r2``, ``n``.

    ``r2`` is the skill score against a constant forecast -- 1 perfect, 0 no
    better than always predicting the mean, negative worse. ``baseline`` sets
    which mean: pass the training targets to score against what the model
    actually had available, or leave it to use ``y`` itself, which is the
    conservative reading (the best constant forecast in hindsight).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if pred.shape != y.shape:
        raise ValueError(f"pred has shape {pred.shape} but y has {y.shape}.")

    err = y - pred
    mse = float(np.average(err ** 2, weights=sample_weight))
    ref = y if baseline is None else np.asarray(baseline,
                                                dtype=np.float64).ravel()
    mu = float(np.average(ref, weights=None))
    var = float(np.average((y - mu) ** 2, weights=sample_weight))
    return {
        "n": int(y.shape[0]),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.average(np.abs(err), weights=sample_weight)),
        "r2": float("nan") if var <= 0.0 else 1.0 - mse / var,
    }


def _onehot(y, classes):
    return (np.asarray(y)[:, None] == np.asarray(classes)[None, :]
            ).astype(np.float64)


def classification_report(y, proba, classes, sample_weight=None):
    """Score predicted probabilities: ``log_loss``, ``brier``,
    ``brier_skill``, ``accuracy``, ``f1_macro``, ``calibration_mcb``, ``n``.

    ``brier`` is the multiclass form, the mean over rows of
    ``sum_k (p_k - onehot_k)**2`` -- a proper scoring rule like log loss, but
    bounded, so it aggregates across datasets without an unbounded tail. Binary
    uses the same K=2 sum, so both tasks share one definition.

    ``brier_skill`` scores it against the class prior: 1 perfect, 0 no better
    than predicting the base rates.

    ``calibration_mcb`` is the CORP miscalibration measure
    (Dimitriadis, Gneiting & Jordan): how much an optimal *monotone*
    recalibration would improve the per-class Brier score. 0 means already
    perfectly calibrated, higher is worse. Fitted in-sample on the scored fold,
    which is the standard CORP diagnostic.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import f1_score, log_loss

    classes = np.asarray(classes)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != classes.shape[0]:
        raise ValueError(
            f"proba must be (n_samples, {classes.shape[0]}); got "
            f"{proba.shape}.")
    if proba.shape[0] != len(y):
        raise ValueError(f"proba has {proba.shape[0]} rows but y has {len(y)}.")

    onehot = _onehot(y, classes)
    pred = classes[proba.argmax(axis=1)]

    brier = float(np.average(np.sum((proba - onehot) ** 2, axis=1),
                             weights=sample_weight))
    prior = np.average(onehot, axis=0, weights=sample_weight)
    brier_ref = float(np.average(np.sum((prior[None, :] - onehot) ** 2, axis=1),
                                 weights=sample_weight))

    mcb_k = []
    for k in range(proba.shape[1]):
        p, yk = proba[:, k], onehot[:, k]
        recal = IsotonicRegression(increasing=True,
                                   out_of_bounds="clip").fit_transform(p, yk)
        mcb_k.append(np.mean((p - yk) ** 2) - np.mean((recal - yk) ** 2))

    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, proba, labels=classes,
                                   sample_weight=sample_weight)),
        "brier": brier,
        "brier_skill": (float("nan") if brier_ref <= 0.0
                        else 1.0 - brier / brier_ref),
        "accuracy": float(np.average((pred == np.asarray(y)).astype(float),
                                     weights=sample_weight)),
        "f1_macro": float(f1_score(y, pred, average="macro",
                                   sample_weight=sample_weight)),
        "calibration_mcb": float(np.mean(mcb_k)),
    }


_LABELS = {
    "rmse": "RMSE (lower better)",
    "mae": "MAE (lower better)",
    "r2": "R2 skill vs the mean (1 perfect, 0 no better)",
    "log_loss": "log loss (lower better)",
    "brier": "Brier score (lower better)",
    "brier_skill": "Brier skill vs the prior (1 perfect, 0 no better)",
    "accuracy": "accuracy",
    "f1_macro": "F1 macro",
    "calibration_mcb": "miscalibration (0 is perfectly calibrated)",
}


def format_report(report, title=None):
    """Render `regression_report` or `classification_report` as a fixed-width
    text block, skill scores last so the eye lands on them."""
    lines = [title] if title else []
    lines.append(f"rows scored: {report['n']}")
    width = max(len(v) for v in _LABELS.values())
    for key, label in _LABELS.items():
        if key in report:
            lines.append(f"{label:<{width}s}  {report[key]:>10.6f}")
    return "\n".join(lines)
