"""Stage-1 probe for hierarchical shrinkage (hs_lambda).

Mechanism -> benefit smoke test BEFORE touching real benchmarks: does shrinking
leaf values toward ancestors help the kind of data ChimeraBoost is weakest on
(high-signal classification Brier), without wrecking regression RMSE or speed?

Synthetic families:
  * high-signal binary  : low-noise logistic boundary with interactions (the
                          electricity/covertype/pol cluster we trail on)
  * noisy binary        : high Bayes error (HS should not help, maybe hurt a bit)
  * regression          : RMSE must stay ~flat (HS is aimed at clf Brier)

Reports test Brier / RMSE and fit time across an hs_lambda sweep, averaged over
seeds. This is a directional probe, not a ship decision.
"""
import time
import numpy as np
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

SWEEP = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
SEEDS = [0, 1, 2, 3, 4]


def _high_signal_binary(rng, n=8000, d=12):
    X = rng.normal(size=(n, d))
    # Strong linear signal + a couple of interactions, low label noise.
    logit = (1.6 * X[:, 0] - 1.3 * X[:, 1] + 1.1 * X[:, 2] * X[:, 3]
             + 0.9 * X[:, 4] + 0.04 * rng.normal(size=n))
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return X, y


def _noisy_binary(rng, n=8000, d=12):
    X = rng.normal(size=(n, d))
    logit = 0.5 * X[:, 0] + 0.4 * X[:, 1] + 1.5 * rng.normal(size=n)  # high noise
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return X, y


def _regression(rng, n=8000, d=12):
    X = rng.normal(size=(n, d))
    y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2] * X[:, 3]
         + 0.5 * rng.normal(size=n))
    return X, y


def _run(make, est_cls, metric_name):
    print(f"\n=== {make.__name__}  ({metric_name}) ===")
    print(f"{'hs_lambda':>10} {metric_name:>10} {'fit_s':>8}")
    for hl in SWEEP:
        scores, times = [], []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            X, y = make(rng)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                                  random_state=s)
            m = est_cls(n_estimators=2000, hs_lambda=hl, random_state=s,
                        thread_count=4)
            t0 = time.time()
            m.fit(Xtr, ytr)
            times.append(time.time() - t0)
            if metric_name == "Brier":
                p = m.predict_proba(Xte)[:, 1]
                scores.append(brier_score_loss(yte, p))
            else:
                scores.append(mean_squared_error(yte, m.predict(Xte)) ** 0.5)
        flag = "  <- baseline" if hl == 0.0 else ""
        print(f"{hl:>10.1f} {np.mean(scores):>10.5f} {np.mean(times):>8.2f}{flag}")


if __name__ == "__main__":
    _run(_high_signal_binary, ChimeraBoostClassifier, "Brier")
    _run(_noisy_binary, ChimeraBoostClassifier, "Brier")
    _run(_regression, ChimeraBoostRegressor, "RMSE")
