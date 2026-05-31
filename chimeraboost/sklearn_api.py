"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import numpy as np
from .booster import GradientBoosting, MulticlassBoosting
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


def _fit_temperature(raw, y, multiclass):
    """Learn the scalar T > 0 minimizing validation log loss of sigmoid(raw/T)
    (binary) or softmax(raw/T) (multiclass). Dividing logits by T is monotonic,
    so predictions are unchanged — only their probabilities are recalibrated.
    `y` is the 0/1 label (binary) or the class index (multiclass)."""
    from scipy.optimize import minimize_scalar

    raw = np.asarray(raw, dtype=np.float64)
    if multiclass:
        rows = np.arange(raw.shape[0])

        def loss(T):
            logits = raw / T
            mx = logits.max(axis=1, keepdims=True)
            log_z = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
            return float(np.mean(log_z - logits[rows, y]))
    else:
        def loss(T):
            z = raw / T
            # Stable binary cross-entropy: softplus(z) - y*z.
            return float(np.mean(np.log1p(np.exp(-np.abs(z)))
                                 + np.maximum(z, 0.0) - y * z))

    res = minimize_scalar(loss, bounds=(0.05, 50.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x) if res.success else 1.0


# Parameters that exist only on the sklearn wrappers, not on the core boosters.
_SKLEARN_ONLY = frozenset({"early_stopping", "validation_fraction",
                           "n_ensembles", "ensemble_n_jobs"})


def _fit_bagged(estimator, X, y, cat_features, eval_set, groups, sample_weight):
    """Train ``estimator.n_ensembles`` bootstrap clones and return them as a list.

    Each member is a clone of ``estimator`` with bagging switched off
    (``n_ensembles=None``) and its own seed, fit on a bootstrap resample (drawn
    with replacement, same size as the training set). Because a member is the
    same estimator class, all per-model machinery — binary/multiclass dispatch,
    ``cat_features``, the early-stopping auto-split, temperature scaling — is
    reused unchanged, and ``cat_features``/``sample_weight``/``groups`` forward
    naturally (which a ``sklearn.ensemble.Bagging`` wrapper would not do).

    Members are independent, so they fit across ``ensemble_n_jobs`` processes.
    When that is >1 and ``thread_count`` is unset, numba threads are divided
    among the workers so the members don't oversubscribe the cores.
    """
    from sklearn.base import clone
    from joblib import Parallel, delayed

    X = (np.asarray(X, dtype=object) if cat_features
         else np.asarray(X, dtype=np.float64))
    y = np.asarray(y)
    groups = None if groups is None else np.asarray(groups)
    n = X.shape[0]
    K = int(estimator.n_ensembles)
    n_jobs = int(estimator.ensemble_n_jobs)

    member_threads = estimator.thread_count
    if n_jobs != 1 and member_threads is None:
        import numba
        member_threads = max(1, numba.config.NUMBA_NUM_THREADS // abs(n_jobs))

    seeds = np.random.default_rng(estimator.random_state).integers(
        0, 2**31 - 1, size=K)

    def _fit_one(seed):
        member = clone(estimator).set_params(
            n_ensembles=None, random_state=int(seed), thread_count=member_threads)
        idx = np.random.default_rng(seed).integers(0, n, size=n)  # bootstrap
        wb = None if sample_weight is None else np.asarray(sample_weight)[idx]
        gb = None if groups is None else groups[idx]
        member.fit(X[idx], y[idx], cat_features=cat_features, eval_set=eval_set,
                   groups=gb, sample_weight=wb)
        return member

    return Parallel(n_jobs=n_jobs)(delayed(_fit_one)(s) for s in seeds)


def _make_eval_split(X, y, validation_fraction, random_state,
                     groups=None, stratify=None):
    """Return (train_idx, val_idx) for automatic early-stopping splits.

    Parameters
    ----------
    stratify : array-like or None
        Class labels for stratified splitting (pass for classification tasks).
    groups : array-like or None
        Group membership array (e.g. ``df['subject_id']``).  When supplied,
        groups are kept intact across the split boundary.  For classification,
        ``StratifiedGroupKFold`` is used so class proportions are preserved;
        for regression ``GroupShuffleSplit`` is used.
    """
    from sklearn.model_selection import (
        ShuffleSplit,
        StratifiedShuffleSplit,
        GroupShuffleSplit,
        StratifiedGroupKFold,
    )

    if groups is not None:
        groups = np.asarray(groups)
        if stratify is not None:
            # StratifiedGroupKFold approximates the desired val fraction via
            # n_splits = round(1 / validation_fraction).
            n_splits = max(2, round(1.0 / validation_fraction))
            splitter = StratifiedGroupKFold(n_splits=n_splits)
            train_idx, val_idx = next(
                splitter.split(X, stratify, groups=groups)
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_fraction,
                random_state=random_state,
            )
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    elif stratify is not None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(X, stratify))
    else:
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(splitter.split(X))

    return train_idx, val_idx


class ChimeraBoostRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosted oblivious trees for regression.

    loss: "RMSE" (default), "MAE", or "Quantile". For "Quantile" pass the level
    via `alpha` (e.g. alpha=0.9 for the 90th-percentile predictor).

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when the validation
        score stops improving.  Requires ``early_stopping_rounds`` (defaults
        to 50 when early stopping is active but the param is None).
    validation_fraction : float, default 0.1
        Fraction of training data to hold out as a validation set when
        *early_stopping* is active and no explicit *eval_set* is passed.
        Ignored when an explicit *eval_set* is given to ``fit``.
    n_ensembles : int or None, default None
        Bagging. ``None`` or ``1`` trains a single model. An int >= 2 trains that
        many independent members on bootstrap resamples and averages their
        predictions, which cuts variance and smooths the output (works well with
        *early_stopping*, since each member early-stops on its own bootstrap).
    ensemble_n_jobs : int, default 1
        Processes used to fit ensemble members in parallel (1 = sequential).
        When >1 and *thread_count* is None, numba threads are split among workers.
    """

    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 loss="RMSE", alpha=0.5, min_child_weight=1.0, thread_count=None,
                 random_state=None, verbose=False, ordered_boosting=False,
                 cat_combinations=False,
                 early_stopping=False, validation_fraction=0.1,
                 n_ensembles=None, ensemble_n_jobs=1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.cat_n_permutations = cat_n_permutations
        self.early_stopping_rounds = early_stopping_rounds
        self.loss = loss
        self.alpha = alpha
        self.min_child_weight = min_child_weight
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.cat_combinations = cat_combinations
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_ensembles = n_ensembles
        self.ensemble_n_jobs = ensemble_n_jobs

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or None
            Column indices to treat as categoricals.
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set.  When provided, automatic splitting is
            skipped regardless of the *early_stopping* setting.
        groups : array-like of shape (n_samples,) or None
            Group labels for the samples (e.g. ``df['subject_id']``).  When
            supplied and *early_stopping* triggers an automatic split, groups
            are kept intact across the train/validation boundary using
            ``GroupShuffleSplit``.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.  Only applied
            to the training set; the validation eval metric is always unweighted.
        """
        if self.n_ensembles and self.n_ensembles > 1:
            self.estimators_ = _fit_bagged(self, X, y, cat_features, eval_set,
                                           groups, sample_weight)
            return self
        self.estimators_ = None
        return self._fit_single(X, y, cat_features, eval_set, groups,
                                sample_weight)

    def _fit_single(self, X, y, cat_features, eval_set, groups, sample_weight):
        """Fit one (non-bagged) model on the data as given."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        es_active = bool(self.early_stopping)
        if es_active and eval_set is None:
            train_idx, val_idx = _make_eval_split(
                X, y, self.validation_fraction, self.random_state,
                groups=groups, stratify=None,
            )
            eval_set = (X[val_idx], y[val_idx])
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]

        # If early stopping is active but patience not explicitly set, use 50.
        # 50 beats 10 on 25/34 benchmark datasets (lr=0.1 keeps improving past a
        # 10-round plateau); see benchmarks/investigate_early_stopping.py.
        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 50

        loss_kwargs = {"alpha": self.alpha} if self.loss == "Quantile" else {}
        kw = {k: v for k, v in self.get_params().items()
              if k not in {"loss", "alpha"} | _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds
        self.model_ = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs,
                                       **kw)
        self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                        sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimators_ is not None:
            return np.mean([m.predict(X) for m in self.estimators_], axis=0)
        return self.model_.predict_raw(X)

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        if self.estimators_ is not None:
            raise NotImplementedError("staged_predict is not defined for a "
                                      "bagged ensemble (n_ensembles > 1).")
        yield from self.model_.staged_predict_raw(X)

    @property
    def best_iteration_(self):
        if self.estimators_ is not None:
            return int(round(np.mean([m.best_iteration_ for m in self.estimators_])))
        return self.model_.best_iteration_

    @property
    def feature_importances_(self):
        if self.estimators_ is not None:
            return np.mean([m.feature_importances_ for m in self.estimators_],
                           axis=0)
        return self.model_.feature_importances_


class ChimeraBoostClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosted oblivious trees for classification.

    Automatically uses binary logloss for 2 classes and softmax multiclass for
    3+. `classes_` preserves the original label values.

    early_stopping : bool, default False
        Whether to use early stopping.  The validation split is always
        stratified to preserve class proportions; when *groups* is passed,
        ``StratifiedGroupKFold`` is used instead.
    validation_fraction : float, default 0.1
        Fraction of training data held out for the automatic validation set.
        Ignored when an explicit *eval_set* is given to ``fit``.
    n_ensembles : int or None, default None
        Bagging. ``None`` or ``1`` trains a single model. An int >= 2 trains that
        many independent members on bootstrap resamples and averages their
        (temperature-calibrated) class probabilities, which cuts variance and
        smooths the output.
    ensemble_n_jobs : int, default 1
        Processes used to fit ensemble members in parallel (1 = sequential).
        When >1 and *thread_count* is None, numba threads are split among workers.
    """

    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, cat_n_permutations=4,
                 early_stopping_rounds=None,
                 min_child_weight=1.0, thread_count=None, random_state=None,
                 verbose=False, ordered_boosting=True,
                 cat_combinations=False,
                 early_stopping=False, validation_fraction=0.1,
                 n_ensembles=None, ensemble_n_jobs=1):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.cat_n_permutations = cat_n_permutations
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = min_child_weight
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose
        self.ordered_boosting = ordered_boosting
        self.cat_combinations = cat_combinations
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_ensembles = n_ensembles
        self.ensemble_n_jobs = ensemble_n_jobs

    def fit(self, X, y, cat_features=None, eval_set=None, groups=None,
            sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X, y : array-like
            Training data.
        cat_features : list of int or None
            Column indices to treat as categoricals.
        eval_set : (X_val, y_val) tuple or None
            Explicit validation set with original class labels.  When provided,
            automatic splitting is skipped.
        groups : array-like of shape (n_samples,) or None
            Group labels (e.g. ``df['subject_id']``).  When supplied and early
            stopping triggers an automatic split, ``StratifiedGroupKFold`` keeps
            groups intact and class proportions balanced across the split.
        sample_weight : array-like of shape (n_samples,) or None
            Per-sample weights.  Normalized to mean 1 internally.  Only applied
            to the training set; the validation eval metric is always unweighted.
        """
        if self.n_ensembles and self.n_ensembles > 1:
            # Fix the global class set up front: a member's bootstrap may miss a
            # rare class, and predict_proba aligns each member's columns to this.
            yarr = np.asarray(y)
            self.classes_ = np.unique(yarr)
            self.n_classes_ = self.classes_.size
            if self.n_classes_ < 2:
                raise ValueError("Need at least 2 classes.")
            self._multiclass = self.n_classes_ > 2
            self.estimators_ = _fit_bagged(self, X, yarr, cat_features, eval_set,
                                           groups, sample_weight)
            return self
        self.estimators_ = None
        return self._fit_single(X, y, cat_features, eval_set, groups,
                                sample_weight)

    def _fit_single(self, X, y, cat_features, eval_set, groups, sample_weight):
        """Fit one (non-bagged) classifier on the data as given."""
        X = (np.asarray(X, dtype=object) if cat_features
             else np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        es_active = bool(self.early_stopping)
        if es_active and eval_set is None:
            train_idx, val_idx = _make_eval_split(
                X, y, self.validation_fraction, self.random_state,
                groups=groups, stratify=y,  # always stratify for classification
            )
            eval_set = (X[val_idx], y[val_idx])
            X, y = X[train_idx], y[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.size

        es_rounds = self.early_stopping_rounds
        if es_active and es_rounds is None:
            es_rounds = 50   # see GradientBoosting/Regressor note above

        kw = {k: v for k, v in self.get_params().items()
              if k not in _SKLEARN_ONLY}
        kw["early_stopping_rounds"] = es_rounds

        self._multiclass = self.n_classes_ > 2
        cal_Xv = cal_y = None   # validation set used to calibrate temperature
        if self._multiclass:
            self.model_ = MulticlassBoosting(**kw)
            self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight)
            self.classes_ = self.model_.classes_
            if eval_set is not None:
                cal_Xv = eval_set[0]
                cal_y = np.searchsorted(self.classes_, np.asarray(eval_set[1]))
        else:
            y01 = (y == self.classes_[1]).astype(np.float64)
            if eval_set is not None:
                cal_Xv = eval_set[0]
                cal_y = (np.asarray(eval_set[1]) == self.classes_[1]).astype(np.float64)
                eval_set = (cal_Xv, cal_y)
            self.model_ = GradientBoosting(loss="Logloss", **kw)
            self.model_.fit(X, y01, cat_features=cat_features, eval_set=eval_set,
                            sample_weight=sample_weight)

        # Temperature scaling on the validation set: dividing raw scores by T > 0
        # is monotonic, so predict() is unchanged while predict_proba() becomes
        # better calibrated (lower log loss).
        self.temperature_ = 1.0
        if cal_Xv is not None:
            raw = self.model_.predict_raw(cal_Xv)
            self.temperature_ = _fit_temperature(raw, cal_y, self._multiclass)
        return self

    def predict_proba(self, X):
        if self.estimators_ is not None:
            # Soft-vote: average members' calibrated probabilities, aligning each
            # member's class columns to the global class set (a member whose
            # bootstrap missed a class simply contributes 0 to that column).
            probas = [m.predict_proba(X) for m in self.estimators_]
            acc = np.zeros((probas[0].shape[0], self.n_classes_))
            for m, p in zip(self.estimators_, probas):
                cols = np.searchsorted(self.classes_, m.classes_)
                acc[:, cols] += p
            return acc / len(self.estimators_)
        raw = self.model_.predict_raw(X) / self.temperature_
        if self._multiclass:
            return self.model_.loss_.transform(raw)            # (n, K)
        p1 = self.model_.loss_.transform(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def best_iteration_(self):
        if self.estimators_ is not None:
            return int(round(np.mean([m.best_iteration_ for m in self.estimators_])))
        return self.model_.best_iteration_

    @property
    def feature_importances_(self):
        if self.estimators_ is not None:
            return np.mean([m.feature_importances_ for m in self.estimators_],
                           axis=0)
        return self.model_.feature_importances_