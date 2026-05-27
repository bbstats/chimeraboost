"""Scikit-learn flavored estimators: fit / predict / predict_proba."""

import numpy as np
from .booster import GradientBoosting, MulticlassBoosting
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class ChimeraBoostRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosted oblivious trees for regression.

    loss: "RMSE" (default), "MAE", or "Quantile". For "Quantile" pass the level
    via `alpha` (e.g. alpha=0.9 for the 90th-percentile predictor).
    """

    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, early_stopping_rounds=None,
                 loss="RMSE", alpha=0.5, thread_count=None,
                 random_state=None, verbose=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.early_stopping_rounds = early_stopping_rounds
        self.loss = loss
        self.alpha = alpha
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, cat_features=None, eval_set=None):
        loss_kwargs = {"alpha": self.alpha} if self.loss == "Quantile" else {}
        kw = {k: v for k, v in self.get_params().items()
              if k not in ("loss", "alpha")}
        self.model_ = GradientBoosting(loss=self.loss, loss_kwargs=loss_kwargs,
                                       **kw)
        self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set)
        return self

    def predict(self, X):
        return self.model_.predict_raw(X)

    def staged_predict(self, X):
        """Yield the prediction after each successive tree."""
        yield from self.model_.staged_predict_raw(X)

    def save(self, path):
        save_model(self, path)
        return path

    @staticmethod
    def load(path):
        return load_model(path)

    @property
    def best_iteration_(self):
        return self.model_.best_iteration_

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


class ChimeraBoostClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosted oblivious trees for classification.

    Automatically uses binary logloss for 2 classes and softmax multiclass for
    3+. `classes_` preserves the original label values.
    """

    def __init__(self, iterations=500, learning_rate=None, depth=6,
                 l2_leaf_reg=3.0, max_bins=128, subsample=1.0, colsample=1.0,
                 cat_smoothing=1.0, early_stopping_rounds=None,
                 thread_count=None, random_state=None, verbose=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.cat_smoothing = cat_smoothing
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, cat_features=None, eval_set=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ < 2:
            raise ValueError("Need at least 2 classes.")

        if self.n_classes_ == 2:
            self._multiclass = False
            y01 = (y == self.classes_[1]).astype(np.float64)
            if eval_set is not None:
                Xv, yv = eval_set
                eval_set = (Xv, (np.asarray(yv) == self.classes_[1]).astype(np.float64))
            self.model_ = GradientBoosting(loss="Logloss", **self.get_params())
            self.model_.fit(X, y01, cat_features=cat_features, eval_set=eval_set)
        else:
            self._multiclass = True
            self.model_ = MulticlassBoosting(**self.get_params())
            self.model_.fit(X, y, cat_features=cat_features, eval_set=eval_set)
            self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        raw = self.model_.predict_raw(X)
        if self._multiclass:
            return self.model_.loss_.transform(raw)            # (n, K)
        p1 = self.model_.loss_.transform(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def best_iteration_(self):
        return self.model_.best_iteration_

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    def save(self, path):
        save_model(self, path)
        return path

    @staticmethod
    def load(path):
        return load_model(path)


def save_model(estimator, path):
    """Persist a fitted ChimeraBoost estimator to disk (pickle)."""
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(estimator, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    """Load a ChimeraBoost estimator saved with save_model / .save()."""
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
