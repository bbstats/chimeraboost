"""Missing-value handling arms for the MISSING_PLAN bake-off (step 2).

Each arm is a transform fitted on the TRAINING rows only and then applied to
train and test; the default ChimeraBoost fit runs on the result. Benchmark-only:
nothing here touches the library until an arm wins.

Design -- augment, never replace. Every arm keeps the original columns exactly
as they are (NaN included, so ChimeraBoost's top-bin routing survives) and
appends ONE filled copy per NaN-bearing column. Arms therefore all have the same
width and differ only in where the copy puts the missing rows. Replacing the
column instead would throw away the single-split "value > t or missing" route,
which rigged the first design against imputation under MNAR. No indicator
columns: the original column's top-bin split already isolates missing rows (an
appended indicator measured an exact tie with the baseline).

Shared rules:
  * Only numeric columns with NaN in the training rows (and at least one
    observed value) get a copy. Categoricals pass through -- their NaN is
    already "__nan__". All-missing numeric columns are left out entirely.
  * No such column => exact pass-through (same array object), so NaN-free
    data ties the baseline exactly.
  * No arm reads y: test rows have none, so using it would leak.
  * Copies are appended at the END, so categorical indices stay valid.

Arms (ARMS maps name -> class); the copy holds, for missing rows:
  ind         1.0 (the copy is a 0/1 missing indicator; redundant, kept only
              to reproduce the tie)
  mia         a value just below the training minimum. The original sends
              missing rows high at every split, the copy sends them low, so
              each tree level can choose -- XGBoost-style learned direction,
              emulated without a kernel change.
  mean        the training mean
  missforest  IterativeImputer(RandomForest), missForest-style at reduced
              cost (RF_TREES trees, <= RF_MAX_SAMPLES rows per tree)
  miceforest  miceforest (LightGBM MICE, optional dependency). 6.0.5 breaks
              on LightGBM >= 4.6 -- see miceforest_status().
  cbmice      IterativeImputer with a lean ChimeraBoost column model
  masked      ONE ChimeraBoost regressor for every column: trained on
              (row with its natural gaps and the target cell hidden, one-hot
              "which column") -> the standardised target value. One model
              instead of p models x k sweeps, NaN-native in its inputs.
"""
import re

import numpy as np

MASKED_MAX_PAIRS = 200_000
MASKED_CELL_BUDGET = 25_000_000   # cap on any design matrix the masked arm builds
MASKED_MIN_PAIRS = 1_000
RF_TREES = 50
RF_MAX_SAMPLES = 10_000


def _numeric_cols(X, cat):
    cat_set = set(cat or ())
    return [j for j in range(X.shape[1]) if j not in cat_set]


def _as_float(X, cols):
    return np.asarray(X[:, cols], dtype=float)


def miceforest_status():
    """(usable, reason). miceforest 6.0.5 calls a private LightGBM method whose
    signature changed in 4.6, so every fit raises there -- and the harness would
    record each one as a silent skip."""
    try:
        import miceforest  # noqa: F401
    except ImportError:
        return False, "miceforest is not installed"
    try:
        import lightgbm
    except ImportError:
        return False, "lightgbm is not installed"
    ver = tuple(int(p) for p in re.findall(r"\d+", lightgbm.__version__)[:2])
    if ver >= (4, 6):
        return False, (f"miceforest needs lightgbm<4.6 (found "
                       f"{lightgbm.__version__}); run CBMiceForest from a "
                       f"separate environment -- see MISSING_PLAN.md")
    return True, ""


class _Arm:
    """fit_transform(Xtr, cat) -> Xtr'; transform(Xte) -> Xte'."""

    def __init__(self, threads=1, random_state=0):
        self.threads = threads
        self.random_state = random_state

    # -- template ---------------------------------------------------------
    def fit_transform(self, X, cat):
        num = _numeric_cols(X, cat)
        F = _as_float(X, num)
        observed = ~np.isnan(F).all(axis=0)
        # Usable numeric columns (at least one observed value), in X indices.
        self.cols_ = [num[c] for c in np.flatnonzero(observed)]
        F = F[:, observed]
        # Columns that get a copy, as indices into the usable block.
        self.nan_local_ = np.flatnonzero(np.isnan(F).any(axis=0))
        if not len(self.nan_local_):
            return X
        self.mean_ = np.nanmean(F, axis=0)
        self._fit(F)
        return self._apply(X)

    def transform(self, X):
        if not len(self.nan_local_):
            return X
        return self._apply(X)

    def _apply(self, X):
        return np.hstack([X, self._fill(_as_float(X, self.cols_))])

    def _predictors(self, F):
        """F with NaN in columns that were complete at fit time mean-filled,
        so imputers never meet a missing pattern they were not fitted on."""
        F = F.copy()
        other = np.setdiff1d(np.arange(F.shape[1]), self.nan_local_)
        if len(other):
            sub = F[:, other]
            bad = np.isnan(sub)
            if bad.any():
                sub[bad] = np.take(self.mean_[other], np.nonzero(bad)[1])
                F[:, other] = sub
        return F

    # -- hooks ------------------------------------------------------------
    def _fit(self, F):
        pass

    def _fill(self, F):
        """(n, len(nan_local_)) NaN-free copy of the NaN-bearing columns."""
        raise NotImplementedError


class IndicatorArm(_Arm):
    def _fill(self, F):
        return np.isnan(F[:, self.nan_local_]).astype(float)


class MIAArm(_Arm):
    def _fit(self, F):
        # Just below the minimum: tree splits only see the order, and a value
        # far below would distort linear leaves and cross features. The
        # binner's greedy borders give the point mass its own bin.
        self.low_ = np.nextafter(np.nanmin(F[:, self.nan_local_], axis=0),
                                 -np.inf)

    def _fill(self, F):
        G = F[:, self.nan_local_]
        return np.where(np.isnan(G), self.low_, G)


class MeanArm(_Arm):
    def _fill(self, F):
        G = F[:, self.nan_local_]
        return np.where(np.isnan(G), self.mean_[self.nan_local_], G)


class _IterativeArm(_Arm):
    """sklearn IterativeImputer over all usable numeric columns."""

    max_iter = 5

    def _estimator(self, n):
        raise NotImplementedError

    def _fit(self, F):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        self.imp_ = IterativeImputer(
            estimator=self._estimator(len(F)), max_iter=self.max_iter,
            initial_strategy="mean", skip_complete=True,
            random_state=self.random_state)
        self.imp_.fit(self._predictors(F))

    def _fill(self, F):
        full = self.imp_.transform(self._predictors(F))
        return full[:, self.nan_local_]


class MissForestArm(_IterativeArm):
    def _estimator(self, n):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=RF_TREES, max_features=0.33,
            max_samples=RF_MAX_SAMPLES if n > RF_MAX_SAMPLES else None,
            n_jobs=self.threads, random_state=self.random_state)


def _lean_chimera(threads, random_state, n_estimators=100):
    """A fast fixed-config regressor for use INSIDE an imputer."""
    from chimeraboost import ChimeraBoostRegressor
    return ChimeraBoostRegressor(
        n_estimators=n_estimators, learning_rate=0.1, depth=6,
        early_stopping=False, linear_leaves=False, cross_features=False,
        refit_full=False, thread_count=threads, random_state=random_state)


class CBMiceArm(_IterativeArm):
    max_iter = 3

    def _estimator(self, n):
        return _lean_chimera(self.threads, self.random_state)


class MiceForestArm(_Arm):
    iterations = 3

    def _frame(self, F):
        import pandas as pd
        return pd.DataFrame(self._predictors(F),
                            columns=[f"c{j}" for j in range(F.shape[1])])

    def _fit(self, F):
        ok, reason = miceforest_status()
        if not ok:
            raise RuntimeError(reason)
        import miceforest as mf
        # mean_match_candidates=0: plain model predictions. The default (5,
        # predictive mean matching) draws a random donor per cell -- right for
        # multiple imputation, but noise for a point fill feeding a predictor;
        # it measured worse (fill RMSE 0.47 vs 0.37 on a planted relation).
        # This is miceforest's strongest setting for our goal.
        self.kernel_ = mf.ImputationKernel(
            self._frame(F), num_datasets=1, mean_match_candidates=0,
            random_state=self.random_state)
        self.kernel_.mice(self.iterations, num_threads=self.threads, verbose=False)
        self.train_fill_ = self.kernel_.complete_data(
            0, iteration=self.iterations).to_numpy(float)
        self.fit_mask_ = np.isnan(F)

    def _fill(self, F):
        # The kernel already holds the training rows' imputations; re-imputing
        # them as "new data" would draw a second, different completion.
        if F.shape == self.fit_mask_.shape and np.array_equal(np.isnan(F),
                                                               self.fit_mask_):
            full = self.train_fill_
        else:
            # iteration must be explicit: miceforest 6.0.5's default (-1) on
            # new data returns iteration 0, the random initial fill -- fill
            # RMSE 3.16 vs 0.37 on a planted relation.
            full = self.kernel_.impute_new_data(
                self._frame(F), iterations=self.iterations,
                random_state=self.random_state, verbose=False,
            ).complete_data(0, iteration=self.iterations).to_numpy(float)
        return full[:, self.nan_local_]


class MaskedArm(_Arm):
    """One regressor for all NaN-bearing columns.

    Training pairs are observed cells of those columns. The input is the cell's
    row, standardised, with ONLY the target cell hidden -- the row keeps its
    natural gaps, which is exactly the context the model meets when it imputes
    (under MCAR identical in distribution; under MAR the missingness carries no
    extra information about the target given the observed values). Extra random
    masking was tried first and rejected: it hid columns that are never missing
    at use time -- under MAR, the very columns that explain the gaps -- and
    pushed training contexts to ~51% missing against ~30% at use time.
    """

    n_estimators = 300

    def _width(self):
        return len(self.mu_) + len(self.nan_local_)

    def _design(self, Z, cols):
        """Rows of Z (standardised, with NaN) + one-hot of the target column."""
        onehot = np.zeros((len(Z), len(self.nan_local_)))
        onehot[np.arange(len(Z)), cols] = 1.0
        return np.hstack([Z, onehot])

    def _fit(self, F):
        rng = np.random.default_rng(self.random_state)
        self.mu_ = self.mean_
        sd = np.nanstd(F, axis=0)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        Z = (F - self.mu_) / self.sd_
        G = Z[:, self.nan_local_]
        rows, cols = np.nonzero(~np.isnan(G))        # observed target cells
        cap = min(MASKED_MAX_PAIRS,
                  max(MASKED_MIN_PAIRS, MASKED_CELL_BUDGET // self._width()))
        if len(rows) > cap:
            pick = rng.choice(len(rows), cap, replace=False)
            rows, cols = rows[pick], cols[pick]
        Xin = Z[rows]                                # fancy index -> a copy
        target = self.nan_local_[cols]
        y = Xin[np.arange(len(rows)), target].copy()
        Xin[np.arange(len(rows)), target] = np.nan
        self.model_ = _lean_chimera(self.threads, self.random_state,
                                    n_estimators=self.n_estimators)
        self.model_.fit(self._design(Xin, cols), y)

    def _predict_cells(self, Z, rows, cols, chunk_rows=None):
        """Model output for cells (rows, cols), built and predicted in chunks
        so the design matrix never exceeds MASKED_CELL_BUDGET."""
        if chunk_rows is None:
            chunk_rows = max(1, MASKED_CELL_BUDGET // self._width())
        out = np.empty(len(rows))
        for s in range(0, len(rows), chunk_rows):
            r, c = rows[s:s + chunk_rows], cols[s:s + chunk_rows]
            out[s:s + chunk_rows] = self.model_.predict(self._design(Z[r], c))
        return out

    def _fill(self, F):
        Z = (F - self.mu_) / self.sd_
        G = F[:, self.nan_local_].copy()
        rows, cols = np.nonzero(np.isnan(G))
        if len(rows):
            gc = self.nan_local_[cols]
            G[rows, cols] = (self._predict_cells(Z, rows, cols) * self.sd_[gc]
                             + self.mu_[gc])
        return G


ARMS = {
    "ind": IndicatorArm,
    "mia": MIAArm,
    "mean": MeanArm,
    "missforest": MissForestArm,
    "miceforest": MiceForestArm,
    "cbmice": CBMiceArm,
    "masked": MaskedArm,
}
