"""Missing-value handling arms for the MISSING_PLAN bake-off (step 2).

Each arm is a transform fitted on the TRAINING rows only and then applied to
train and test; the default ChimeraBoost fit runs on the result. Benchmark-only:
nothing here touches the library until an arm wins.

Shared rules:
  * Only numeric columns holding NaN in the training rows are touched ("nan
    columns"). Categoricals pass through -- their NaN is already "__nan__".
  * No NaN in the training rows => exact pass-through (same array object), so
    the plain strata tie the baseline exactly.
  * Every imputation arm appends missing-indicator columns (Perez-Lebel 2022;
    Le Morvan & Varoquaux 2025: indicators help even under MCAR).
  * New columns are appended at the END, so categorical indices stay valid.

Arms (ARMS maps name -> class):
  ind      one 0/1 "was missing" column per nan column
  mia      + indicators + a mirrored copy of each nan column with NaN set below
           the training minimum. The binner puts NaN in the TOP bin of the
           original; the copy puts it at the bottom, so each split can choose
           "missing goes high" or "missing goes low" -- XGBoost-style learned
           direction, emulated without a kernel change.
  mean     mean fill + indicators
  missforest  IterativeImputer(RandomForest) + indicators
  miceforest  miceforest (LightGBM MICE, optional dependency) + indicators.
           miceforest 6.0.5 breaks on LightGBM >= 4.6; it needs lightgbm<4.6,
           which would also move the LightGBM competitor -- install it in a
           separate environment for the bake-off.
  cbmice   IterativeImputer with a lean ChimeraBoost column model + indicators
  masked   ONE ChimeraBoost regressor for every column, trained on randomly
           re-masked copies of the training rows to predict a held-out cell
           from (its row with NaNs, one-hot "which column"). One model instead
           of p models x k sweeps, and it learns from masked context the way
           it is used at imputation time.
"""
import numpy as np

MASKED_MAX_PAIRS = 200_000
RF_TREES = 50
RF_MAX_SAMPLES = 10_000


def _numeric_cols(X, cat):
    cat_set = set(cat or ())
    return [j for j in range(X.shape[1]) if j not in cat_set]


def _as_float(X, cols):
    return np.asarray(X[:, cols], dtype=float)


class _Arm:
    """fit_transform(Xtr, cat) -> Xtr'; transform(Xte) -> Xte'."""

    indicators = True

    def __init__(self, threads=1, random_state=0):
        self.threads = threads
        self.random_state = random_state

    # -- template ---------------------------------------------------------
    def fit_transform(self, X, cat):
        num = _numeric_cols(X, cat)
        F = _as_float(X, num)
        has = np.isnan(F).any(axis=0)
        self.num_ = num
        self.nan_local_ = np.flatnonzero(has)            # into F
        self.nan_cols_ = [num[c] for c in self.nan_local_]  # into X
        if not self.nan_cols_:
            return X
        self.mean_ = np.nanmean(F, axis=0)
        self.mean_[np.isnan(self.mean_)] = 0.0
        self._fit(F)
        return self._apply(X)

    def transform(self, X):
        if not self.nan_cols_:
            return X
        return self._apply(X)

    def _apply(self, X):
        F = _as_float(X, self.num_)
        miss = np.isnan(F[:, self.nan_local_])
        parts = [self._replace(X, F)]
        if self.indicators:
            parts.append(miss.astype(float))
        extra = self._extra(F)
        if extra is not None:
            parts.append(extra)
        return np.hstack(parts)

    def _replace(self, X, F):
        """X with nan columns filled (default: unchanged)."""
        filled = self._fill(F)
        if filled is None:
            return X
        X = X.copy()
        X[:, self.nan_cols_] = filled
        return X

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
        return None

    def _extra(self, F):
        return None


class IndicatorArm(_Arm):
    pass


class MIAArm(_Arm):
    def _fit(self, F):
        G = F[:, self.nan_local_]
        lo = np.nanmin(G, axis=0)
        hi = np.nanmax(G, axis=0)
        span = np.where(np.isfinite(hi - lo) & (hi > lo), hi - lo, 1.0)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        self.low_ = lo - span          # clearly below every training value

    def _extra(self, F):
        G = F[:, self.nan_local_]
        return np.where(np.isnan(G), self.low_, G)


class MeanArm(_Arm):
    def _fill(self, F):
        G = F[:, self.nan_local_]
        return np.where(np.isnan(G), self.mean_[self.nan_local_], G)


class _IterativeArm(_Arm):
    """sklearn IterativeImputer over all numeric columns."""

    max_iter = 5

    def _estimator(self, n):
        raise NotImplementedError

    def _fit(self, F):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        self.imp_ = IterativeImputer(
            estimator=self._estimator(len(F)), max_iter=self.max_iter,
            initial_strategy="mean", skip_complete=True,
            random_state=self.random_state, keep_empty_features=True)
        self.imp_.fit(self._predictors(F))

    def _fill(self, F):
        full = self.imp_.transform(self._predictors(F))
        return full[:, self.nan_local_]


class MissForestArm(_IterativeArm):
    def _estimator(self, n):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=RF_TREES, max_features=0.33,
            max_samples=min(n, RF_MAX_SAMPLES) if n > RF_MAX_SAMPLES else None,
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
        import miceforest as mf
        df = self._frame(F)
        # mean_match_candidates=0: plain model predictions. The default (5,
        # predictive mean matching) draws a random donor per cell -- right for
        # multiple imputation, but noise for a point fill feeding a predictor;
        # it measured worse (fill RMSE 0.47 vs 0.37 on a planted relation).
        # This is miceforest's strongest setting for our goal.
        self.kernel_ = mf.ImputationKernel(
            df, num_datasets=1, mean_match_candidates=0,
            random_state=self.random_state)
        self.kernel_.mice(self.iterations, num_threads=self.threads, verbose=False)
        self.train_fill_ = self.kernel_.complete_data(
            0, iteration=self.iterations).to_numpy(float)
        self.n_fit_ = len(F)
        self.fit_mask_ = np.isnan(F)

    def _fill(self, F):
        # The kernel already holds the training rows' imputations; re-imputing
        # them as "new data" would draw a second, different completion.
        if len(F) == self.n_fit_ and np.array_equal(np.isnan(F), self.fit_mask_):
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
    """One regressor for all nan columns, trained on re-masked rows."""

    n_estimators = 300

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
        if len(rows) > MASKED_MAX_PAIRS:
            pick = rng.choice(len(rows), MASKED_MAX_PAIRS, replace=False)
            rows, cols = rows[pick], cols[pick]
        rate = float(np.isnan(G).mean())             # re-mask at the real rate
        Xin = Z[rows].copy()
        Xin[rng.random(Xin.shape) < rate] = np.nan
        target_global = self.nan_local_[cols]
        y = Z[rows, target_global]
        Xin[np.arange(len(rows)), target_global] = np.nan
        self.model_ = _lean_chimera(self.threads, self.random_state,
                                    n_estimators=self.n_estimators)
        self.model_.fit(self._design(Xin, cols), y)

    def _fill(self, F):
        Z = (F - self.mu_) / self.sd_
        G = F[:, self.nan_local_].copy()
        rows, cols = np.nonzero(np.isnan(G))
        if len(rows):
            pred = self.model_.predict(self._design(Z[rows], cols))
            gc = self.nan_local_[cols]
            G[rows, cols] = pred * self.sd_[gc] + self.mu_[gc]
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
