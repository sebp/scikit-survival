from abc import ABCMeta, abstractmethod
from functools import partial
import threading
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (
    BaseForest,
    _accumulate_prediction,
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from sklearn.tree._tree import DTYPE
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import SurvivalAnalysisMixin
from ..metrics import concordance_index_censored
from ..tree import ExtraSurvivalTree, SurvivalTree
from ..tree._criterion import get_unique_times
from ..tree.tree import _array_to_step_function
from ..util import check_array_survival

__all__ = ["RandomSurvivalForest", "ExtraSurvivalTrees"]

MAX_INT = np.iinfo(np.int32).max


def _more_tags_patch(self):
    # BaseForest._more_tags calls
    # type(self.estimator)(criterion=self.criterions),
    # which is incompatible with LogrankCriterion
    if isinstance(self, _BaseSurvivalForest):
        estimator = type(self.estimator)()
    else:
        estimator = type(self.estimator)(criterion=self.criterion)
    return {"allow_nan": _safe_tags(estimator, key="allow_nan")}


BaseForest._more_tags = _more_tags_patch


class _BaseSurvivalForest(BaseForest, metaclass=ABCMeta):
    """
    Base class for forest-based estimators for survival analysis.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=None,
            max_samples=max_samples,
        )

    @property
    def feature_importances_(self):
        """Not implemented"""
        raise NotImplementedError()

    def fit(self, X, y, sample_weight=None):
        """Build a forest of survival trees from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        self._validate_params()

        X = self._validate_data(X, dtype=DTYPE, accept_sparse="csc", ensure_min_samples=2, force_all_finite=False)
        event, time = check_array_survival(X, y)

        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values.
        estimator = type(self.estimator)()
        missing_values_in_feature_mask = estimator._compute_missing_values_in_feature_mask(
            X, estimator_name=self.__class__.__name__
        )

        self.n_features_in_ = X.shape[1]
        time = time.astype(np.float64)
        self.unique_times_, self.is_event_time_ = get_unique_times(time, event)
        self.n_outputs_ = self.unique_times_.shape[0]

        y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
        y_numeric[:, 0] = time
        y_numeric[:, 1] = event.astype(np.float64)

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples=X.shape[0], max_samples=self.max_samples)

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                f"n_estimators={self.n_estimators} must be larger or equal to "
                f"len(estimators_)={len(self.estimators_)} when warm_start==True"
            )

        if n_more_estimators == 0:
            warnings.warn("Warm-start fitting without increasing n_estimators does not fit new trees.", stacklevel=2)
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False, random_state=random_state) for i in range(n_more_estimators)]

            y_tree = (
                y_numeric,
                self.unique_times_,
                self.is_event_time_,
            )
            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer="threads")(
                delayed(_parallel_build_trees)(
                    t,
                    self.bootstrap,
                    X,
                    y_tree,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score_and_attributes(X, (event, time))

        return self

    def _set_oob_score_and_attributes(self, X, y):
        """Calculate out of bag predictions and score."""
        n_samples = X.shape[0]
        event, time = y

        predictions = np.zeros(n_samples)
        n_predictions = np.zeros(n_samples)

        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)

        for estimator in self.estimators_:
            unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
            p_estimator = estimator.predict(X[unsampled_indices, :], check_input=False)

            predictions[unsampled_indices] += p_estimator
            n_predictions[unsampled_indices] += 1

        if (n_predictions == 0).any():
            warnings.warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few trees were used "
                "to compute any reliable oob estimates.",
                stacklevel=3,
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        self.oob_score_ = concordance_index_censored(event, time, predictions)[0]

    def _predict(self, predict_fn, X):
        check_is_fitted(self, "estimators_")
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if predict_fn == "predict":
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        def _get_fn(est, name):
            fn = getattr(est, name)
            if name in ("predict_cumulative_hazard_function", "predict_survival_function"):
                fn = partial(fn, return_array=True)
            return fn

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(_get_fn(e, predict_fn), X, [y_hat], lock) for e in self.estimators_
        )

        y_hat /= len(self.estimators_)

        return y_hat

    def predict(self, X):
        """Predict risk score.

        The ensemble risk score is the total number of events,
        which can be estimated by the sum of the estimated
        ensemble cumulative hazard function :math:`\\hat{H}_e`.

        .. math::

            \\sum_{j=1}^{n} \\hat{H}_e(T_{j} \\mid x) ,

        where :math:`n` denotes the total number of distinct
        event times in the training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        risk_scores : ndarray, shape = (n_samples,)
            Predicted risk scores.
        """
        return self._predict("predict", X)

    def predict_cumulative_hazard_function(self, X, return_array=False):
        arr = self._predict("predict_cumulative_hazard_function", X)
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)

    def predict_survival_function(self, X, return_array=False):
        arr = self._predict("predict_survival_function", X)
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)


class RandomSurvivalForest(SurvivalAnalysisMixin, _BaseSurvivalForest):
    """A random survival forest.

    A random survival forest is a meta estimator that fits a number of
    survival trees on various sub-samples of the dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample
    size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    In each survival tree, the quality of a split is measured by the
    log-rank splitting rule.

    See the :ref:`User Guide </user_guide/random-survival-forest.ipynb>`,
    [1]_ and [2]_ for further description.

    Parameters
    ----------
    n_estimators : integer, optional, default: 100
        The number of trees in the forest.

    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional, default: None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, optional, default: None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    low_memory : boolean, default: False
        If set, ``predict`` computations use reduced memory but ``predict_cumulative_hazard_function``
        and ``predict_survival_function`` are not implemented.

    Attributes
    ----------
    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.

    See also
    --------
    sksurv.tree.SurvivalTree
        A single survival tree.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    Compared to scikit-learn's random forest models, :class:`RandomSurvivalForest`
    currently does not support controlling the depth of a tree based on the log-rank
    test statistics or it's associated p-value, i.e., the parameters
    `min_impurity_decrease` or `min_impurity_split` are absent.
    In addition, the `feature_importances_` attribute is not available.
    It is recommended to estimate feature importances via
    `permutation-based methods <https://eli5.readthedocs.io>`_.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behavior during
    fitting, ``random_state`` has to be fixed.

    References
    ----------
    .. [1] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841–860.

    .. [2] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25–31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    _parameter_constraints = {
        **BaseForest._parameter_constraints,
        **SurvivalTree._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        low_memory=False,
    ):
        super().__init__(
            estimator=SurvivalTree(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
                "low_memory",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.low_memory = low_memory

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazard function.

        For each tree in the ensemble, the cumulative hazard
        function (CHF) for an individual with feature vector
        :math:`x` is computed from all samples of the bootstrap
        sample that are in the same terminal node as :math:`x`.
        It is estimated by the Nelson–Aalen estimator.
        The ensemble CHF at time :math:`t` is the average
        value across all trees in the ensemble at the
        specified time point.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of length `n_samples`
            of :class:`sksurv.functions.StepFunction` instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.ensemble import RandomSurvivalForest

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = RandomSurvivalForest().fit(X, y)

        Estimate the cumulative hazard function for the first 5 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:5])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        return super().predict_cumulative_hazard_function(X, return_array)

    def predict_survival_function(self, X, return_array=False):
        """Predict survival function.

        For each tree in the ensemble, the survival function
        for an individual with feature vector :math:`x` is
        computed from all samples of the bootstrap sample that
        are in the same terminal node as :math:`x`.
        It is estimated by the Kaplan-Meier estimator.
        The ensemble survival function at time :math:`t` is
        the average value across all trees in the ensemble at
        the specified time point.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : boolean
            If set, return an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`
            will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.ensemble import RandomSurvivalForest

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = RandomSurvivalForest().fit(X, y)

        Estimate the survival function for the first 5 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:5])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        return super().predict_survival_function(X, return_array)


class ExtraSurvivalTrees(SurvivalAnalysisMixin, _BaseSurvivalForest):
    """An extremely random survival forest.

    This class implements a meta estimator that fits a number of randomized
    survival trees (a.k.a. extra-trees) on various sub-samples of the dataset
    and uses averaging to improve the predictive accuracy and control
    over-fitting. The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    In each randomized survival tree, the quality of a split is measured by
    the log-rank splitting rule.

    Compared to :class:`RandomSurvivalForest`, randomness goes one step
    further in the way splits are computed. As in
    :class:`RandomSurvivalForest`, a random subset of candidate features is
    used, but instead of looking for the most discriminative thresholds,
    thresholds are drawn at random for each candidate feature and the best of
    these randomly-generated thresholds is picked as the splitting rule.

    Parameters
    ----------
    n_estimators : integer, optional, default: 100
        The number of trees in the forest.

    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional, default: None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None, optional, default: None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    low_memory : boolean, default: False
        If set, ``predict`` computations use reduced memory but ``predict_cumulative_hazard_function``
        and ``predict_survival_function`` are not implemented.

    Attributes
    ----------
    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    n_features_in_ : int
        The number of features when ``fit`` is performed.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.

    See also
    --------
    sksurv.tree.SurvivalTree
        A single survival tree.
    """

    _parameter_constraints = {
        **BaseForest._parameter_constraints,
        **SurvivalTree._parameter_constraints,
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
        low_memory=False,
    ):
        super().__init__(
            estimator=ExtraSurvivalTree(),
            n_estimators=n_estimators,
            estimator_params=(
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "random_state",
                "low_memory",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.low_memory = low_memory

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazard function.

        For each tree in the ensemble, the cumulative hazard
        function (CHF) for an individual with feature vector
        :math:`x` is computed from all samples of the bootstrap
        sample that are in the same terminal node as :math:`x`.
        It is estimated by the Nelson–Aalen estimator.
        The ensemble CHF at time :math:`t` is the average
        value across all trees in the ensemble at the
        specified time point.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of length `n_samples`
            of :class:`sksurv.functions.StepFunction` instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.ensemble import ExtraSurvivalTrees

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = ExtraSurvivalTrees().fit(X, y)

        Estimate the cumulative hazard function for the first 5 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:5])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        return super().predict_cumulative_hazard_function(X, return_array)

    def predict_survival_function(self, X, return_array=False):
        """Predict survival function.

        For each tree in the ensemble, the survival function
        for an individual with feature vector :math:`x` is
        computed from all samples of the bootstrap sample that
        are in the same terminal node as :math:`x`.
        It is estimated by the Kaplan-Meier estimator.
        The ensemble survival function at time :math:`t` is
        the average value across all trees in the ensemble at
        the specified time point.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : boolean, default: False
            If set, return an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability of
            survival for each `self.unique_times_`, otherwise an array of
            length `n_samples` of :class:`sksurv.functions.StepFunction`
            instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.ensemble import ExtraSurvivalTrees

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = ExtraSurvivalTrees().fit(X, y)

        Estimate the survival function for the first 5 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:5])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        return super().predict_survival_function(X, return_array)
