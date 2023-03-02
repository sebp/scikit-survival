from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.tree import _tree
from sklearn.tree._classes import DENSE_SPLITTERS, SPARSE_SPLITTERS
from sklearn.tree._splitter import Splitter
from sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder, Tree
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_random_state

from ..base import SurvivalAnalysisMixin
from ..functions import StepFunction
from ..util import check_array_survival
from ._criterion import LogrankCriterion

__all__ = ["SurvivalTree"]

DTYPE = _tree.DTYPE


def _array_to_step_function(x, array):
    n_samples = array.shape[0]
    funcs = np.empty(n_samples, dtype=np.object_)
    for i in range(n_samples):
        funcs[i] = StepFunction(x=x,
                                y=array[i])
    return funcs


class SurvivalTree(BaseEstimator, SurvivalAnalysisMixin):
    """A survival tree.

    The quality of a split is measured by the
    log-rank splitting rule.

    See [1]_, [2]_ and [3]_ for further description.

    Parameters
    ----------
    splitter : {'best', 'random'}, default: 'best'
        The strategy used to choose the split at each node. Supported
        strategies are 'best' to choose the best split and 'random' to choose
        the best random split.

    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.

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
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional, default: None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    Attributes
    ----------
    event_times_ : array of shape = (n_event_times,)
        Unique time points where events occurred.

    max_features_ : int,
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object.

    See also
    --------
    sksurv.ensemble.RandomSurvivalForest
        An ensemble of SurvivalTrees.

    References
    ----------
    .. [1] Leblanc, M., & Crowley, J. (1993). Survival Trees by Goodness of Split.
           Journal of the American Statistical Association, 88(422), 457–467.

    .. [2] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841–860.

    .. [3] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25–31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    _parameter_constraints = {
        "splitter": [StrOptions({"best", "random"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="neither"),
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 0.5, closed="right"),
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
            StrOptions({"auto", "sqrt", "log2"}, deprecated={"auto"}),
            None,
        ],
        "random_state": ["random_state"],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
    }

    def __init__(self,
                 *,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a survival tree from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self
        """
        random_state = check_random_state(self.random_state)

        if check_input:
            X = self._validate_data(X, ensure_min_samples=2, accept_sparse="csc")
            event, time = check_array_survival(X, y)
            time = time.astype(np.float64)
            self.event_times_ = np.unique(time[event])
            if issparse(X):
                X.sort_indices()

            y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
            y_numeric[:, 0] = time
            y_numeric[:, 1] = event.astype(np.float64)
        else:
            y_numeric, self.event_times_ = y

        n_samples, self.n_features_in_ = X.shape
        params = self._check_params(n_samples)

        self.n_outputs_ = self.event_times_.shape[0]
        # one "class" for CHF, one for survival function
        self.n_classes_ = np.ones(self.n_outputs_, dtype=np.intp) * 2

        # Build tree
        criterion = LogrankCriterion(self.n_outputs_, n_samples, self.event_times_)

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                params["min_samples_leaf"],
                params["min_weight_leaf"],
                random_state)

        self.tree_ = Tree(self.n_features_in_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if params["max_leaf_nodes"] < 0:
            builder = DepthFirstTreeBuilder(
                splitter,
                params["min_samples_split"],
                params["min_samples_leaf"],
                params["min_weight_leaf"],
                params["max_depth"],
                0.0,  # min_impurity_decrease
            )
        else:
            builder = BestFirstTreeBuilder(
                splitter,
                params["min_samples_split"],
                params["min_samples_leaf"],
                params["min_weight_leaf"],
                params["max_depth"],
                params["max_leaf_nodes"],
                0.0,  # min_impurity_decrease
            )

        builder.build(self.tree_, X, y_numeric, sample_weight)

        return self

    def _check_params(self, n_samples):
        self._validate_params()

        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)

        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)

        if isinstance(self.min_samples_leaf, (Integral, np.integer)):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        self._check_max_features()

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")

        min_weight_leaf = self.min_weight_fraction_leaf * n_samples

        return {
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "min_weight_leaf": min_weight_leaf,
        }

    def _check_max_features(self):
        if isinstance(self.max_features, str):
            if self.max_features in ("auto", "sqrt"):
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))

        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, (Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        if not 0 < max_features <= self.n_features_in_:
            raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

    def _validate_X_predict(self, X, check_input, accept_sparse="csr"):
        """Validate X whenever one tries to predict"""
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, accept_sparse=accept_sparse, reset=False)
        else:
            # The number of features is checked regardless of `check_input`
            self._check_n_features(X, reset=False)

        return X

    def predict(self, X, check_input=True):
        """Predict risk score.

        The risk score is the total number of events, which can
        be estimated by the sum of the estimated cumulative
        hazard function :math:`\\hat{H}_h` in terminal node :math:`h`.

        .. math::

            \\sum_{j=1}^{n(h)} \\hat{H}_h(T_{j} \\mid x) ,

        where :math:`n(h)` denotes the number of distinct event times
        of samples belonging to the same terminal node as :math:`x`.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        risk_scores : ndarray, shape = (n_samples,)
            Predicted risk scores.
        """
        chf = self.predict_cumulative_hazard_function(X, check_input, return_array=True)
        return chf.sum(1)

    def predict_cumulative_hazard_function(self, X, check_input=True, return_array=False):
        """Predict cumulative hazard function.

        The cumulative hazard function (CHF) for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Nelson–Aalen estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.event_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.event_times_`, otherwise an array of length `n_samples`
            of :class:`sksurv.functions.StepFunction` instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.tree import SurvivalTree

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = SurvivalTree().fit(X, y)

        Estimate the cumulative hazard function for the first 5 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:5])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input, accept_sparse="csr")

        pred = self.tree_.predict(X)
        arr = pred[..., 0]
        if return_array:
            return arr
        return _array_to_step_function(self.event_times_, arr)

    def predict_survival_function(self, X, check_input=True, return_array=False):
        """Predict survival function.

        The survival function for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Kaplan-Meier estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean, default: False
            If set, return an array with the probability
            of survival for each `self.event_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability of
            survival for each `self.event_times_`, otherwise an array of
            length `n_samples` of :class:`sksurv.functions.StepFunction`
            instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.tree import SurvivalTree

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = SurvivalTree().fit(X, y)

        Estimate the survival function for the first 5 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:5])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input, accept_sparse="csr")

        pred = self.tree_.predict(X)
        arr = pred[..., 1]
        if return_array:
            return arr
        return _array_to_step_function(self.event_times_, arr)

    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array-like, shape = (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self, "tree_")
        self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse matrix, shape = (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)
