from math import ceil
import numbers
import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import _tree
from sklearn.tree._splitter import Splitter
from sklearn.tree._tree import BestFirstTreeBuilder, DepthFirstTreeBuilder, Tree
from sklearn.tree._classes import DENSE_SPLITTERS
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

from ..base import SurvivalAnalysisMixin
from ..functions import StepFunction
from ..util import check_arrays_survival
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
    splitter : string, optional, default: "best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

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
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    presort : deprecated, optional, default: 'deprecated'
        This parameter is deprecated and will be removed in a future version.

    Attributes
    ----------
    event_times_ : array of shape = (n_event_times,)
        Unique time points where events occurred.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

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

    def __init__(self,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 presort='deprecated'):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.presort = presort

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Build a survival tree from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = (n_samples, n_features), optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self
        """
        random_state = check_random_state(self.random_state)

        if check_input:
            X, event, time = check_arrays_survival(X, y)
            self.event_times_ = np.unique(time[event])

            y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
            y_numeric[:, 0] = time.astype(np.float64)
            y_numeric[:, 1] = event.astype(np.float64)
        else:
            y_numeric, self.event_times_ = y

        n_samples, self.n_features_ = X.shape
        params = self._check_params(n_samples)

        self.n_outputs_ = self.event_times_.shape[0]
        # one "class" for CHF, one for survival function
        self.n_classes_ = np.ones(self.n_outputs_, dtype=np.intp) * 2

        # Build tree
        criterion = LogrankCriterion(self.n_outputs_, n_samples, self.event_times_)

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = DENSE_SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                params["min_samples_leaf"],
                params["min_weight_leaf"],
                random_state)

        self.tree_ = Tree(self.n_features_, self.n_classes_, self.n_outputs_)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if params["max_leaf_nodes"] < 0:
            builder = DepthFirstTreeBuilder(splitter,
                                            params["min_samples_split"],
                                            params["min_samples_leaf"],
                                            params["min_weight_leaf"],
                                            params["max_depth"],
                                            0.0,  # min_impurity_decrease
                                            params["min_impurity_split"])
        else:
            builder = BestFirstTreeBuilder(splitter,
                                           params["min_samples_split"],
                                           params["min_samples_leaf"],
                                           params["min_weight_leaf"],
                                           params["max_depth"],
                                           params["max_leaf_nodes"],
                                           0.0,  # min_impurity_decrease
                                           params["min_impurity_split"])

        builder.build(self.tree_, X, y_numeric, sample_weight, X_idx_sorted)

        return self

    def _check_params(self, n_samples):
        # Check parameters
        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero.")

        max_leaf_nodes = self._check_max_leaf_nodes()
        min_samples_leaf = self._check_min_samples_leaf(n_samples)
        min_samples_split = self._check_min_samples_split(n_samples)
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        self._check_max_features()

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")

        min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        min_impurity_split = 1e-7

        if self.presort != 'deprecated':
            warnings.warn("The parameter 'presort' is deprecated and has no "
                          "effect. It will be removed in v0.24. You can "
                          "suppress this warning by not passing any value "
                          "to the 'presort' parameter.", DeprecationWarning)

        return {
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "min_impurity_split": min_impurity_split,
            "min_weight_leaf": min_weight_leaf,
        }

    def _check_max_leaf_nodes(self):
        max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)
        if not isinstance(max_leaf_nodes, (numbers.Integral, np.integer)):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % max_leaf_nodes)
        if -1 < max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {} must be either None "
                              "or larger than 1").format(max_leaf_nodes))
        return max_leaf_nodes

    def _check_min_samples_leaf(self, n_samples):
        if isinstance(self.min_samples_leaf, (numbers.Integral, np.integer)):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))
        # FIXME throw exception if min_samples_leaf < 2
        return min_samples_leaf

    def _check_min_samples_split(self, n_samples):
        if isinstance(self.min_samples_split, (numbers.Integral, np.integer)):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)
        return min_samples_split

    def _check_max_features(self):
        if isinstance(self.max_features, str):
            if self.max_features in ("auto", "sqrt"):
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict"""
        if check_input:
            X = check_array(X, dtype=DTYPE)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s."
                             % (self.n_features_, n_features))

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
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        risk_scores : ndarray, shape = (n_samples,)
            Predicted risk scores.
        """
        chf = self.predict_cumulative_hazard_function(X, check_input)
        return chf.sum(1)

    def predict_cumulative_hazard_function(self, X, check_input=True, return_array="warn"):
        """Predict cumulative hazard function.

        The cumulative hazard function (CHF) for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Nelson–Aalen estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean
            If set, return an array with the cumulative hazard rate
            for each `self.event_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.event_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction` will be returned.

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

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:5], return_array=False)

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        if return_array == "warn":
            warnings.warn(
                "predict_cumulative_hazard_function will return an array of StepFunction instances in 0.14. "
                "Use return_array=True to keep the old behavior.",
                FutureWarning)

        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)

        pred = self.tree_.predict(X)
        arr = pred[..., 0]
        if return_array:
            return arr
        return _array_to_step_function(self.event_times_, arr)

    def predict_survival_function(self, X, check_input=True, return_array="warn"):
        """Predict survival function.

        The survival function for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Kaplan-Meier estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean
            If set, return an array with the probability
            of survival for each `self.event_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability
            of survival for each `self.event_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`
            will be returned.

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

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:5], return_array=False)

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        if return_array == "warn":
            warnings.warn(
                "predict_survival_function will return an array of StepFunction instances in 0.14. "
                "Use return_array=True to keep the old behavior.",
                FutureWarning)

        check_is_fitted(self, 'tree_')
        X = self._validate_X_predict(X, check_input)

        pred = self.tree_.predict(X)
        arr = pred[..., 1]
        if return_array:
            return arr
        return _array_to_step_function(self.event_times_, arr)
