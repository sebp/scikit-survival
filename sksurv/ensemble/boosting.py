# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numbers
import warnings

import numpy

from sklearn.base import BaseEstimator
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.ensemble._base import BaseEnsemble
from sklearn.ensemble._gb import BaseGradientBoosting, VerboseReporter
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_consistent_length, check_random_state, column_or_1d, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import squared_norm

from scipy.sparse import csc_matrix, csr_matrix, issparse

from ..base import SurvivalAnalysisMixin
from ..util import check_arrays_survival
from .survival_loss import LOSS_FUNCTIONS, CensoredSquaredLoss, \
    IPCWLeastSquaresError


__all__ = ['ComponentwiseGradientBoostingSurvivalAnalysis', 'GradientBoostingSurvivalAnalysis']


def _sample_binomial_plus_one(p, size, random_state):
    drop_model = random_state.binomial(1, p=p, size=size)
    n_dropped = numpy.sum(drop_model)
    if n_dropped == 0:
        idx = random_state.randint(0, size)
        drop_model[idx] = 1
        n_dropped = 1
    return drop_model, n_dropped


class ComponentwiseLeastSquares(BaseEstimator):

    def __init__(self, component):
        self.component = component

    def fit(self, X, y, sample_weight):
        xw = X[:, self.component] * sample_weight
        b = numpy.dot(xw, y)
        if b == 0:
            self.coef_ = 0
        else:
            a = numpy.dot(xw, xw)
            self.coef_ = b / a

        return self

    def predict(self, X):
        return X[:, self.component] * self.coef_


def _fit_stage_componentwise(X, residuals, sample_weight, **fit_params):
    """Fit component-wise weighted least squares model"""
    n_features = X.shape[1]

    base_learners = []
    error = numpy.empty(n_features)
    for component in range(n_features):
        learner = ComponentwiseLeastSquares(component).fit(X, residuals, sample_weight)
        l_pred = learner.predict(X)
        error[component] = squared_norm(residuals - l_pred)
        base_learners.append(learner)

    # TODO: could use bottleneck.nanargmin for speed
    best_component = numpy.nanargmin(error)
    best_learner = base_learners[best_component]
    return best_learner


class ComponentwiseGradientBoostingSurvivalAnalysis(BaseEnsemble, SurvivalAnalysisMixin):
    r"""Gradient boosting with component-wise least squares as base learner.

    See [1]_ for further description.

    Parameters
    ----------
    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.

    learning_rate : float, optional, default: 0.1
        learning rate shrinks the contribution of each base learner by `learning_rate`.
        There is a trade-off between `learning_rate` and `n_estimators`.

    n_estimators : int, default: 100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional, default: 1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    dropout_rate : float, optional, default: 0.0
        If larger than zero, the residuals at each iteration are only computed
        from a random subset of base learners. The value corresponds to the
        percentage of base learners that are dropped. In each iteration,
        at least one base learner is dropped. This is an alternative regularization
        to shrinkage, i.e., setting `learning_rate < 1.0`.

    random_state : int seed, RandomState instance, or None, default: None
        The seed of the pseudo random number generator to use when
        shuffling the data.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    Attributes
    ----------
    coef\_ : array, shape = (n_features + 1,)
        The aggregated coefficients. The first element `coef\_[0]` corresponds
        to the intercept. If loss is `coxph`, the intercept will always be zero.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    estimators_ : list of base learners
        The collection of fitted sub-estimators.

    train_score_ : array, shape = (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    oob_improvement_ : array, shape = (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    References
    ----------
    .. [1] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
           "Survival ensembles", Biostatistics, 7(3), 355-73, 2006
    """
    def __init__(self, loss="coxph", learning_rate=0.1, n_estimators=100, subsample=1.0,
                 dropout_rate=0, random_state=None, verbose=0):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.verbose = verbose

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in ]0; 1] but "
                             "was %r" % self.subsample)

        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be within ]0; 1] but "
                             "was %r" % self.learning_rate)

        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be within [0; 1[, but "
                             "was %r" % self.dropout_rate)

        if self.loss not in LOSS_FUNCTIONS:
            raise ValueError("Loss {!r} not supported.".format(self.loss))

    def _fit(self, X, event, time, sample_weight, random_state):  # noqa: C901
        n_samples = X.shape[0]
        # account for intercept
        Xi = numpy.column_stack((numpy.ones(n_samples), X))
        y = numpy.fromiter(zip(event, time), dtype=[('event', numpy.bool), ('time', numpy.float64)])
        y_pred = numpy.zeros(n_samples)

        do_oob = self.subsample < 1.0
        if do_oob:
            n_inbag = max(1, int(self.subsample * n_samples))

        do_dropout = self.dropout_rate > 0
        if do_dropout:
            scale = numpy.ones(int(self.n_estimators), dtype=float)

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, 0)

        for num_iter in range(int(self.n_estimators)):
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                subsample_weight = sample_weight * sample_mask.astype(numpy.float64)

                # OOB score before adding this stage
                old_oob_score = self.loss_(y[~sample_mask],
                                           y_pred[~sample_mask],
                                           sample_weight[~sample_mask])
            else:
                subsample_weight = sample_weight

            residuals = self.loss_.negative_gradient(y, y_pred, sample_weight=sample_weight)

            best_learner = _fit_stage_componentwise(Xi, residuals, subsample_weight)
            self.estimators_.append(best_learner)

            if do_dropout:
                drop_model, n_dropped = _sample_binomial_plus_one(self.dropout_rate, num_iter + 1, random_state)

                scale[num_iter] = 1. / (n_dropped + 1.)

                y_pred[:] = 0
                for m in range(num_iter + 1):
                    if drop_model[m] == 1:
                        scale[m] *= n_dropped / (n_dropped + 1.)
                    else:
                        y_pred += self.learning_rate * scale[m] * self.estimators_[m].predict(Xi)
            else:
                y_pred += self.learning_rate * best_learner.predict(Xi)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[num_iter] = self.loss_(y[sample_mask], y_pred[sample_mask],
                                                         sample_weight[sample_mask])
                self.oob_improvement_[num_iter] = (old_oob_score
                                                   - self.loss_(y[~sample_mask], y_pred[~sample_mask],
                                                                sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[num_iter] = self.loss_(y, y_pred, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(num_iter, self)

    def fit(self, X, y, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        sample_weight : array-like, shape = (n_samples,), optional
            Weights given to each sample. If omitted, all samples have weight 1.

        Returns
        -------
        self
        """
        X, event, time = check_arrays_survival(X, y)

        n_samples, n_features = X.shape

        if sample_weight is None:
            sample_weight = numpy.ones(n_samples, dtype=numpy.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            check_consistent_length(X, sample_weight)

        random_state = check_random_state(self.random_state)

        self._check_params()

        self.estimators_ = []
        self.n_features_ = n_features
        self.loss_ = LOSS_FUNCTIONS[self.loss](1)
        if isinstance(self.loss_, (CensoredSquaredLoss, IPCWLeastSquaresError)):
            time = numpy.log(time)

        self.train_score_ = numpy.zeros((self.n_estimators,), dtype=numpy.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = numpy.zeros(self.n_estimators,
                                                dtype=numpy.float64)

        self._fit(X, event, time, sample_weight, random_state)

        return self

    def predict(self, X):
        """Predict risk scores.

        If `loss='coxph'`, predictions can be interpreted as log hazard ratio
        corresponding to the linear predictor of a Cox proportional hazards
        model. If `loss='squared'` or `loss='ipcwls'`, predictions are the
        time to event.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        risk_score : array, shape = (n_samples,)
            Predicted risk scores.
        """
        check_is_fitted(self, 'estimators_')

        if X.shape[1] != self.n_features_:
            raise ValueError('Dimensions of X are inconsistent with training data: '
                             'expected %d features, but got %s' % (self.n_features_, X.shape[1]))

        n_samples = X.shape[0]
        Xi = numpy.column_stack((numpy.ones(n_samples), X))
        pred = numpy.zeros(n_samples, dtype=float)

        for estimator in self.estimators_:
            pred += self.learning_rate * estimator.predict(Xi)

        return self.loss_._scale_raw_prediction(pred)

    @property
    def coef_(self):
        """Return the aggregated coefficients.

        Returns
        -------
        coef_ : ndarray, shape = (n_features + 1,)
            Coefficients of features. The first element denotes the intercept.
        """
        coef = numpy.zeros(self.n_features_ + 1, dtype=float)

        for estimator in self.estimators_:
            coef[estimator.component] += self.learning_rate * estimator.coef_

        return coef

    @property
    def feature_importances_(self):
        imp = numpy.empty(self.n_features_ + 1, dtype=object)
        for i in range(imp.shape[0]):
            imp[i] = []

        for k, estimator in enumerate(self.estimators_):
            imp[estimator.component].append(k + 1)

        def _importance(x):
            if len(x) > 0:
                return numpy.min(x)
            return numpy.nan

        ret = numpy.array([_importance(x) for x in imp])
        return ret

    def _make_estimator(self, append=True, random_state=None):
        # we don't need _make_estimator
        raise NotImplementedError()


class GradientBoostingSurvivalAnalysis(BaseGradientBoosting, SurvivalAnalysisMixin):
    r"""Gradient-boosted Cox proportional hazard loss with
    regression trees as base learner.

    In each stage, a regression tree is fit on the negative gradient
    of the loss function.

    For more details on gradient boosting see [1]_ and [2]_, for gradient
    boosting of the proportional hazards model see [3]_. When using a non-zero
    `dropout_rate`, regularization is applied during training following [4]_.

    Parameters
    ----------
    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.

    learning_rate : float, optional, default: 0.1
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int, default: 100
        The number of regression trees to create. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    criterion : string, optional, default: 'friedman_mse'
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

    min_samples_split : integer, optional, default: 2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : integer, optional, default: 1
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the input samples required to be at a
        leaf node.

    max_depth : integer, optional, default: 3
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        Ignored if ``max_leaf_nodes`` is not None.

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

    min_impurity_decrease : float, optional, default: 0.
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    random_state : int seed, RandomState instance, or None, default: None
        The seed of the pseudo random number generator to use when
        shuffling the data.

    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=n_features`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional, default: None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    presort : deprecated, optional, default: 'deprecated'
        This parameter is deprecated and will be removed in a future version.

    subsample : float, optional, default: 1.0
        The fraction of samples to be used for fitting the individual regression
        trees. If smaller than 1.0, this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    dropout_rate : float, optional, default: 0.0
        If larger than zero, the residuals at each iteration are only computed
        from a random subset of base learners. The value corresponds to the
        percentage of base learners that are dropped. In each iteration,
        at least one base learner is dropped. This is an alternative regularization
        to shrinkage, i.e., setting `learning_rate < 1.0`.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    ccp_alpha : non-negative float, optional, default: 0.0.
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.

    Attributes
    ----------
    n_estimators\_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

    feature_importances\_ : ndarray, shape = (n_features,)
        The feature importances (the higher, the more important the feature).

    estimators_ : ndarray of DecisionTreeRegressor, shape = (n_estimators, 1)
        The collection of fitted sub-estimators.

    train_score_ : ndarray, shape = (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    oob_improvement_ : ndarray, shape = (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    References
    ----------
    .. [1] J. H. Friedman, "Greedy function approximation: A gradient boosting machine,"
           The Annals of Statistics, 29(5), 1189–1232, 2001.
    .. [2] J. H. Friedman, "Stochastic gradient boosting,"
           Computational Statistics & Data Analysis, 38(4), 367–378, 2002.
    .. [3] G. Ridgeway, "The state of boosting,"
           Computing Science and Statistics, 172–181, 1999.
    .. [4] K. V. Rashmi and R. Gilad-Bachrach,
           "DART: Dropouts meet multiple additive regression trees,"
           in 18th International Conference on Artificial Intelligence and Statistics,
           2015, 489–497.
    """
    def __init__(self, loss="coxph", learning_rate=0.1, n_estimators=100,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=None,
                 min_impurity_decrease=0., random_state=None,
                 max_features=None, max_leaf_nodes=None,
                 presort='deprecated',
                 subsample=1.0, dropout_rate=0.0,
                 verbose=0,
                 ccp_alpha=0.0):
        super().__init__(loss=loss,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         subsample=subsample,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_depth=max_depth,
                         min_impurity_split=min_impurity_split,
                         min_impurity_decrease=min_impurity_decrease,
                         init=None,
                         random_state=random_state,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         presort=presort,
                         verbose=verbose,
                         ccp_alpha=ccp_alpha)
        self.dropout_rate = dropout_rate

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        self.n_estimators = int(self.n_estimators)
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be within ]0; 1] but "
                             "was %r" % self.learning_rate)

        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in ]0; 1] but "
                             "was %r" % self.subsample)

        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be within [0; 1[, but "
                             "was %r" % self.dropout_rate)

        max_features = self._check_max_features()

        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.max_depth = int(self.max_depth)
        if self.max_leaf_nodes:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.max_features_ = max_features

        if self.presort != 'deprecated':
            warnings.warn("The parameter 'presort' is deprecated and has no "
                          "effect. It will be removed in v0.24. You can "
                          "suppress this warning by not passing any value "
                          "to the 'presort' parameter.", DeprecationWarning)

        if self.loss not in LOSS_FUNCTIONS:
            raise ValueError("Loss {!r} not supported.".format(self.loss))

        self.loss_ = LOSS_FUNCTIONS[self.loss](1)

    def _check_max_features(self):
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(numpy.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(numpy.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, numpy.integer)):
            if self.max_features < 1:
                raise ValueError("max_features must be in (0, n_features]")
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features * self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, 1.0]")
        return max_features

    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
                   random_state, scale, X_idx_sorted, X_csc=None, X_csr=None):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        assert sample_mask.dtype == numpy.bool
        loss = self.loss_

        # whether to use dropout in next iteration
        do_dropout = self.dropout_rate > 0. and 0 < i < len(scale) - 1

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)

            # induce regression tree on residuals
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_split=self.min_impurity_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                presort=self.presort,
                ccp_alpha=self.ccp_alpha)

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(numpy.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight,
                     check_input=False, X_idx_sorted=X_idx_sorted)

            # add tree to ensemble
            self.estimators_[i, k] = tree

            # update tree leaves
            if do_dropout:
                # select base learners to be dropped for next iteration
                drop_model, n_dropped = _sample_binomial_plus_one(self.dropout_rate, i + 1, random_state)

                # adjust scaling factor of tree that is going to be trained in next iteration
                scale[i + 1] = 1. / (n_dropped + 1.)

                raw_predictions[:, k] = 0
                for m in range(i + 1):
                    if drop_model[m] == 1:
                        # adjust scaling factor of dropped trees
                        scale[m] *= n_dropped / (n_dropped + 1.)
                    else:
                        # pseudoresponse of next iteration (without contribution of dropped trees)
                        raw_predictions[:, k] += (
                            self.learning_rate * scale[m] * self.estimators_[m, k].predict(X).ravel())
            else:
                # update tree leaves
                loss.update_terminal_regions(
                    tree.tree_, X, y, residual, raw_predictions, sample_weight,
                    sample_mask, learning_rate=self.learning_rate, k=k)

        return raw_predictions

    def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = numpy.ones((n_samples, ), dtype=numpy.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.dropout_rate > 0.:
            scale = numpy.ones(self.n_estimators, dtype=float)
        else:
            scale = None

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                y_oob_sample = y[~sample_mask]
                old_oob_score = loss_(y_oob_sample,
                                      raw_predictions[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i, X, y, raw_predictions, sample_weight, sample_mask,
                random_state, scale, X_idx_sorted, X_csc, X_csr)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             raw_predictions[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score - loss_(y_oob_sample,
                                          raw_predictions[~sample_mask],
                                          sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

        if self.dropout_rate > 0.:
            self.scale_ = scale

        return i + 1

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        sample_weight : array-like, shape = (n_samples,), optional
            Weights given to each sample. If omitted, all samples have weight 1.

        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
            Returns self.
        """
        X, event, time = check_arrays_survival(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, self.n_features_ = X.shape

        X = X.astype(DTYPE)
        sample_weight_is_none = sample_weight is None
        if sample_weight_is_none:
            sample_weight = numpy.ones(n_samples, dtype=numpy.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)

        check_consistent_length(X, sample_weight)

        self._check_params()

        if isinstance(self.loss_, (CensoredSquaredLoss, IPCWLeastSquaresError)):
            time = numpy.log(time)

        self._init_state()
        if sample_weight_is_none:
            self.init_.fit(X, (event, time))
        else:
            self.init_.fit(X, (event, time), sample_weight)

        raw_predictions = self.loss_.get_init_raw_predictions(X, self.init_)
        begin_at_stage = 0

        # The rng state must be preserved if warm_start is True
        self._rng = check_random_state(self.random_state)

        X_idx_sorted = None

        # fit the boosting stages
        y = numpy.fromiter(zip(event, time), dtype=[('event', numpy.bool), ('time', numpy.float64)])
        n_stages = self._fit_stages(X, y, raw_predictions, sample_weight, self._rng,
                                    begin_at_stage, monitor, X_idx_sorted)
        # change shape of arrays after fit (early-stopping or additional tests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _dropout_predict_stage(self, X, i, K, score):
        for k in range(K):
            tree = self.estimators_[i, k].tree_
            score += self.learning_rate * self.scale_[i] * tree.predict(X).reshape((-1, 1))
        return score

    def _dropout_raw_predict(self, X):
        raw_predictions = self._raw_predict_init(X)

        n_estimators, K = self.estimators_.shape
        for i in range(n_estimators):
            self._dropout_predict_stage(X, i, K, raw_predictions)

        return raw_predictions

    def _dropout_staged_raw_predict(self, X):
        X = check_array(X, dtype=DTYPE, order="C")
        raw_predictions = self._raw_predict_init(X)

        n_estimators, K = self.estimators_.shape
        for i in range(n_estimators):
            self._dropout_predict_stage(X, i, K, raw_predictions)
            yield raw_predictions.copy()

    def _raw_predict(self, X):
        # if dropout wasn't used during training, proceed as usual,
        # otherwise consider scaling factor of individual trees
        if not hasattr(self, "scale_"):
            return super()._raw_predict(X)
        else:
            return self._dropout_raw_predict(X)

    def _init_decision_function(self, X):  # pragma: no cover
        return super()._init_decision_function(X).reshape(-1, 1)

    def _decision_function(self, X):  # pragma: no cover
        return self._raw_predict(X)

    def predict(self, X):
        """Predict risk scores.

        If `loss='coxph'`, predictions can be interpreted as log hazard ratio
        similar to the linear predictor of a Cox proportional hazards
        model. If `loss='squared'` or `loss='ipcwls'`, predictions are the
        time to event.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape = (n_samples,)
            The risk scores.
        """
        check_is_fitted(self, 'estimators_')

        X = check_array(X, dtype=DTYPE, order="C")
        score = self._raw_predict(X)
        if score.shape[1] == 1:
            score = score.ravel()

        return self.loss_._scale_raw_prediction(score)

    def staged_predict(self, X):
        """Predict risk scores at each stage for X.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        If `loss='coxph'`, predictions can be interpreted as log hazard ratio
        similar to the linear predictor of a Cox proportional hazards
        model. If `loss='squared'` or `loss='ipcwls'`, predictions are the
        time to event.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : generator of array of shape = (n_samples,)
            The predicted value of the input samples.
        """
        check_is_fitted(self, 'estimators_')

        # if dropout wasn't used during training, proceed as usual,
        # otherwise consider scaling factor of individual trees
        if not hasattr(self, "scale_"):
            aiter = self._staged_raw_predict(X)
        else:
            aiter = self._dropout_staged_raw_predict(X)

        for raw_predictions in aiter:
            y = self.loss_._scale_raw_prediction(raw_predictions)
            yield y.ravel()
