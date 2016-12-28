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
from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_X_y, check_array, check_consistent_length, check_random_state
from sklearn.utils.extmath import safe_sparse_dot, squared_norm, fast_dot

from scipy.optimize import minimize

import numpy
import numexpr
import warnings

from ._prsvm import survival_constraints_simple, survival_constraints_with_support_vectors
from ..base import SurvivalAnalysisMixin
from ..bintrees import AVLTree, RBTree
from ..util import check_arrays_survival


class Counter(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, x, y, status, time=None):
        self.x, self.y = check_X_y(x, y)

        assert numpy.issubdtype(y.dtype, numpy.integer), \
            "y vector must have integer type, but was {0}".format(y.dtype)
        assert y.min() == 0, "minimum element of y vector must be 0"

        if time is None:
            self.status = check_array(status, dtype=bool, ensure_2d=False)
            check_consistent_length(self.x, self.status)
        else:
            self.status = check_array(status, dtype=bool, ensure_2d=False)
            self.time = check_array(time, ensure_2d=False)
            check_consistent_length(self.x, self.status, self.time)

        self.eps = numpy.finfo(self.x.dtype).eps

    def update_sort_order(self, w):
        xw = numpy.dot(self.x, w)
        order = xw.argsort(kind='mergesort')
        self.xw = xw[order]
        self.order = order
        return xw

    @abstractmethod
    def calculate(self, v):
        """Return l_plus, xv_plus, l_minus, xv_minus"""


class OrderStatisticTreeSurvivalCounter(Counter):
    """Counting method used by :class:`LargeScaleOptimizer` for survival analysis.

    Parameters
    ----------
    x : array, shape = [n_samples, n_features]
        Feature matrix

    y : array of int, shape = [n_samples]
        Unique ranks of samples, starting with 0.

    status : array of bool, shape = [n_samples]
        Event indicator of samples.

    tree_class : type
        Which class to use as order statistic tree

    time : array, shape = [n_samples]
        Survival times.
    """
    def __init__(self, x, y, status, tree_class, time=None):
        super().__init__(x, y, status, time)
        self._tree_class = tree_class

    def calculate(self, v):
        # only self.xw is sorted, for everything else use self.order
        # the order of return values is with respect to original order of samples, NOT self.order
        xv = numpy.dot(self.x, v)

        od = self.order

        n_samples = self.x.shape[0]
        l_plus = numpy.zeros(n_samples, dtype=int)
        l_minus = numpy.zeros(n_samples, dtype=int)
        xv_plus = numpy.zeros(n_samples, dtype=float)
        xv_minus = numpy.zeros(n_samples, dtype=float)

        j = 0
        tree = self._tree_class(n_samples)
        for i in range(n_samples):
            while j < n_samples and 1 - self.xw[j] + self.xw[i] > 0:
                tree.insert(self.y[od[j]], xv[od[j]])
                j += 1

            # larger (root of t, y[od[i]])
            count, vec_sum = tree.count_larger_with_event(self.y[od[i]], self.status[od[i]])
            l_plus[od[i]] = count
            xv_plus[od[i]] = vec_sum

        tree = self._tree_class(n_samples)
        j = n_samples - 1
        for i in range(j, -1, -1):
            while j >= 0 and 1 - self.xw[i] + self.xw[j] > 0:
                if self.status[od[j]]:
                    tree.insert(self.y[od[j]], xv[od[j]])
                j -= 1

            # smaller (root of T, y[od[i]])
            count, vec_sum = tree.count_smaller(self.y[od[i]])
            l_minus[od[i]] = count
            xv_minus[od[i]] = vec_sum

        return l_plus, xv_plus, l_minus, xv_minus


class SurvivalCounter(Counter):

    def __init__(self, x, y, status, n_relevance_levels, time=None):
        super().__init__(x, y, status, time)
        self.n_relevance_levels = n_relevance_levels

    def _count_values(self):
        """Return dict mapping relevance level to sample index"""
        indices = {yi: [i] for i, yi in enumerate(self.y) if self.status[i]}

        return indices

    def calculate(self, v):
        n_samples = self.x.shape[0]
        l_plus = numpy.zeros(n_samples, dtype=int)
        l_minus = numpy.zeros(n_samples, dtype=int)
        xv_plus = numpy.zeros(n_samples, dtype=float)
        xv_minus = numpy.zeros(n_samples, dtype=float)
        indices = self._count_values()

        od = self.order

        for relevance in range(self.n_relevance_levels):
            j = 0
            count_plus = 0
            # relevance levels are unique, therefore count can only be 1 or 0
            count_minus = 1 if relevance in indices else 0
            xv_count_plus = 0
            xv_count_minus = numpy.dot(self.x.take(indices.get(relevance, []), axis=0), v).sum()

            for i in range(n_samples):
                if self.y[od[i]] != relevance or not self.status[od[i]]:
                    continue

                while j < n_samples and 1 - self.xw[j] + self.xw[i] > 0:
                    if self.y[od[j]] > relevance:
                        count_plus += 1
                        xv_count_plus += numpy.dot(self.x[od[j], :], v)
                        l_minus[od[j]] += count_minus
                        xv_minus[od[j]] += xv_count_minus

                    j += 1

                l_plus[od[i]] = count_plus
                xv_plus[od[i]] += xv_count_plus
                count_minus -= 1
                xv_count_minus -= numpy.dot(self.x.take(od[i], axis=0), v)

        return l_plus, xv_plus, l_minus, xv_minus


class RankSVMOptimizer(object, metaclass=ABCMeta):
    """Abstract base class for all optimizers"""
    def __init__(self, alpha, rank_ratio, timeit=False):
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.timeit = timeit

        self._last_w = None
        # cache gradient computations
        self._last_gradient_w = None
        self._last_gradient = None

    @abstractmethod
    def _objective_func(self, w):
        """Evaluate objective function at w"""

    @abstractmethod
    def _update_constraints(self, w):
        """Update constraints"""

    @abstractmethod
    def _gradient_func(self, w):
        """Evaluate gradient at w"""

    @abstractmethod
    def _hessian_func(self, w, s):
        """Evaluate Hessian at w"""

    @abstractproperty
    def n_coefficients(self):
        """Return number of coefficients (includes intercept)"""

    def _update_constraints_if_necessary(self, w):
        needs_update = (w != self._last_w).any()
        if needs_update:
            self._update_constraints(w)
            self._last_w = w.copy()
        return needs_update

    def _do_objective_func(self, w):
        self._update_constraints_if_necessary(w)
        return self._objective_func(w)

    def _do_gradient_func(self, w):
        if self._last_gradient_w is not None and (w == self._last_gradient_w).all():
            return self._last_gradient

        self._update_constraints_if_necessary(w)
        self._last_gradient_w = w.copy()
        self._last_gradient = self._gradient_func(w)
        return self._last_gradient

    def _init_coefficients(self):
        w = numpy.zeros(self.n_coefficients)
        self._update_constraints(w)
        self._last_w = w.copy()
        return w

    def run(self, **kwargs):
        w = self._init_coefficients()

        timings = None
        if self.timeit:
            import timeit

            def _inner():
                return minimize(self._do_objective_func, w, method='newton-cg',
                                jac=self._do_gradient_func, hessp=self._hessian_func, **kwargs)

            timer = timeit.Timer(_inner)
            timings = timer.repeat(self.timeit, number=1)

        opt_result = minimize(self._do_objective_func, w, method='newton-cg',
                              jac=self._do_gradient_func, hessp=self._hessian_func, **kwargs)
        opt_result['timings'] = timings

        return opt_result


class SimpleOptimizer(RankSVMOptimizer):
    """Simple optimizer, which explicitly constructs matrix of all pairs of samples"""
    def __init__(self, x, y, alpha, rank_ratio, timeit=False):
        super().__init__(alpha, rank_ratio, timeit)
        self.data_x = x
        self.constraints = survival_constraints_simple(numpy.asarray(y, dtype=numpy.uint8))

        self.L = numpy.ones(self.constraints.shape[0])

    @property
    def n_coefficients(self):
        return self.data_x.shape[1]

    def _objective_func(self, w):
        val = 0.5 * squared_norm(w) + 0.5 * self.alpha * squared_norm(self.L)
        return val

    def _update_constraints(self, w):
        self.xw = numpy.dot(self.data_x, w)
        self.L = 1 - self.constraints.dot(self.xw)
        numpy.maximum(0, self.L, out=self.L)
        support_vectors = numpy.nonzero(self.L > 0)[0]
        self.Asv = self.constraints[support_vectors, :]

    def _gradient_func(self, w):
        # sum over columns without running into overflow problems
        # scipy.sparse.spmatrix.sum uses dtype of matrix, which is too small
        col_sum = numpy.asmatrix(numpy.ones((1, self.Asv.shape[0]), dtype=numpy.int_)) * self.Asv
        v = numpy.asarray(col_sum).squeeze()

        z = fast_dot(self.data_x.T, (self.Asv.T.dot(self.Asv.dot(self.xw)) - v))
        return w + self.alpha * z

    def _hessian_func(self, w, s):
        z = self.alpha * self.Asv.dot(numpy.dot(self.data_x, s))
        return s + numpy.dot(safe_sparse_dot(z.T, self.Asv), self.data_x).T


class PRSVMOptimizer(RankSVMOptimizer):
    """PRSVM optimizer that after each iteration of Newton's method constructs matrix of support vector pairs"""
    def __init__(self, x, y, alpha, rank_ratio, timeit=False):
        super().__init__(alpha, rank_ratio, timeit)
        self.data_x = x
        self.data_y = numpy.asarray(y, dtype=numpy.uint8)
        self._constraints = lambda w: survival_constraints_with_support_vectors(self.data_y, w)

    @property
    def n_coefficients(self):
        return self.data_x.shape[1]

    def _objective_func(self, w):
        z = self.Aw.shape[0] + squared_norm(self.AXw) - 2. * self.AXw.sum()
        val = 0.5 * squared_norm(w) + 0.5 * self.alpha * z
        return val

    def _update_constraints(self, w):
        xw = numpy.dot(self.data_x, w)
        self.Aw = self._constraints(xw)
        self.AXw = self.Aw.dot(xw)

    def _gradient_func(self, w):
        # sum over columns without running into overflow problems
        # scipy.sparse.spmatrix.sum uses dtype of matrix, which is too small
        col_sum = numpy.asmatrix(numpy.ones((1, self.Aw.shape[0]), dtype=numpy.int_)) * self.Aw
        v = numpy.asarray(col_sum).squeeze()
        z = fast_dot(self.data_x.T, self.Aw.T.dot(self.AXw) - v)
        return w + self.alpha * z

    def _hessian_func(self, w, s):
        v = self.Aw.dot(numpy.dot(self.data_x, s))
        z = self.alpha * fast_dot(self.data_x.T, self.Aw.T.dot(v))
        return s + z


class LargeScaleOptimizer(RankSVMOptimizer):
    """Optimizer that does not explicitly create matrix of constraints

    Parameters
    ----------
    alpha : float
        Regularization parameter.

    rank_ratio : float
        Trade-off between regression and ranking objectives.

    fit_intercept : bool
        Whether to fit an intercept. Only used if regression objective
        is optimized (rank_ratio < 1.0).

    counter : object
        Instance of :class:`Counter` subclass.

    References
    ----------
    Lee, C.-P., & Lin, C.-J. (2014). Supplement Materials for "Large-scale linear RankSVM". Neural Computation, 26(4),
        781–817. doi:10.1162/NECO_a_00571
    """
    def __init__(self, alpha, rank_ratio, fit_intercept, counter, timeit=False):
        super().__init__(alpha, rank_ratio, timeit)

        self._counter = counter
        self._regr_penalty = (1.0 - rank_ratio) * alpha
        self._rank_penalty = rank_ratio * alpha
        self._has_time = hasattr(self._counter, 'time') and self._regr_penalty > 0
        self._fit_intercept = fit_intercept if self._has_time else False

    @property
    def n_coefficients(self):
        n = self._counter.x.shape[1]
        if self._fit_intercept:
            n += 1
        return n

    def _init_coefficients(self):
        w = super()._init_coefficients()
        if self._fit_intercept:
            w[0] = self._counter.time.mean()
        return w

    def _split_coefficents(self, w):
        """Split into intercept/bias and feature-specific coefficients"""
        if self._fit_intercept:
            bias = w[0]
            wf = w[1:]
        else:
            bias = 0.0
            wf = w
        return bias, wf

    def _objective_func(self, w):
        bias, wf = self._split_coefficents(w)

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(wf)

        xw = self._xw
        val = 0.5 * squared_norm(wf)
        if self._has_time:
            val += 0.5 * self._regr_penalty * squared_norm(self.y_compressed - bias
                                                           - xw.compress(self.regr_mask, axis=0))

        val += 0.5 * self._rank_penalty * numexpr.evaluate(
            'sum(xw * ((l_plus + l_minus) * xw - xv_plus - xv_minus - 2 * (l_minus - l_plus)) + l_minus)')

        return val

    def _update_constraints(self, w):
        bias, wf = self._split_coefficents(w)

        self._xw = self._counter.update_sort_order(wf)

        if self._has_time:
            pred_time = self._counter.time - self._xw - bias
            self.regr_mask = (pred_time > 0) | self._counter.status
            self.y_compressed = self._counter.time.compress(self.regr_mask, axis=0)

    def _gradient_func(self, w):
        bias, wf = self._split_coefficents(w)

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(wf)
        x = self._counter.x

        xw = self._xw
        z = numexpr.evaluate('(l_plus + l_minus) * xw - xv_plus - xv_minus - l_minus + l_plus')

        grad = wf + self._rank_penalty * fast_dot(x.T, z)
        if self._has_time:
            xc = x.compress(self.regr_mask, axis=0)
            xcs = numpy.dot(xc, wf)
            grad += self._regr_penalty * (fast_dot(xc.T, xcs) + xc.sum(axis=0) * bias - fast_dot(xc.T, self.y_compressed))

            # intercept
            if self._fit_intercept:
                grad_intercept = self._regr_penalty * (xcs.sum() + xc.shape[0] * bias - self.y_compressed.sum())
                grad = numpy.concatenate(([grad_intercept], grad))

        return grad

    def _hessian_func(self, w, s):
        s_bias, s_feat = self._split_coefficents(s)

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(s_feat)
        x = self._counter.x

        xs = numpy.dot(x, s_feat)
        xs = numexpr.evaluate('(l_plus + l_minus) * xs - xv_plus - xv_minus')

        hessp = s_feat + self._rank_penalty * fast_dot(x.T, xs)
        if self._has_time:
            xc = x.compress(self.regr_mask, axis=0)
            hessp += self._regr_penalty * fast_dot(xc.T, numpy.dot(xc, s_feat))

            # intercept
            if self._fit_intercept:
                xsum = xc.sum(axis=0)
                hessp += self._regr_penalty * xsum * s_bias
                hessp_intercept = self._regr_penalty * xc.shape[0] * s_bias + self._regr_penalty * numpy.dot(xsum, s_feat)
                hessp = numpy.concatenate(([hessp_intercept], hessp))

        return hessp


class NonlinearLargeScaleOptimizer(RankSVMOptimizer):
    """Optimizer that does not explicitly create matrix of constraints

    Parameters
    ----------
    alpha : float
        Regularization parameter.

    rank_ratio : float
        Trade-off between regression and ranking objectives.

    counter : object
        Instance of :class:`Counter` subclass.

    References
    ----------
    Lee, C.-P., & Lin, C.-J. (2014). Supplement Materials for "Large-scale linear RankSVM". Neural Computation, 26(4),
        781–817. doi:10.1162/NECO_a_00571
    """
    def __init__(self, alpha, rank_ratio, fit_intercept, counter, timeit=False):
        super().__init__(alpha, rank_ratio, timeit)

        self._counter = counter
        self._fit_intercept = fit_intercept
        self._rank_penalty = rank_ratio * alpha
        self._regr_penalty = (1.0 - rank_ratio) * alpha
        self._has_time = hasattr(self._counter, 'time') and self._regr_penalty > 0
        self._fit_intercept = fit_intercept if self._has_time else False

    @property
    def n_coefficients(self):
        n = self._counter.x.shape[0]
        if self._fit_intercept:
            n += 1
        return n

    def _init_coefficients(self):
        w = super()._init_coefficients()
        if self._fit_intercept:
            w[0] = self._counter.time.mean()
        return w

    def _split_coefficents(self, w):
        """Split into intercept/bias and feature-specific coefficients"""
        if self._fit_intercept:
            bias = w[0]
            wf = w[1:]
        else:
            bias = 0.0
            wf = w
        return bias, wf

    def _update_constraints(self, beta_bias):
        bias, beta = self._split_coefficents(beta_bias)

        self._Kw = self._counter.update_sort_order(beta)

        if self._has_time:
            pred_time = self._counter.time - self._Kw - bias
            self.regr_mask = (pred_time > 0) | self._counter.status
            self.y_compressed = self._counter.time.compress(self.regr_mask, axis=0)

    def _objective_func(self, beta_bias):
        bias, beta = self._split_coefficents(beta_bias)

        Kw = self._Kw

        val = 0.5 * numpy.dot(beta, Kw)
        if self._has_time:
            val += 0.5 * self._regr_penalty * squared_norm(self.y_compressed - bias
                                                           - Kw.compress(self.regr_mask, axis=0))

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(beta)
        val += 0.5 * self._rank_penalty * numexpr.evaluate(
            'sum(Kw * ((l_plus + l_minus) * Kw - xv_plus - xv_minus - 2 * (l_minus - l_plus)) + l_minus)')

        return val

    def _gradient_func(self, beta_bias):
        bias, beta = self._split_coefficents(beta_bias)

        K = self._counter.x
        Kw = self._Kw

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(beta)
        z = numexpr.evaluate('(l_plus + l_minus) * Kw - xv_plus - xv_minus - l_minus + l_plus')

        gradient = Kw + self._rank_penalty * fast_dot(K, z)
        if self._has_time:
            K_comp = K.compress(self.regr_mask, axis=0)
            K_comp_beta = numpy.dot(K_comp, beta)
            gradient += self._regr_penalty * (fast_dot(K_comp.T, K_comp_beta)
                                              + K_comp.sum(axis=0) * bias - fast_dot(K_comp.T, self.y_compressed))

            # intercept
            if self._fit_intercept:
                grad_intercept = self._regr_penalty * (K_comp_beta.sum() + K_comp.shape[0] * bias - self.y_compressed.sum())
                gradient = numpy.concatenate(([grad_intercept], gradient))

        return gradient

    def _hessian_func(self, beta, s):
        s_bias, s_feat = self._split_coefficents(s)

        K = self._counter.x
        Ks = numpy.dot(K, s_feat)

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(s_feat)
        xs = numexpr.evaluate('(l_plus + l_minus) * Ks - xv_plus - xv_minus')

        hessian = Ks + self._rank_penalty * fast_dot(K, xs)
        if self._has_time:
            K_comp = K.compress(self.regr_mask, axis=0)
            hessian += self._regr_penalty * fast_dot(K_comp.T, numpy.dot(K_comp, s_feat))

            # intercept
            if self._fit_intercept:
                xsum = K_comp.sum(axis=0)
                hessian += self._regr_penalty * xsum * s_bias
                hessian_intercept = self._regr_penalty * K_comp.shape[0] * s_bias \
                                    + self._regr_penalty * numpy.dot(xsum, s_feat)
                hessian = numpy.concatenate(([hessian_intercept], hessian))

        return hessian


class BaseSurvivalSVM(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, alpha=1, rank_ratio=1.0, fit_intercept=False,
                 max_iter=20, verbose=False, tol=None,
                 optimizer=None, random_state=None, timeit=False):
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.optimizer = optimizer
        self.random_state = random_state
        self.timeit = timeit

        self.coef_ = None
        self.optimizer_result_ = None

    def _create_optimizer(self, X, y, status):
        """Samples are ordered by relevance"""
        if self.optimizer is None:
            self.optimizer = 'avltree'

        times, ranks = y

        if self.optimizer == 'simple':
            optimizer = SimpleOptimizer(X, status, self.alpha, self.rank_ratio, timeit=self.timeit)
        elif self.optimizer == 'PRSVM':
            optimizer = PRSVMOptimizer(X, status, self.alpha, self.rank_ratio, timeit=self.timeit)
        elif self.optimizer == 'direct-count':
            optimizer = LargeScaleOptimizer(self.alpha, self.rank_ratio, self.fit_intercept,
                                            SurvivalCounter(X, ranks, status, len(ranks), times), timeit=self.timeit)
        elif self.optimizer == 'rbtree':
            optimizer = LargeScaleOptimizer(self.alpha, self.rank_ratio, self.fit_intercept,
                                            OrderStatisticTreeSurvivalCounter(X, ranks, status, RBTree, times),
                                            timeit=self.timeit)
        elif self.optimizer == 'avltree':
            optimizer = LargeScaleOptimizer(self.alpha, self.rank_ratio, self.fit_intercept,
                                            OrderStatisticTreeSurvivalCounter(X, ranks, status, AVLTree, times),
                                            timeit=self.timeit)
        else:
            raise ValueError('unknown optimizer: {0}'.format(self.optimizer))

        return optimizer

    @abstractmethod
    def _fit(self, X, time, event, samples_order):
        """Create and run optimizer"""

    @abstractmethod
    def predict(self, X):
        """Predict risk score"""

    def fit(self, X, y):
        """Build a survival support vector machine model from training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix.

        y : structured array, shape = [n_samples]
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        X, event, time = check_arrays_survival(X, y)

        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        if not 0 <= self.rank_ratio <= 1:
            raise ValueError("rank_ratio must be in [0; 1]")

        if self.fit_intercept and self.rank_ratio == 1.0:
            raise ValueError("fit_intercept=True is only meaningful if rank_ratio < 1.0")

        if self.rank_ratio < 1.0:
            if self.optimizer in {'simple', 'PRSVM'}:
                raise ValueError("optimizer '%s' does not implement regression objective" % self.optimizer)

            if (time <= 0).any():
                raise ValueError("observed time contains values smaller or equal to zero")

            # log-transform time
            time = numpy.log(time)
            assert numpy.isfinite(time).all()

        random_state = check_random_state(self.random_state)
        samples_order = BaseSurvivalSVM._argsort_and_resolve_ties(time, random_state)

        opt_result = self._fit(X, time, event, samples_order)
        coef = opt_result.x
        if self.fit_intercept:
            self.coef_ = coef[1:]
            self.intercept_ = coef[0]
        else:
            self.coef_ = coef

        if not opt_result.success:
            warnings.warn(('Optimization did not converge: ' + opt_result.message), stacklevel=2)
        self.optimizer_result_ = opt_result

        return self

    @staticmethod
    def _argsort_and_resolve_ties(time, random_state):
        """Like numpy.argsort, but resolves ties uniformly at random"""
        n_samples = len(time)
        order = numpy.argsort(time, kind="mergesort")

        i = 0
        while i < n_samples - 1:
            inext = i + 1
            while inext < n_samples and time[order[i]] == time[order[inext]]:
                inext += 1

            if i + 1 != inext:
                # resolve ties randomly
                random_state.shuffle(order[i:inext])
            i = inext
        return order


class FastSurvivalSVM(BaseSurvivalSVM, SurvivalAnalysisMixin):
    """Efficient Training of linear Survival Support Vector Machine

    Training data consists of *n* triplets :math:`(\\mathbf{x}_i, y_i, \\delta_i)`,
    where :math:`\\mathbf{x}_i` is a *d*-dimensional feature vector, :math:`y_i > 0`
    the survival time or time of censoring, and :math:`\\delta_i \\in \\{0,1\\}`
    the binary event indicator. Using the training data, the objective is to
    minimize the following function:

    .. math::

         \\arg \\min_{\\mathbf{w}, b} \\frac{1}{2} \\mathbf{w}^\\top \\mathbf{w}
         + \\frac{\\alpha}{2} \\left[ r \\sum_{i,j \\in \\mathcal{P}}
         \\max(0, 1 - (\\mathbf{w}^\\top \\mathbf{x}_i - \\mathbf{w}^\\top \\mathbf{x}_j))^2
         + (1 - r) \\sum_{i=0}^n \\left( \\zeta_{\\mathbf{w}, b} (y_i, x_i, \\delta_i)
         \\right)^2 \\right]

        \\zeta_{\\mathbf{w},b} (y_i, \\mathbf{x}_i, \\delta_i) =
        \\begin{cases}
        \\max(0, y_i - \\mathbf{w}^\\top \\mathbf{x}_i - b) \\quad \\text{if $\\delta_i = 0$,} \\\\
        y_i - \\mathbf{w}^\\top \\mathbf{x}_i - b \\quad \\text{if $\\delta_i = 1$,} \\\\
        \\end{cases}

        \\mathcal{P} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1 \\}_{i,j=1,\\dots,n}

    The hyper-parameter :math:`\\alpha > 0` determines the amount of regularization
    to apply: a smaller value increases the amount of regularization and a
    higher value reduces the amount of regularization. The hyper-parameter
    :math:`r \\in [0; 1]` determines the trade-off between the ranking objective
    and the regresson objective. If :math:`r = 1` it reduces to the ranking
    objective, and if :math:`r = 0` to the regression objective. If the regression
    objective is used, survival/censoring times are log-transform and thus cannot be
    zero or negative.

    See :class:`sksurv.svm.FastKernelSurvivalSVM` for an efficient implementation
    of kernel Survival Support Vector Machine.

    Parameters
    ----------
    alpha : float, positive (default: 1)
        Weight of penalizing the squared hinge loss in the objective function

    rank_ratio : float, optional (default=1.0)
        Mixing parameter between regression and ranking objective with ``0 <= rank_ratio <= 1``.
        If ``rank_ratio = 1``, only ranking is performed, if ``rank_ratio = 0``, only regression
        is performed. A non-zero value is only allowed if optimizer is one of 'avltree', 'rbtree',
        or 'direct-count'.

    fit_intercept : boolean, optional (default=False)
        Whether to calculate an intercept for the regression model. If set to ``False``, no intercept
        will be calculated. Has no effect if ``rank_ratio = 1``, i.e., only ranking is performed.

    max_iter : int, optional (default: 20)
        Maximum number of iterations to perform in Newton optimization

    verbose : bool, optional (default: False)
        Whether to print messages during optimization

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.

    optimizer : "avltree" | "direct-count" | "PRSVM" | "rbtree" | "simple", optional (default: avltree)
        Which optimizer to use.

    random_state : int or :class:`numpy.random.RandomState` instance, optional
        Random number generator (used to resolve ties in survival times).

    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``optimizer_result_`` attribute.

    Attributes
    ----------
    `coef_`:
        Coefficients of the features in the decision function.

    `optimizer_result_`:
        Stats returned by the optimizer. See :class:`scipy.optimize.optimize.OptimizeResult`.

    References
    ----------

    .. [1] Pölsterl, S., Navab, N., and Katouzian, A.,
           "Fast Training of Support Vector Machines for Survival Analysis",
           Machine Learning and Knowledge Discovery in Databases: European Conference,
           ECML PKDD 2015, Porto, Portugal,
           Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)
    """
    def __init__(self, alpha=1, rank_ratio=1.0, fit_intercept=False,
                 max_iter=20, verbose=False, tol=None,
                 optimizer=None, random_state=None, timeit=False):
        super().__init__(alpha=alpha, rank_ratio=rank_ratio, fit_intercept=fit_intercept,
                         max_iter=max_iter, verbose=verbose, tol=tol,
                         optimizer=optimizer, random_state=random_state,
                         timeit=timeit)

    def _fit(self, X, time, event, samples_order):
        data_y = (time[samples_order], numpy.arange(len(samples_order)))
        status = event[samples_order]

        optimizer = self._create_optimizer(X[samples_order], data_y, status)
        opt_result = optimizer.run(tol=self.tol, options={'maxiter': self.max_iter, 'disp': self.verbose})
        return opt_result

    def predict(self, X):
        """Rank samples according to survival times

        Lower ranks indicate shorter survival, higher ranks longer survival.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Predicted ranks.
        """
        val = numpy.dot(X, self.coef_)
        if hasattr(self, "intercept_"):
            val += self.intercept_

        # Order by increasing survival time if objective is pure ranking
        if self.rank_ratio == 1:
            val *= -1
        else:
            # model was fitted on log(time), transform to original scale
            val = numpy.exp(val)

        return val


class FastKernelSurvivalSVM(BaseSurvivalSVM, SurvivalAnalysisMixin):
    """Efficient Training of kernel Survival Support Vector Machine.

    Parameters
    ----------
    alpha : float, positive (default: 1)
        Weight of penalizing the squared hinge loss in the objective function

    rank_ratio : float, optional (default=1.0)
        Mixing parameter between regression and ranking objective with ``0 <= rank_ratio <= 1``.
        If ``rank_ratio = 1``, only ranking is performed, if ``rank_ratio = 0``, only regression
        is performed. A non-zero value is only allowed if optimizer is one of 'avltree', 'PRSVM',
        or 'rbtree'.

    fit_intercept : boolean, optional (default=False)
        Whether to calculate an intercept for the regression model. If set to ``False``, no intercept
        will be calculated. Has no effect if ``rank_ratio = 1``, i.e., only ranking is performed.

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"

    degree : int (default=3)
        Degree for poly kernels. Ignored by other kernels.

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
        Ignored by other kernels.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as call

    max_iter : int, optional (default: 20)
        Maximum number of iterations to perform in Newton optimization

    verbose : bool, optional (default: False)
        Whether to print messages during optimization

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.

    optimizer : "avltree" | "rbtree", optional (default: "rbtree")
        Which optimizer to use.

    random_state : int or :class:`numpy.random.RandomState` instance, optional
        Random number generator (used to resolve ties in survival times).

    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``optimizer_result_`` attribute.

    Attributes
    ----------
    `coef_`:
        Coefficients of the features in the decision function.

    `fit_X_`:
        Training data.

    `optimizer_result_`:
        Stats returned by the optimizer. See :class:`scipy.optimize.optimize.OptimizeResult`.

    References
    ----------

    .. [1] Pölsterl, S., Navab, N., and Katouzian, A.,
           *An Efficient Training Algorithm for Kernel Survival Support Vector Machines*
           4th Workshop on Machine Learning in Life Sciences,
           23 September 2016, Riva del Garda, Italy
    """
    def __init__(self, alpha=1, rank_ratio=1.0, fit_intercept=False, kernel="rbf",
                 gamma=None, degree=3, coef0=1, kernel_params=None, max_iter=20, verbose=False, tol=None,
                 optimizer=None, random_state=None, timeit=False):
        super().__init__(alpha=alpha, rank_ratio=rank_ratio, fit_intercept=fit_intercept,
                         max_iter=max_iter, verbose=verbose, tol=tol,
                         optimizer=optimizer, random_state=random_state,
                         timeit=timeit)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    @property
    def _pairwise(self):
        # tell sklearn.utils.metaestimators._safe_split function that we expect kernel matrix
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _create_optimizer(self, kernel_mat, y, status):
        if self.optimizer is None:
            self.optimizer = 'rbtree'

        times, ranks = y

        if self.optimizer == 'rbtree':
            optimizer = NonlinearLargeScaleOptimizer(self.alpha, self.rank_ratio, self.fit_intercept,
                                                     OrderStatisticTreeSurvivalCounter(kernel_mat, ranks, status, RBTree, times),
                                                     timeit=self.timeit)
        elif self.optimizer == 'avltree':
            optimizer = NonlinearLargeScaleOptimizer(self.alpha, self.rank_ratio, self.fit_intercept,
                                                     OrderStatisticTreeSurvivalCounter(kernel_mat, ranks, status, AVLTree, times),
                                                     timeit=self.timeit)
        else:
            raise ValueError('unknown optimizer: {0}'.format(self.optimizer))

        return optimizer

    def _fit(self, X, time, event, samples_order):
        # don't reorder X here, because it might be a precomputed kernel matrix
        kernel_mat = self._get_kernel(X)
        if (numpy.abs(kernel_mat.T - kernel_mat) > 1e-12).any():
            raise ValueError('kernel matrix is not symmetric')

        data_y = (time[samples_order], numpy.arange(len(samples_order)))
        status = event[samples_order]

        optimizer = self._create_optimizer(kernel_mat[numpy.ix_(samples_order, samples_order)], data_y, status)
        opt_result = optimizer.run(tol=self.tol, options={'maxiter': self.max_iter, 'disp': self.verbose})

        # reorder coefficients according to order in original training data,
        # i.e., reverse ordering according to samples_order
        self.fit_X_ = X
        if self.fit_intercept:
            opt_result.x[samples_order + 1] = opt_result.x[1:].copy()
        else:
            opt_result.x[samples_order] = opt_result.x.copy()

        return opt_result

    def predict(self, X):
        kernel_mat = self._get_kernel(X, self.fit_X_)

        val = numpy.dot(kernel_mat, self.coef_)
        if hasattr(self, "intercept_"):
            val += self.intercept_

        # Order by increasing survival time if objective is pure ranking
        if self.rank_ratio == 1:
            val *= -1
        else:
            # model was fitted on log(time), transform to original scale
            val = numpy.exp(val)

        return val
