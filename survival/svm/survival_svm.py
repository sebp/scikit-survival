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
from sklearn.decomposition import KernelPCA
from sklearn.utils import check_X_y, check_array, check_consistent_length, check_random_state
from sklearn.utils.extmath import safe_sparse_dot, squared_norm, fast_dot

from scipy.optimize import minimize
import six

import numpy
import numexpr
import warnings

from ._prsvm import survival_constraints_simple, survival_constraints_with_support_vectors
from ..bintrees import AVLTree, RBTree
from ..util import check_arrays_survival


class Counter(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __init__(self, x, y, status, time=None):
        self.x, self.y = check_X_y(x, y)

        if not numpy.issubdtype(y.dtype, numpy.integer):
            raise TypeError("y vector must have integer type, but was {0}".format(y.dtype))
        if y.min() != 0:
            raise ValueError("minimum element of y vector must be 0")

        if time is None:
            self.status = check_array(status, dtype=bool, ensure_2d=False)
            check_consistent_length(self.x, self.status)
        else:
            self.status = check_array(status, dtype=bool, ensure_2d=False)
            self.time = check_array(time, ensure_2d=False)
            check_consistent_length(self.x, self.status, self.time)

        self.eps = numpy.finfo(self.x.dtype).eps

    def update_sort_order(self, w):
        self.xw = numpy.dot(self.x, w)
        order = self.xw.argsort(kind='mergesort')
        self.xw = self.xw[order]
        self.x = self.x[order]
        self.y = self.y[order]
        self.status = self.status[order]
        if hasattr(self, 'time'):
            self.time = self.time[order]
        return order

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
        # self.x is already sorted by call to self.update_sort_order,
        # therefore x * v is as well
        xv = numpy.dot(self.x, v)

        n_samples = len(self.xw)
        l_plus = numpy.zeros(n_samples, dtype=int)
        l_minus = numpy.zeros(n_samples, dtype=int)
        xv_plus = numpy.zeros(n_samples, dtype=float)
        xv_minus = numpy.zeros(n_samples, dtype=float)

        j = 0
        tree = self._tree_class(n_samples)
        for i in range(n_samples):
            while j < n_samples and 1 - self.xw[j] + self.xw[i] > 0:
                tree.insert(self.y[j], xv[j])
                j += 1

            # larger (root of t, y[i])
            count, vec_sum = tree.count_larger_with_event(self.y[i], self.status[i])
            l_plus[i] = count
            xv_plus[i] = vec_sum

        tree = self._tree_class(n_samples)
        j = n_samples - 1
        for i in range(j, -1, -1):
            while j >= 0 and 1 - self.xw[i] + self.xw[j] > 0:
                if self.status[j]:
                    tree.insert(self.y[j], xv[j])
                j -= 1

            # smaller (root of T, y[i])
            count, vec_sum = tree.count_smaller(self.y[i])
            l_minus[i] = count
            xv_minus[i] = vec_sum

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

        for relevance in range(self.n_relevance_levels):
            j = 0
            count_plus = 0
            # relevance levels are unique, therefore count can only be 1 or 0
            count_minus = 1 if relevance in indices else 0
            xv_count_plus = 0
            xv_count_minus = numpy.dot(self.x.take(indices.get(relevance, []), axis=0), v).sum()

            for i in range(n_samples):
                if self.y[i] != relevance or not self.status[i]:
                    continue

                while j < n_samples and 1 - self.xw[j] + self.xw[i] > 0:
                    if self.y[j] > relevance:
                        count_plus += 1
                        xv_count_plus += numpy.dot(self.x[j, :], v)
                        l_minus[j] += count_minus
                        xv_minus[j] += xv_count_minus

                    j += 1

                l_plus[i] = count_plus
                xv_plus[i] += xv_count_plus
                count_minus -= 1
                xv_count_minus -= numpy.dot(self.x.take(i, axis=0), v)

        return l_plus, xv_plus, l_minus, xv_minus


class RankSVMOptimizer(six.with_metaclass(ABCMeta, object)):
    """Abstract base class for all optimizers"""
    def __init__(self, alpha, rank_ratio, timeit=False):
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.timeit = timeit

    @abstractmethod
    def _objective_func(self, w):
        pass

    @abstractmethod
    def _update_constraints(self, w):
        pass

    @abstractmethod
    def _gradient_func(self, w):
        pass

    @abstractmethod
    def _hessian_func(self, w, s):
        pass

    @abstractproperty
    def n_features(self):
        pass

    def run(self, **kwargs):
        w = numpy.zeros(self.n_features)

        timings = None
        if self.timeit:
            import timeit

            def _inner():
                return minimize(self._objective_func, w, method='newton-cg', callback=self._update_constraints,
                                jac=self._gradient_func, hessp=self._hessian_func, **kwargs)

            timer = timeit.Timer(_inner)
            timings = timer.repeat(self.timeit, number=1)

        opt_result = minimize(self._objective_func, w, method='newton-cg', callback=self._update_constraints,
                              jac=self._gradient_func, hessp=self._hessian_func, **kwargs)
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
    def n_features(self):
        return self.data_x.shape[1]

    def _objective_func(self, w):
        self._update_constraints(w)
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
    def n_features(self):
        return self.data_x.shape[1]

    def _objective_func(self, w):
        self._update_constraints(w)

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

    counter : object
        Instance of :class:`Counter` subclass.

    References
    ----------
    Lee, C.-P., & Lin, C.-J. (2014). Supplement Materials for "Large-scale linear RankSVM". Neural Computation, 26(4),
        781–817. doi:10.1162/NECO_a_00571
    """
    def __init__(self, alpha, rank_ratio, counter, timeit=False):
        super().__init__(alpha, rank_ratio, timeit)

        self._counter = counter
        self._regr_penalty = (1.0 - rank_ratio) * alpha
        self._rank_penalty = rank_ratio * alpha
        self._has_time = hasattr(self._counter, 'time') and self._regr_penalty > 0

        self._last_w = None
        self._last_gradient = None

    @property
    def n_features(self):
        return self._counter.x.shape[1]

    def _objective_func(self, w):
        self._update_constraints(w)

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(w)
        x = self._counter.x

        xs = numpy.dot(x, w)
        val = 0.5 * squared_norm(w)
        if self._has_time:
            val += 0.5 * self._regr_penalty * squared_norm(self.y_compressed
                                                           - xs.compress(self.regr_mask, axis=0))

        val += 0.5 * self._rank_penalty * numexpr.evaluate(
            'sum(xs * ((l_plus + l_minus) * xs - xv_plus - xv_minus - 2 * (l_minus - l_plus)) + l_minus)')

        return val

    def _update_constraints(self, w):
        self._counter.update_sort_order(w)

        if self._has_time:
            pred_time = self._counter.time - self._counter.xw
            self.regr_mask = (pred_time > 0) | self._counter.status
            self.y_compressed = self._counter.time.compress(self.regr_mask, axis=0)

    def _gradient_func(self, w):
        if self._last_w is not None and (w == self._last_w).all():
            return self._last_gradient

        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(w)
        x = self._counter.x

        xs = numpy.dot(x, w)
        z = numexpr.evaluate('(l_plus + l_minus) * xs - xv_plus - xv_minus - l_minus + l_plus')

        if self._has_time:
            w = w + self._regr_penalty * (fast_dot(x.T, xs)
                                          - fast_dot(x.compress(self.regr_mask, axis=0).T, self.y_compressed))

        self._last_gradient = w + self._rank_penalty * fast_dot(x.T, z)
        self._last_w = w
        return self._last_gradient

    def _hessian_func(self, w, s):
        l_plus, xv_plus, l_minus, xv_minus = self._counter.calculate(s)
        x = self._counter.x

        xs = numpy.dot(x, s)
        xs = numexpr.evaluate('(l_plus + l_minus) * xs - xv_plus - xv_minus')

        if self._has_time:
            s = s + self._regr_penalty * fast_dot(x.T, numpy.dot(x, s))

        return s + self._rank_penalty * fast_dot(x.T, xs)


class FastSurvivalSVM(BaseEstimator):
    """Efficient Training of Survival Support Vector Machine

    Training data consists of *n* triplets :math:`(\mathbf{x}_i, y_i, \delta_i)`,
    where :math:`\mathbf{x}_i` is a *d*-dimensional feature vector, :math:`y_i > 0`
    the survival time or time of censoring, and :math:`\delta_i \in \{0,1\}`
    the binary event indicator. Using the training data, the objective is to
    minimize the following function:

    .. math::

         \\arg \min_{\mathbf{w}, b} \\frac{1}{2} \mathbf{w}^T \mathbf{w}
         + \\frac{\\alpha}{2} \left[ r \sum_{i,j \in \mathcal{P}}
         \max(0, 1 - (\mathbf{w}^T \mathbf{x}_i - \mathbf{w}^T \mathbf{x}_j))^2
         + (1 - r) \sum_{i=0}^n \left( \zeta_{\mathbf{w}, b} (y_i, x_i, \delta_i)
         \\right)^2 \\right]

        \zeta_{\mathbf{w},b} (y_i, \mathbf{x}_i, \delta_i) =
        \\begin{cases}
        \max(0, y_i - \mathbf{w}^T \mathbf{x}_i - b) & \\text{if $\delta_i = 0$,} \\
        y_i - \mathbf{w}^T \mathbf{x}_i - b & \\text{if $\delta_i = 1$,} \\
        \end{cases}

        \mathcal{P} = \{ (i, j)~|~y_i > y_j \land \delta_j = 1 \}_{i,j=1,\dots,n}

    The hyper-parameter :math:`\\alpha > 0` determines the amount of regularization
    to apply: a smaller value increases the amount of regularization and a
    higher value reduces the amount of regularization. The hyper-parameter
    :math:`r \in [0; 1]` determines the trade-off between the ranking objective
    and the regresson objective. If :math:`r = 1` it reduces to the ranking
    objective, and if :math:`r = 0` to the regression objective. If the regression
    objective is used, it is advised to log-transform the survival/censoring
    time first.

    Parameters
    ----------
    alpha : float, positive
        Weight of penalizing the squared hinge loss in the objective function (default: 1)

    rank_ratio : float, optional (default=1.0)
        Mixing parameter between regression and ranking objective with ``0 <= rank_ratio <= 1``.
        If ``rank_ratio = 1``, only ranking is performed, if ``rank_ratio = 0``, only regression
        is performed. A non-zero value is only allowed if optimizer is one of 'avltree', 'PRSVM',
        or 'rbtree'.

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"

    fit_intercept : boolean, optional (default=False)
        Whether to calculate an intercept for the regression model. If set to ``False``, no intercept
        will be calculated. Has no effect if ``rank_ratio = 1``, i.e., only ranking is performed.

    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
        Ignored by other kernels.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as call

    max_iter : int, optional
        Maximum number of iterations to perform in Newton optimization (default: 20)

    verbose : bool, optional
        Whether to print messages during optimization (default: False)

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.

    optimizer : "avltree" | "direct-count" | "PRSVM" | "rbtree" | "simple", optional
        Which optimizer to use. default: avltree

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
           In Proceedings of the European Conference on Machine Learning and
           Principles and Practice of Knowledge Discovery in Databases (ECML PKDD),
           2015.
    """
    def __init__(self, alpha=1, rank_ratio=1.0, kernel="linear", fit_intercept=False,
                 gamma=None, degree=3, coef0=1, kernel_params=None, max_iter=20, verbose=False, tol=None,
                 optimizer=None, random_state=None, timeit=False):
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.kernel = kernel
        self.fit_intercept = fit_intercept
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
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

        if self.fit_intercept:
            X = numpy.column_stack((numpy.ones(X.shape[0]), X))

        times, ranks = y

        if self.optimizer == 'simple':
            optimizer = SimpleOptimizer(X, status, self.alpha, self.rank_ratio, timeit=self.timeit)
        elif self.optimizer == 'PRSVM':
            optimizer = PRSVMOptimizer(X, status, self.alpha, self.rank_ratio, timeit=self.timeit)
        elif self.optimizer == 'direct-count':
            optimizer = LargeScaleOptimizer(self.alpha, self.rank_ratio,
                                            SurvivalCounter(X, ranks, status, len(ranks), times), timeit=self.timeit)
        elif self.optimizer == 'rbtree':
            optimizer = LargeScaleOptimizer(self.alpha, self.rank_ratio,
                                            OrderStatisticTreeSurvivalCounter(X, ranks, status, RBTree, times),
                                            timeit=self.timeit)
        elif self.optimizer == 'avltree':
            optimizer = LargeScaleOptimizer(self.alpha, self.rank_ratio,
                                            OrderStatisticTreeSurvivalCounter(X, ranks, status, AVLTree, times),
                                            timeit=self.timeit)
        else:
            raise ValueError('unknown optimizer: {0}'.format(self.optimizer))

        return optimizer

    def fit(self, X, y):
        """Build a survival support vector machine model from training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data matrix.

        y : structered array, shape = [n_samples]
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        if not (0 <= self.rank_ratio <= 1):
            raise ValueError("rank_ratio must be in [0; 1]")

        if self.rank_ratio < 1.0 and self.optimizer in {'simple', 'PRSVM'}:
            raise ValueError("optimizer '%s' does not implement regression objective" % self.optimizer)

        X, event, time = check_arrays_survival(X, y)

        if self.kernel == 'linear':
            X_transform = X
            self.transform_ = None
        else:
            self.transform_ = KernelPCA(kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0,
                                        kernel_params=self.kernel_params)
            X_transform = self.transform_.fit_transform(X)

        random_state = check_random_state(self.random_state)
        samples_order = FastSurvivalSVM._argsort_and_resolve_ties(time, random_state)
        data_y = (time[samples_order], numpy.arange(len(samples_order)))
        status = event[samples_order]

        optimizer = self._create_optimizer(X_transform[samples_order, :], data_y, status)
        opt_result = optimizer.run(tol=self.tol, options={'maxiter': self.max_iter, 'disp': self.verbose})
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
        if self.transform_ is None:
            X_transform = X
        else:
            X_transform = self.transform_.transform(X)

        val = numpy.dot(X_transform, self.coef_)
        if hasattr(self, "intercept_"):
             val += self.intercept_

        return -val
