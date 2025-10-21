from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import validate_data

from ..base import SurvivalAnalysisMixin
from ..exceptions import NoComparablePairException
from ..util import check_array_survival
from ._minlip import create_difference_matrix

__all__ = ["MinlipSurvivalAnalysis", "HingeLossSurvivalSVM"]


class QPSolver(metaclass=ABCMeta):
    r"""Abstract base class for quadratic program solvers.

    This class defines the interface for solvers that minimize a quadratic
    objective function subject to linear inequality constraints,
    formulated as:

    .. math::

        \min_{x} \quad (1/2)x^T P x + q^T x \\
        \text{subject to} \quad G x \preceq h

    Parameters
    ----------
    max_iter : int or None
        Maximum number of iterations to perform.
    verbose : bool
        Enable verbose output of the solver.
    """

    @abstractmethod
    def __init__(self, max_iter, verbose):
        self.max_iter = max_iter
        self.verbose = verbose

    @abstractmethod
    def solve(self, P, q, G, h):
        """Find solution to QP.

        Parameters
        ----------
        P : array-like, shape=(n_variables, n_variables)
            Quadratic part of the objective function.
        q : array-like, shape=(n_variables,)
            Linear part of the objective function.
        G : array-like, shape=(n_constraints, n_variables)
            Matrix for inequality constraints.
        h : array-like, shape=(n_constraints,)
            Vector for inequality constraints.

        Returns
        -------
        x : ndarray, shape=(n_variables,)
            The optimal solution.
        n_iter : int
            Number of iterations performed by the solver.
        """


class OsqpSolver(QPSolver):
    def __init__(self, max_iter, verbose):
        super().__init__(
            max_iter=max_iter,
            verbose=verbose,
        )

    def solve(self, P, q, G, h):
        import osqp

        P = sparse.csc_matrix(P)

        solver_opts = self._get_options()
        m = osqp.OSQP()
        m.setup(P=sparse.csc_matrix(P), q=q, A=G, l=None, u=h, **solver_opts)  # noqa: E741
        results = m.solve(raise_error=False)

        solved_codes = (
            osqp.SolverStatus.OSQP_SOLVED,
            osqp.SolverStatus.OSQP_SOLVED_INACCURATE,
        )

        if results.info.status_val == osqp.SolverStatus.OSQP_MAX_ITER_REACHED:  # max iter reached
            warnings.warn(
                (f"OSQP solver did not converge: {results.info.status}"),
                category=ConvergenceWarning,
                stacklevel=2,
            )
        elif results.info.status_val not in solved_codes:  # pragma: no cover
            # none of SOLVED, SOLVED_INACCURATE
            raise RuntimeError(f"OSQP solver failed: {results.info.status}")

        n_iter = results.info.iter
        return results.x[np.newaxis], n_iter

    def _get_options(self):
        """Returns a dictionary of OSQP solver options."""
        solver_opts = {
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": self.max_iter or 4000,
            "polishing": True,
            "verbose": self.verbose,
        }
        return solver_opts


class EcosSolver(QPSolver):
    r"""Solves QP by expressing it as second-order cone program:

    .. math::

        \min \quad c^T x \\
        \text{subject to} \quad G x \preceq_K h

    where the last inequality is generalized, i.e. :math:`h - G x`
    belongs to the cone :math:`K`.

    Parameters
    ----------
    max_iter : int or None
        Maximum number of iterations to perform.
    verbose : bool
        Enable verbose output of the solver.
    cond : float or None, default: None
        Condition number for eigenvalue decomposition.
    """

    EXIT_OPTIMAL = 0  # Optimal solution found
    EXIT_PINF = 1  # Certificate of primal infeasibility found
    EXIT_DINF = 2  # Certificate of dual infeasibility found
    EXIT_MAXIT = -1  # Maximum number of iterations reached
    EXIT_NUMERICS = -2  # Numerical problems (unreliable search direction)
    EXIT_OUTCONE = -3  # Numerical problems (slacks or multipliers outside cone)
    EXIT_INACC_OFFSET = 10

    def __init__(self, max_iter, verbose, cond=None):
        super().__init__(
            max_iter=max_iter,
            verbose=verbose,
        )
        self.cond = cond

    def solve(self, P, q, G, h):
        import ecos

        n_pairs = P.shape[0]
        L, max_eigval = self._decompose(P)

        # minimize wrt t,x
        c = np.empty(n_pairs + 1)
        c[1:] = q
        c[0] = 0.5 * max_eigval

        zerorow = np.zeros((1, L.shape[1]))
        G_quad = np.block(
            [
                [-1, zerorow],
                [1, zerorow],
                [np.zeros((L.shape[0], 1)), -2 * L],
            ]
        )
        G_lin = sparse.hstack((sparse.csc_matrix((G.shape[0], 1)), G))
        G_all = sparse.vstack((G_lin, sparse.csc_matrix(G_quad)), format="csc")

        n_constraints = G.shape[0]
        h_all = np.empty(G_all.shape[0])
        h_all[:n_constraints] = h
        h_all[n_constraints : (n_constraints + 2)] = 1
        h_all[(n_constraints + 2) :] = 0

        dims = {
            "l": G.shape[0],  # scalar, dimension of positive orthant
            "q": [G_quad.shape[0]],  # vector with dimensions of second order cones
        }
        results = ecos.solve(c, G_all, h_all, dims, verbose=self.verbose, max_iters=self.max_iter or 1000)
        self._check_success(results)

        # drop solution for t
        x = results["x"][1:]
        n_iter = results["info"]["iter"]
        return x[np.newaxis], n_iter

    def _check_success(self, results):  # pylint: disable=no-self-use
        """Checks if the ECOS solver converged successfully.

        Parameters
        ----------
        results : dict
            The results dictionary returned by ``ecos.solve``.

        Raises
        -------
        RuntimeError
            If the solver failed for an unknown reason or found primal/dual infeasibility.
        """
        exit_flag = results["info"]["exitFlag"]
        if exit_flag in (EcosSolver.EXIT_OPTIMAL, EcosSolver.EXIT_OPTIMAL + EcosSolver.EXIT_INACC_OFFSET):
            return

        if exit_flag == EcosSolver.EXIT_MAXIT:
            warnings.warn(
                "ECOS solver did not converge: maximum iterations reached", category=ConvergenceWarning, stacklevel=3
            )
        elif exit_flag == EcosSolver.EXIT_PINF:  # pragma: no cover
            raise RuntimeError("Certificate of primal infeasibility found")
        elif exit_flag == EcosSolver.EXIT_DINF:  # pragma: no cover
            raise RuntimeError("Certificate of dual infeasibility found")
        else:  # pragma: no cover
            raise RuntimeError(f"Unknown problem in ECOS solver, exit status: {exit_flag}")

    def _decompose(self, P):
        """Performs eigenvalue decomposition of P.

        Parameters
        ----------
        P : array-like, shape=(n_variables, n_variables)
            Quadratic part of the objective function.

        Returns
        -------
        decomposed : ndarray
            Decomposed matrix.
        largest_eigenvalue : float
            The largest eigenvalue of P.
        """
        # from scipy.linalg.pinvh
        s, u = linalg.eigh(P)
        largest_eigenvalue = np.max(np.abs(s))

        cond = self.cond
        if cond is None:
            t = u.dtype
            cond = largest_eigenvalue * max(P.shape) * np.finfo(t).eps

        not_below_cutoff = abs(s) > -cond
        assert not_below_cutoff.all(), f"matrix has negative eigenvalues: {s.min()}"

        above_cutoff = abs(s) > cond
        u = u[:, above_cutoff]
        s = s[above_cutoff]

        # set maximum eigenvalue to 1
        decomposed = u * np.sqrt(s / largest_eigenvalue)
        return decomposed.T, largest_eigenvalue


class MinlipSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    r"""Survival model based on a minimal Lipschitz smoothness strategy.

    This model is related to :class:`sksurv.svm.FastKernelSurvivalSVM` but
    minimizes a different objective function, focusing on Lipschitz
    smoothness rather than maximal margin. The optimization problem is
    formulated as:

    .. math::

            \min_{\mathbf{w}}\quad
        \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
        + \gamma \sum_{i = 1}^n \xi_i \\
        \text{subject to}\quad
        \mathbf{w}^\top \mathbf{x}_i - \mathbf{w}^\top \mathbf{x}_j \geq y_i - y_j - \xi_i,\quad
        \forall (i, j) \in \mathcal{P}_\text{1-NN}, \\
        \xi_i \geq 0,\quad \forall i = 1,\dots,n.

        \mathcal{P}_\text{1-NN} = \{ (i, j) \mid y_i > y_j \land \delta_j = 1
        \land \nexists k : y_i > y_k > y_j \land \delta_k = 1 \}_{i,j=1}^n.

    See [1]_ for further description.

    Parameters
    ----------
    alpha : float, optional, default: 1
        Weight of penalizing the hinge loss in the objective function.
        Must be greater than 0.

    solver : {'ecos', 'osqp'}, optional, default: 'ecos'
        Which quadratic program solver to use.

    kernel : str or callable, optional, default: 'linear'.
        Kernel mapping used internally. This parameter is directly passed to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`.
        If `kernel` is a string, it must be one of the metrics
        in `sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS` or "precomputed".
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    gamma : float, optional, default: None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for :mod:`sklearn.metrics.pairwise`.
        Ignored by other kernels.

    degree : int, optional, default: 3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, optional, default: 1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict, optional, default: None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    pairs : {'all', 'nearest', 'next'}, optional, default: 'nearest'
        Which constraints to use in the optimization problem.

        - all: Use all comparable pairs. Scales quadratically in number of samples
          (cf. :class:`sksurv.svm.HingeLossSurvivalSVM`).
        - nearest: Only considers comparable pairs :math:`(i, j)` where :math:`j` is the
          uncensored sample with highest survival time smaller than :math:`y_i`.
          Scales linearly in number of samples.
        - next: Only compare against direct nearest neighbor according to observed time,
          disregarding its censoring status. Scales linearly in number of samples.

    verbose : bool, optional, default: False
        Enable verbose output of solver.

    timeit : bool, int, or None, optional, default: False
        If ``True`` or a non-zero integer, the time taken for optimization is measured.
        If an integer is provided, the optimization is repeated that many times.
        Results can be accessed from the ``timings_`` attribute.

    max_iter : int or None, optional, default: None
        The maximum number of iterations taken for the solvers to converge.
        If ``None``, use solver's default value.

    Attributes
    ----------
    X_fit_ : ndarray, shape = (n_samples, `n_features_in_`)
        Training data.

    coef_ : ndarray, shape = (n_samples,), dtype = float
        Coefficients of the features in the decision function.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray, shape = (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.

    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A. K., and Van Huffel, S.
           Learning transformation models for ranking and survival analysis.
           The Journal of Machine Learning Research, 12, 819-862. 2011
    """

    _parameter_constraints = {
        "solver": [StrOptions({"ecos", "osqp"})],
        "alpha": [Interval(numbers.Real, 0, None, closed="neither")],
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS.keys()) | {"precomputed"}),
            callable,
        ],
        "degree": [Interval(numbers.Integral, 0, None, closed="left")],
        "gamma": [Interval(numbers.Real, 0.0, None, closed="left"), None],
        "coef0": [Interval(numbers.Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "pairs": [StrOptions({"all", "nearest", "next"})],
        "verbose": ["boolean"],
        "timeit": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "max_iter": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        alpha=1.0,
        *,
        solver="ecos",
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        pairs="nearest",
        verbose=False,
        timeit=None,
        max_iter=None,
    ):
        self.solver = solver
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.pairs = pairs
        self.verbose = verbose
        self.timeit = timeit
        self.max_iter = max_iter

    def __sklearn_tags__(self):
        # tell sklearn.utils.metaestimators._safe_split function that we expect kernel matrix
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def _setup_qp(self, K, D, time):
        n_pairs = D.shape[0]
        P = D.dot(D.dot(K).T).T
        q = -D.dot(time)

        Dt = D.T.astype(P.dtype)  # cast constraints to correct type
        G = sparse.vstack(
            (
                Dt,  # upper bound
                -Dt,  # lower bound
                -sparse.eye(n_pairs, dtype=P.dtype),  # lower bound >= 0
            ),
            format="csc",
        )
        n_constraints = Dt.shape[0]
        h = np.empty(G.shape[0], dtype=float)
        h[: 2 * n_constraints] = self.alpha
        h[-n_pairs:] = 0.0

        return {"P": P, "q": q, "G": G, "h": h}

    def _fit(self, x, event, time):
        D = create_difference_matrix(event.astype(np.uint8), time, kind=self.pairs)
        if D.shape[0] == 0:
            raise NoComparablePairException("Data has no comparable pairs, cannot fit model.")

        max_iter = self.max_iter
        if self.solver == "ecos":
            solver = EcosSolver(max_iter=max_iter, verbose=self.verbose)
        elif self.solver == "osqp":
            solver = OsqpSolver(max_iter=max_iter, verbose=self.verbose)

        K = self._get_kernel(x)
        problem_data = self._setup_qp(K, D, time)

        if self.timeit is not None:
            import timeit

            def _inner():
                return solver.solve(**problem_data)

            timer = timeit.Timer(_inner)
            self.timings_ = timer.repeat(self.timeit, number=1)

        coef, n_iter = solver.solve(**problem_data)
        self._update_coef(coef, D)
        self.n_iter_ = n_iter
        self.X_fit_ = x

    def _update_coef(self, coef, D):
        self.coef_ = coef * D

    def fit(self, X, y):
        """Build a MINLIP survival model from training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        y : structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event and ``False`` indicates right-censoring.
            The second field is a float with the time of event or time of censoring.

        Returns
        -------
        self
        """
        self._validate_params()
        X = validate_data(self, X, ensure_min_samples=2)
        event, time = check_array_survival(X, y)
        self._fit(X, event, time)

        return self

    def predict(self, X):
        """Predict risk score of experiencing an event.

        Higher values indicate an increased risk of experiencing an event,
        lower values a decreased risk of experiencing an event. The scores
        have no unit and are only meaningful to rank samples by their risk
        of experiencing an event.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape = (n_samples,)
            Predicted risk.
        """
        X = validate_data(self, X, reset=False)
        K = self._get_kernel(X, self.X_fit_)
        pred = -np.dot(self.coef_, K.T)
        return pred.ravel()


class HingeLossSurvivalSVM(MinlipSurvivalAnalysis):
    r"""Naive implementation of kernel survival support vector machine.

    This implementation creates a new set of samples by building the difference
    between any two feature vectors in the original data. This approach
    requires :math:`O(\text{n_samples}^4)` space and
    :math:`O(\text{n_samples}^6 \cdot \text{n_features})` time, making it
    computationally intensive for large datasets.

    The optimization problem is formulated as:

    .. math::

        \min_{\mathbf{w}}\quad
        \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
        + \gamma \sum_{i = 1}^n \xi_i \\
        \text{subject to}\quad
        \mathbf{w}^\top \phi(\mathbf{x})_i - \mathbf{w}^\top \phi(\mathbf{x})_j \geq 1 - \xi_{ij},\quad
        \forall (i, j) \in \mathcal{P}, \\
        \xi_i \geq 0,\quad \forall (i, j) \in \mathcal{P}.

        \mathcal{P} = \{ (i, j) \mid y_i > y_j \land \delta_j = 1 \}_{i,j=1,\dots,n}.

    See [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    alpha : float, optional, default: 1
        Weight of penalizing the hinge loss in the objective function. Must be greater than 0.

    solver : {'ecos', 'osqp'}, optional, default: 'ecos'
        Which quadratic program solver to use.

    kernel : str or callable, optional, default: 'linear'
        Kernel mapping used internally. This parameter is directly passed to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`.
        If `kernel` is a string, it must be one of the metrics
        in `sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS` or "precomputed".
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    gamma : float or None, optional, default: None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for :mod:`sklearn.metrics.pairwise`.
        Ignored by other kernels.

    degree : int, optional, default: 3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, optional, default: 1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict or None, optional, default: None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    pairs : {'all', 'nearest', 'next'}, optional, default: 'all'
        Which constraints to use in the optimization problem.

        - all: Use all comparable pairs. Scales quadratically in number of samples.
        - nearest: Only considers comparable pairs :math:`(i, j)` where :math:`j` is the
          uncensored sample with highest survival time smaller than :math:`y_i`.
          Scales linearly in number of samples (cf. :class:`sksurv.svm.MinlipSurvivalAnalysis`).
        - next: Only compare against direct nearest neighbor according to observed time,
          disregarding its censoring status. Scales linearly in number of samples.

    verbose : bool, optional, default: False
        If ``True``, enable verbose output of the solver.

    timeit : bool, int, or None, optional, default: False
        If ``True`` or a non-zero integer, the time taken for optimization is measured.
        If an integer is provided, the optimization is repeated that many times.
        Results can be accessed from the ``timings_`` attribute.

    max_iter : int or None, optional, default: None
        The maximum number of iterations taken for the solvers to converge.
        If ``None``, use solver's default value.

    Attributes
    ----------
    X_fit_ : ndarray, shape = (n_samples, `n_features_in_`)
        Training data.

    coef_ : ndarray, shape = (n_samples,), dtype = float
        Coefficients of the features in the decision function.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray, shape = (`n_features_in_`,), dtype = object
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.

    See also
    --------
    sksurv.svm.NaiveSurvivalSVM : The linear naive survival SVM based on liblinear.

    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A., & Van Huffel, S.
           Support Vector Machines for Survival Analysis. In Proc. of the 3rd Int. Conf.
           on Computational Intelligence in Medicine and Healthcare (CIMED). 1-8. 2007

    .. [2] Evers, L., Messow, C.M.,
           "Sparse kernel methods for high-dimensional survival data",
           Bioinformatics 24(14), 1632-8, 2008.

    .. [3] Van Belle, V., Pelckmans, K., Suykens, J.A., Van Huffel, S.,
           "Survival SVM: a practical scalable algorithm",
           In: Proc. of 16th European Symposium on Artificial Neural Networks,
           89-94, 2008.
    """

    _parameter_constraints = MinlipSurvivalAnalysis._parameter_constraints

    def __init__(
        self,
        alpha=1.0,
        *,
        solver="ecos",
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        pairs="all",
        verbose=False,
        timeit=None,
        max_iter=None,
    ):
        super().__init__(
            solver=solver,
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            pairs=pairs,
            verbose=verbose,
            timeit=timeit,
            max_iter=max_iter,
        )

    def _setup_qp(self, K, D, time):
        n_pairs = D.shape[0]

        P = D.dot(D.dot(K).T).T
        q = -np.ones(n_pairs)

        G = sparse.vstack((-sparse.eye(n_pairs), sparse.eye(n_pairs)), format="csc")
        h = np.empty(2 * n_pairs)
        h[:n_pairs] = 0
        h[n_pairs:] = self.alpha

        return {"P": P, "q": q, "G": G, "h": h}

    def _update_coef(self, coef, D):
        sv = np.flatnonzero(coef > 1e-5)
        self.coef_ = coef[:, sv] * D[sv, :]
