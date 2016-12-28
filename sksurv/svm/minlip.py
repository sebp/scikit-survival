import numpy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels

from ..base import SurvivalAnalysisMixin
from ..util import check_arrays_survival
from ._minlip import create_difference_matrix

__all__ = ['MinlipSurvivalAnalysis', 'HingeLossSurvivalSVM']


class MinlipSurvivalAnalysis(BaseEstimator, SurvivalAnalysisMixin):
    """Survival model related to survival SVM, using a minimal Lipschitz smoothness strategy
    instead of a maximal margin strategy.

    .. math::

          \\min_{\\mathbf{w}}\\quad
          \\frac{1}{2} \\lVert \\mathbf{w} \\rVert_2^2
          + \\gamma \\sum_{i = 1}^n \\xi_i \\\\
          \\text{subject to}\\quad
          \\mathbf{w}^\\top \\mathbf{x}_i - \\mathbf{w}^\\top \\mathbf{x}_j \\geq y_i - y_j - \\xi_i,\\quad
          \\forall (i, j) \\in \\mathcal{P}_\\text{1-NN}, \\\\
          \\xi_i \\geq 0,\\quad \\forall i = 1,\\dots,n.

          \\mathcal{P}_\\text{1-NN} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1
          \\land \\nexists k : y_i > y_k > y_j \\land \\delta_k = 1 \\}_{i,j=1}^n.

    Parameters
    ----------
    solver : "cvxpy" | "cvxopt", optional (default: cvxpy)
        Which quadratic program solver to use.

    alpha : float, positive  (default: 1)
        Weight of penalizing the hinge loss in the objective function.

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
        Ignored by other kernels.

    degree : int (default=3)
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as call

    pairs : "all" | "nearest" | "next", optional (default: "nearest")
        Which constraints to use in the optimization problem.

        - all: Use all comparable pairs. Scales quadratic in number of samples
          (cf. :class:`sksurv.svm.HingeLossSurvivalSVM`).
        - nearest: Only considers comparable pairs :math:`(i, j)` where :math:`j` is the
          uncensored sample with highest survival time smaller than :math:`y_i`.
          Scales linear in number of samples.
        - next: Only compare against direct nearest neighbor according to observed time,
          disregarding its censoring status. Scales linear in number of samples.

    verbose : bool (default: False)
        Enable verbose output of solver

    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``timings_`` attribute.

    Attributes
    ----------
    `X_fit_` :
        Training data.

    `coef_` :
        Coefficients of the features in the decision function.

    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A. K., and Van Huffel, S.
           Learning transformation models for ranking and survival analysis.
           The Journal of Machine Learning Research, 12, 819-862. 2011
    """

    def __init__(self, solver="cvxpy",
                 alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None,
                 pairs="nearest", verbose=False, timeit=None):
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

    def _fit(self, x, event, time):
        D = create_difference_matrix(event.astype(numpy.uint8), time, kind=self.pairs)
        K = self._get_kernel(x)

        if self.solver == "cvxpy":
            fit_func = self._fit_cvxpy
        elif self.solver == "cvxopt":
            fit_func = self._fit_cvxopt
        else:
            raise ValueError("unknown solver: {}".format(self.solver))

        if self.timeit is not None:
            import timeit

            def _inner():
                return fit_func(K, D, time)

            timer = timeit.Timer(_inner)
            self.timings_ = timer.repeat(self.timeit, number=1)

        coef, sv = fit_func(K, D, time)

        if sv is None:
            self.coef_ = coef * D
        else:
            self.coef_ = coef[:, sv] * D[sv, :]
        self.X_fit_ = x

    def _fit_cvxpy(self, K, D, time):
        import cvxpy

        n_pairs = D.shape[0]

        a = cvxpy.Variable(n_pairs)
        P = D.dot(D.dot(K).T).T
        q = D.dot(time)

        obj = cvxpy.Minimize(0.5 * cvxpy.quad_form(a, P) - a.T * q)
        assert obj.is_dcp()

        alpha = cvxpy.Parameter(sign="positive", value=self.alpha)
        constraints = [a >= 0., -alpha <= D.T * a, D.T * a <= alpha]

        prob = cvxpy.Problem(obj, constraints)
        prob.solve(verbose=self.verbose)

        return a.value.T.A, None

    def _fit_cvxopt(self, K, D, time):
        import cvxopt

        n_samples = K.shape[0]

        P = D.dot(D.dot(K).T).T
        q = -D.dot(time)

        high = numpy.repeat(self.alpha, n_samples * 2)

        n_pairs = D.shape[0]
        G = sparse.vstack((D.T, -D.T, -sparse.eye(n_pairs)))
        h = numpy.concatenate((high, numpy.zeros(n_pairs)))

        Gsp = cvxopt.matrix(G.toarray())
        # Gsp = cvxopt.spmatrix(G.data, G.row, G.col, G.shape)

        cvxopt.solvers.options["show_progress"] = int(self.verbose)
        sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), Gsp, cvxopt.matrix(h))

        return numpy.array(sol['x']).T, None

    def fit(self, X, y):
        """Build a MINLIP survival model from training data.

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
        self._fit(X, event, time)

        return self

    def predict(self, X):
        """Predict risk score of experiencing an event.

        Higher scores indicate shorter survival (high risk),
        lower scores longer survival (low risk).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Predicted risk.
        """
        K = self._get_kernel(X, self.X_fit_)
        pred = -numpy.dot(self.coef_, K.T)
        return pred.ravel()


class HingeLossSurvivalSVM(MinlipSurvivalAnalysis):
    """Naive implementation of kernel survival support vector machine.

    A new set of samples is created by building the difference between any two feature
    vectors in the original data, thus this version requires :math:`O(\\text{n_samples}^4)` space and
    :math:`O(\\text{n_samples}^6 \\cdot \\text{n_features})`.

    See :class:`sksurv.svm.NaiveSurvivalSVM` for the linear naive survival SVM based on liblinear.

    .. math::

          \\min_{\\mathbf{w}}\\quad
          \\frac{1}{2} \\lVert \\mathbf{w} \\rVert_2^2
          + \\gamma \\sum_{i = 1}^n \\xi_i \\\\
          \\text{subject to}\\quad
          \\mathbf{w}^\\top \\phi(\\mathbf{x})_i - \\mathbf{w}^\\top \\phi(\\mathbf{x})_j \\geq 1 - \\xi_{ij},\\quad
          \\forall (i, j) \\in \\mathcal{P}, \\\\
          \\xi_i \\geq 0,\\quad \\forall (i, j) \\in \\mathcal{P}.

          \\mathcal{P} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1 \\}_{i,j=1,\\dots,n}.

    Parameters
    ----------
    solver : "cvxpy" | "cvxopt", optional (default: cvxpy)
        Which quadratic program solver to use.

    alpha : float, positive
        Weight of penalizing the hinge loss in the objective function (default: 1)

    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
        Ignored by other kernels.

    degree : int (default=3)
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as call

    pairs : "all" | "nearest" | "next", optional (default: "all")
        Which constraints to use in the optimization problem.

        - all: Use all comparable pairs. Scales quadratic in number of samples.
        - nearest: Only considers comparable pairs :math:`(i, j)` where :math:`j` is the
          uncensored sample with highest survival time smaller than :math:`y_i`.
          Scales linear in number of samples (cf. :class:`sksurv.svm.MinlipSurvivalSVM`).
        - next: Only compare against direct nearest neighbor according to observed time,
          disregarding its censoring status. Scales linear in number of samples.

    verbose : bool (default: False)
        Enable verbose output of solver.

    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``timings_`` attribute.

    Attributes
    ----------
    `X_fit_` :
        Training data.

    `coef_` :
        Coefficients of the features in the decision function.

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

    def __init__(self, solver="cvxpy",
                 alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None,
                 pairs="all", verbose=False, timeit=None):
        super().__init__(solver=solver, alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
                         kernel_params=kernel_params, pairs=pairs, verbose=verbose, timeit=timeit)

    def _fit_cvxpy(self, K, D, time):
        import cvxpy

        n_pairs = D.shape[0]

        a = cvxpy.Variable(n_pairs)
        alpha = cvxpy.Parameter(sign="positive", value=self.alpha)
        P = D.dot(D.dot(K).T).T

        obj = cvxpy.Minimize(0.5 * cvxpy.quad_form(a, P) - cvxpy.sum_entries(a))
        constraints = [a >= 0., a <= alpha]

        prob = cvxpy.Problem(obj, constraints)
        prob.solve(verbose=self.verbose)

        coef = a.value.T.A
        sv = numpy.flatnonzero(coef > 1e-5)
        return coef, sv

    def _fit_cvxopt(self, K, D, time):
        import cvxopt

        n_pairs = D.shape[0]

        P = D.dot(D.dot(K).T).T
        q = -numpy.ones(n_pairs)

        G = numpy.vstack((-numpy.eye(n_pairs), numpy.eye(n_pairs)))
        h = numpy.concatenate((numpy.zeros(n_pairs), numpy.repeat(self.alpha, n_pairs)))

        cvxopt.solvers.options["show_progress"] = int(self.verbose)
        sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h))

        coef = numpy.array(sol['x']).T
        sv = numpy.flatnonzero(coef > 1e-5)
        return coef, sv
