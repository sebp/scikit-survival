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
import numpy as np
from sklearn.linear_model import Ridge

from ..base import SurvivalAnalysisMixin
from ..nonparametric import ipc_weights
from ..util import check_array_survival


class IPCRidge(Ridge, SurvivalAnalysisMixin):
    """Accelerated failure time model with inverse probability of censoring weights.

    This model assumes a regression model of the form

    .. math::

        \\log y = \\beta_0 + \\mathbf{X} \\beta + \\epsilon

    L2-shrinkage is applied to the coefficients :math:`\\beta` and
    each sample is weighted by the inverse probability of censoring
    to account for right censoring (under the assumption that
    censoring is independent of the features, i.e., random censoring).

    See [1]_ for further description.

    Parameters
    ----------
    alpha : float, optional, default: 1.0
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.
        `alpha` must be a non-negative float i.e. in `[0, inf)`.

        For numerical reasons, using `alpha = 0` is not advised.

    fit_intercept : bool, default: True
        Whether to fit the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. ``X`` and ``y`` are expected to be centered).

    copy_X : bool, default: True
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, default: None
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
        For 'lbfgs' solver, the default value is 15000.

    tol : float, default: 1e-4
        Precision of the solution. Note that `tol` has no effect for solvers 'svd' and
        'cholesky'.

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', \
            'sag', 'saga', 'lbfgs'}, default: 'auto'
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. It is the most stable solver, in particular more stable
          for singular matrices than 'cholesky' at the cost of being slower.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.

        - 'lbfgs' uses L-BFGS-B algorithm implemented in
          `scipy.optimize.minimize`. It can be used only when `positive`
          is True.

        All solvers except 'svd' support both dense and sparse data. However, only
        'lsqr', 'sag', 'sparse_cg', and 'lbfgs' support sparse input when
        `fit_intercept` is True.

    positive : bool, default: False
        When set to ``True``, forces the coefficients to be positive.
        Only 'lbfgs' solver is supported in this case.

    random_state : int, RandomState instance, default: None
        Used when ``solver`` == 'sag' or 'saga' to shuffle the data.

    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Weight vector.

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    n_iter_ : None or ndarray of shape (n_targets,)
        Actual number of iterations for each target. Available only for
        sag and lsqr solvers. Other solvers will return None.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] W. Stute, "Consistent estimation under random censorship when covariables are
           present", Journal of Multivariate Analysis, vol. 45, no. 1, pp. 89-103, 1993.
           doi:10.1006/jmva.1993.1028.
    """

    _parameter_constraints = {**Ridge._parameter_constraints}

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-3,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )

    @property
    def _predict_risk_score(self):
        return False

    def fit(self, X, y):
        """Build an accelerated failure time model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_array_survival(X, y)

        weights = ipc_weights(event, time)
        super().fit(X, np.log(time), sample_weight=weights)

        return self

    def predict(self, X):
        """Predict using the linear accelerated failure time model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values on original scale (NOT log scale).
        """
        return np.exp(super().predict(X))

    def score(self, X, y, sample_weight=None):
        return SurvivalAnalysisMixin.score(self, X, y)
