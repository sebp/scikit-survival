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
import itertools

import numpy
from scipy.misc import comb
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

from ..base import SurvivalAnalysisMixin
from ..util import check_arrays_survival


class NaiveSurvivalSVM(LinearSVC, SurvivalAnalysisMixin):
    """Naive version of linear Survival Support Vector Machine.

    Uses regular linear support vector classifier (liblinear).
    A new set of samples is created by building the difference between any two feature
    vectors in the original data, thus this version requires `O(n_samples^2)` space.

    See :class:`sksurv.svm.HingeLossSurvivalSVM` for the kernel naive survival SVM.

    .. math::

          \\min_{\\mathbf{w}}\\quad
          \\frac{1}{2} \\lVert \\mathbf{w} \\rVert_2^2
          + \\gamma \\sum_{i = 1}^n \\xi_i \\\\
          \\text{subject to}\\quad
          \\mathbf{w}^\\top \\mathbf{x}_i - \\mathbf{w}^\\top \\mathbf{x}_j \\geq 1 - \\xi_{ij},\\quad
          \\forall (i, j) \\in \\mathcal{P}, \\\\
          \\xi_i \\geq 0,\\quad \\forall (i, j) \\in \\mathcal{P}.

          \\mathcal{P} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1 \\}_{i,j=1,\\dots,n}.

    Parameters
    ----------
    alpha : float, positive (default=1.0)
        Weight of penalizing the squared hinge loss in the objective function (default: 1)

    loss : string, 'hinge' or 'squared_hinge' (default='squared_hinge')
        Specifies the loss function. 'hinge' is the standard SVM loss
        (used e.g. by the SVC class) while 'squared_hinge' is the
        square of the hinge loss.

    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to `coef_`
        vectors that are sparse.

    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.

    verbose : int (default=0)
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    max_iter : int (default=1000)
        The maximum number of iterations to be run.

    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A., & Van Huffel, S.
           Support Vector Machines for Survival Analysis. In Proc. of the 3rd Int. Conf.
           on Computational Intelligence in Medicine and Healthcare (CIMED). 1-8. 2007

    .. [2] Evers, L., Messow, C.M.,
           "Sparse kernel methods for high-dimensional survival data",
           Bioinformatics 24(14), 1632-8, 2008.

    """
    def __init__(self, penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                 alpha=1.0, verbose=0, random_state=None, max_iter=1000):
        super().__init__(penalty=penalty,
                         loss=loss,
                         dual=dual,
                         tol=tol,
                         verbose=verbose,
                         random_state=random_state,
                         max_iter=max_iter,
                         fit_intercept=False)
        self.alpha = alpha

    def _get_survival_pairs(self, X, y, random_state):
        X, event, time = check_arrays_survival(X, y)

        idx = numpy.arange(X.shape[0], dtype=int)
        random_state.shuffle(idx)

        n_pairs = int(comb(X.shape[0], 2))
        x_pairs = numpy.empty((n_pairs, X.shape[1]), dtype=float)
        y_pairs = numpy.empty(n_pairs, dtype=numpy.int8)
        k = 0
        for xi, xj in itertools.combinations(idx, 2):
            if time[xi] > time[xj] and event[xj]:
                x_pairs[k, :] = X[xi, :] - X[xj, :]
                y_pairs[k] = 1
                k += 1
            elif time[xi] < time[xj] and event[xi]:
                x_pairs[k, :] = X[xi, :] - X[xj, :]
                y_pairs[k] = -1
                k += 1
            elif time[xi] == time[xj] and (event[xi] or event[xj]):
                x_pairs[k, :] = X[xi, :] - X[xj, :]
                y_pairs[k] = 1 if event[xj] else -1
                k += 1

        x_pairs.resize((k, X.shape[1]))
        y_pairs.resize(k)
        return x_pairs, y_pairs

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
        random_state = check_random_state(self.random_state)

        x_pairs, y_pairs = self._get_survival_pairs(X, y, random_state)

        self.C = self.alpha
        return super().fit(x_pairs, y_pairs)

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
        return -self.decision_function(X)
