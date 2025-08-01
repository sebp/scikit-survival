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

import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.svm import LinearSVC
from sklearn.utils.validation import _get_feature_names, check_random_state, validate_data

from ..base import SurvivalAnalysisMixin
from ..exceptions import NoComparablePairException
from ..util import check_array_survival


class NaiveSurvivalSVM(SurvivalAnalysisMixin, LinearSVC):
    r"""Naive implementation of linear Survival Support Vector Machine.

    This class uses a regular linear support vector classifier (liblinear)
    to implement a survival SVM. It constructs a new dataset by computing
    the difference between feature vectors of comparable pairs from the
    original data. This approach results in a space complexity of
    :math:`O(\text{n_samples}^2)`.

    The optimization problem is formulated as:

    .. math::

        \min_{\mathbf{w}}\quad
        \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
        + \gamma \sum_{i = 1}^n \xi_i \\
        \text{subject to}\quad
        \mathbf{w}^\top \mathbf{x}_i - \mathbf{w}^\top \mathbf{x}_j \geq 1 - \xi_{ij},\quad
        \forall (i, j) \in \mathcal{P}, \\
        \xi_i \geq 0,\quad \forall (i, j) \in \mathcal{P}.

        \mathcal{P} = \{ (i, j) \mid y_i > y_j \land \delta_j = 1 \}_{i,j=1,\dots,n}.

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    alpha : float, optional, default: 1.0
        Weight of penalizing the squared hinge loss in the objective function. Must be greater than 0.

    loss : {'hinge', 'squared_hinge'}, optional,default: 'squared_hinge'
        Specifies the loss function. 'hinge' is the standard SVM loss
        (used e.g. by the SVC class) while 'squared_hinge' is the
        square of the hinge loss.

    penalty : {'l1', 'l2'}, optional,default: 'l2'
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to `coef_`
        vectors that are sparse.

    dual : bool, optional,default: True
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    tol : float, optional, default: 1e-4
        Tolerance for stopping criteria.

    verbose : int, optional, default: 0
        If ``True``, enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int, :class:`numpy.random.RandomState` instance, or None, optional, default: None
        Used to resolve ties in survival times. Pass an int for reproducible output across
        multiple :meth:`fit` calls.

    max_iter : int, optional, default: 1000
        The maximum number of iterations taken for the solver to converge.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations run by the optimization routine to fit the model.

    See also
    --------
    sksurv.svm.FastSurvivalSVM : Alternative implementation with reduced time complexity for training.
    sksurv.svm.HingeLossSurvivalSVM : Non-linear version of the naive survival SVM based on kernel functions.

    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A., & Van Huffel, S.
           Support Vector Machines for Survival Analysis. In Proc. of the 3rd Int. Conf.
           on Computational Intelligence in Medicine and Healthcare (CIMED). 1-8. 2007

    .. [2] Evers, L., Messow, C.M.,
           "Sparse kernel methods for high-dimensional survival data",
           Bioinformatics 24(14), 1632-8, 2008.

    """

    _parameter_constraints = {
        "penalty": LinearSVC._parameter_constraints["penalty"],
        "loss": LinearSVC._parameter_constraints["loss"],
        "dual": LinearSVC._parameter_constraints["dual"],
        "tol": LinearSVC._parameter_constraints["tol"],
        "alpha": LinearSVC._parameter_constraints["C"],
        "verbose": LinearSVC._parameter_constraints["verbose"],
        "random_state": LinearSVC._parameter_constraints["random_state"],
        "max_iter": LinearSVC._parameter_constraints["max_iter"],
    }

    def __init__(
        self,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual=False,
        tol=1e-4,
        alpha=1.0,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        super().__init__(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
            fit_intercept=False,
        )
        self.alpha = alpha

    def _get_survival_pairs(self, X, y, random_state):  # pylint: disable=no-self-use
        """Generates comparable pairs from survival data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            and time of event or time of censoring.
        random_state : RandomState instance
            Random number generator used for shuffling.

        Returns
        -------
        x_pairs : ndarray, shape = (n_pairs, n_features)
            Feature differences for comparable pairs.
        y_pairs : ndarray, shape = (n_pairs,)
            Labels for comparable pairs (1 or -1).

        Raises
        ------
        NoComparablePairException
            If no comparable pairs can be formed from the input data.
        """
        feature_names = _get_feature_names(X)

        X = validate_data(self, X, ensure_min_samples=2)
        event, time = check_array_survival(X, y)

        idx = np.arange(X.shape[0], dtype=int)
        random_state.shuffle(idx)

        n_pairs = int(comb(X.shape[0], 2))
        x_pairs = np.empty((n_pairs, X.shape[1]), dtype=float)
        y_pairs = np.empty(n_pairs, dtype=np.int8)
        k = 0
        for xi, xj in itertools.combinations(idx, 2):
            if time[xi] > time[xj] and event[xj]:
                np.subtract(X[xi, :], X[xj, :], out=x_pairs[k, :])
                y_pairs[k] = 1
                k += 1
            elif time[xi] < time[xj] and event[xi]:
                np.subtract(X[xi, :], X[xj, :], out=x_pairs[k, :])
                y_pairs[k] = -1
                k += 1
            elif time[xi] == time[xj] and (event[xi] or event[xj]):
                np.subtract(X[xi, :], X[xj, :], out=x_pairs[k, :])
                y_pairs[k] = 1 if event[xj] else -1
                k += 1

        x_pairs.resize((k, X.shape[1]), refcheck=False)
        y_pairs.resize(k, refcheck=False)

        if feature_names is not None:
            x_pairs = pd.DataFrame(x_pairs, columns=feature_names)
        return x_pairs, y_pairs

    def fit(self, X, y, sample_weight=None):
        """Build a survival support vector machine model from training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        y : structured array, shape = (n_samples,)
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event and ``False`` indicates right-censoring.
            The second field is a float with the time of event or time of censoring.

        sample_weight : array-like, shape = (n_samples,), optional
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.

        Returns
        -------
        self
        """
        random_state = check_random_state(self.random_state)

        x_pairs, y_pairs = self._get_survival_pairs(X, y, random_state)
        if x_pairs.shape[0] == 0:
            raise NoComparablePairException("Data has no comparable pairs, cannot fit model.")

        self.C = self.alpha
        return super().fit(x_pairs, y_pairs, sample_weight=sample_weight)

    def predict(self, X):
        """Predict risk scores.

        Predictions are risk scores (i.e. higher values indicate an
        increased risk of experiencing an event). The scores have no
        unit and are only meaningful to rank samples by their risk
        of experiencing an event.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features,)
            The input samples.

        Returns
        -------
        y : ndarray, shape = (n_samples,), dtype = float
            Predicted risk scores.
        """
        return -self.decision_function(X)
