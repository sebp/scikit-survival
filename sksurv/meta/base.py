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

from sklearn.utils.metaestimators import _safe_split


def _fit_and_score(est, x, y, scorer, train_index, test_index, parameters, fit_params, predict_params):
    """Train survival model on given data and return its score on test data"""
    X_train, y_train = _safe_split(est, x, y, train_index)
    train_params = fit_params.copy()

    # Training
    est.set_params(**parameters)
    est.fit(X_train, y_train, **train_params)

    # Testing
    test_predict_params = predict_params.copy()
    X_test, y_test = _safe_split(est, x, y, test_index, train_index)

    score = scorer(est, X_test, y_test, **test_predict_params)
    if not isinstance(score, numbers.Number):
        raise ValueError(f"scoring must return a number, got {score!s} ({type(score)}) instead.")

    return score
