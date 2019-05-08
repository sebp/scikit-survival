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
import numpy
from sklearn.exceptions import ConvergenceWarning
import warnings

from ._coxnet import call_fit_group_penalty


class GroupOrthonormalizer:

    def fit(self, X, groups):
        X = numpy.asarray(X)

        # standardize
        X_mean = numpy.mean(X, axis=0)
        X_std = numpy.std(X, ddof=1, axis=0)
        assert numpy.all(X_std > 0)
        X = (X - X_mean) / X_std

        sqrt_samples = numpy.sqrt(X.shape[0])
        transforms = []
        for i in range(len(groups) - 1):
            start = groups[i]
            end = groups[i + 1]
            assert start < end
            assert end <= X.shape[1]

            U, s, V = numpy.linalg.svd(X[:, start:end], full_matrices=False)
            r = numpy.flatnonzero(s > 1e-10)

            # ensure 1/n * X_j^T @ X_j is identity matrix
            T = V[:, r].T * sqrt_samples / s[r]
            transforms.append(T)

        self.groups_ = groups
        self.transforms_ = transforms
        self.mean_ = X_mean
        self.scale_ = X_std

        return X

    def transform(self, X):
        X = (numpy.asarray(X) - self.mean_) / self.scale_
        groups = self.groups_

        assert X.shape[1] == groups[-1]

        transformed = []
        for i in range(len(groups) - 1):
            start = groups[i]
            end = groups[i + 1]

            XX = X[:, start:end].dot(self.transforms_[i])
            transformed.append(XX)

        return numpy.column_stack(transformed)

    def inverse_transform(self, coef):
        from scipy import sparse

        cc = sparse.csc_matrix(coef)
        T = sparse.block_diag(self.transforms_).dot(cc)

        coef_new = T.toarray()

        # unstandardize
        coef_new = coef_new / self.scale_[:, numpy.newaxis]
        return coef_new


def group_penalty_path(X, time, event_num, groups, penalty, alphas, create_path,
                       alpha_min_ratio, l1_ratio, max_iter, tol, verbose):
    # TODO: sort rows by time (descending)
    # see CoxnetSurvivalAnalysis._pre_fit
    # TODO group lasso implies l1_ratio=1.0

    orthonormer = GroupOrthonormalizer()
    orthonormer.fit(X, groups)
    x_norm = numpy.asfortranarray(orthonormer.transform(X))

    coefs, alphas_, n_iter = call_fit_group_penalty(
        x_norm,
        time,
        event_num,
        groups,
        penalty,
        alphas,
        create_path,
        alpha_min_ratio,
        l1_ratio,
        max_iter,
        tol,
        verbose)

    coefs = orthonormer.inverse_transform(coefs)
    assert numpy.isfinite(coefs).all()

    if n_iter >= max_iter:
        warnings.warn('Optimization terminated early, you might want'
                      ' to increase the number of iterations (max_iter=%d).'
                      % max_iter,
                      category=ConvergenceWarning,
                      stacklevel=2)

    return coefs, alphas_.ravel(), n_iter
