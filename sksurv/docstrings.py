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
_PRED_SURV_FN_EXAMPLE_DOC = """
        .. plot::

            >>> import matplotlib.pyplot as plt
            >>> from sksurv.datasets import load_veterans_lung_cancer
            >>> from sksurv.preprocessing import OneHotEncoder
            >>> from sksurv.{estimator_mod} import {estimator_class}

            Load the data and encode categorical features.

            >>> X, y = load_veterans_lung_cancer()
            >>> Xt = OneHotEncoder().fit_transform(X)

            Fit the model.

            >>> estimator = {estimator_class}().fit(Xt, y)

            Estimate the survival function for the first 10 samples.

            >>> surv_funcs = estimator.predict_survival_function(Xt.iloc[:10])

            Plot the estimated survival functions.

            >>> for fn in surv_funcs:
            ...     plt.step(fn.x, fn(fn.x), where="post")
            ...
            [...]
            >>> plt.ylim(0, 1)
            (0.0, 1.0)
            >>> plt.show()  # doctest: +SKIP
"""

_PRED_CUMHAZ_FN_EXAMPLE_DOC = """
        .. plot::

            >>> import matplotlib.pyplot as plt
            >>> from sksurv.datasets import load_veterans_lung_cancer
            >>> from sksurv.preprocessing import OneHotEncoder
            >>> from sksurv.{estimator_mod} import {estimator_class}

            Load the data and encode categorical features.

            >>> X, y = load_veterans_lung_cancer()
            >>> Xt = OneHotEncoder().fit_transform(X)

            Fit the model.

            >>> estimator = {estimator_class}().fit(Xt, y)

            Estimate the cumulative hazard function for the first 10 samples.

            >>> chf_funcs = estimator.predict_cumulative_hazard_function(Xt.iloc[:10])

            Plot the estimated cumulative hazard functions.

            >>> for fn in chf_funcs:
            ...     plt.step(fn.x, fn(fn.x), where="post")
            ...
            [...]
            >>> plt.show()  # doctest: +SKIP
"""


def append_survival_function_example(*, estimator_mod, estimator_class):
    """Append example of using predict_survival_function to API doc"""

    def func(f):
        f.__doc__ += _PRED_SURV_FN_EXAMPLE_DOC.format(
            estimator_mod=estimator_mod,
            estimator_class=estimator_class,
        )
        return f

    return func


def append_cumulative_hazard_example(*, estimator_mod, estimator_class):
    """Append example of using predict_cumulative_hazard_function to API doc"""

    def func(f):
        f.__doc__ += _PRED_CUMHAZ_FN_EXAMPLE_DOC.format(
            estimator_mod=estimator_mod,
            estimator_class=estimator_class,
        )
        return f

    return func
