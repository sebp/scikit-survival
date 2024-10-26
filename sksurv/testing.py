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
from importlib import import_module
import inspect
from pathlib import Path
import pkgutil

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

import sksurv
from sksurv.metrics import concordance_index_censored


def assert_cindex_almost_equal(event_indicator, event_time, estimate, expected):
    result = concordance_index_censored(event_indicator, event_time, estimate)
    assert_array_equal(result[1:], expected[1:])
    concordant, discordant, tied_risk = result[1:4]
    cc = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
    assert_almost_equal(result[0], cc)
    assert_almost_equal(result[0], expected[0])


def assert_survival_function_properties(surv_fns):
    if not np.isfinite(surv_fns).all():
        raise AssertionError("survival function contains values that are not finite")
    if np.any(surv_fns < 0.0):
        raise AssertionError("survival function contains negative values")
    if np.any(surv_fns > 1.0):
        raise AssertionError("survival function contains values larger 1")

    d = np.apply_along_axis(np.diff, 1, surv_fns)
    if np.any(d > 0):
        raise AssertionError("survival functions are not monotonically decreasing")

    # survival function at first time point
    num_closer_to_zero = np.sum(1.0 - surv_fns[:, 0] >= surv_fns[:, 0])
    if num_closer_to_zero / surv_fns.shape[0] > 0.5:
        raise AssertionError(f"most ({num_closer_to_zero}) probabilities at first time point are closer to 0 than 1")

    # survival function at last time point
    num_closer_to_one = np.sum(1.0 - surv_fns[:, -1] < surv_fns[:, -1])
    if num_closer_to_one / surv_fns.shape[0] > 0.5:
        raise AssertionError(f"most ({num_closer_to_one}) probabilities at last time point are closer to 1 than 0")


def assert_chf_properties(chf):
    if not np.isfinite(chf).all():
        raise AssertionError("chf contains values that are not finite")
    if np.any(chf < 0.0):
        raise AssertionError("chf contains negative values")

    d = np.apply_along_axis(np.diff, 1, chf)
    if np.any(d < 0):
        raise AssertionError("chf are not monotonically increasing")

    # chf at first time point
    num_closer_to_one = np.sum(1.0 - chf[:, 0] < chf[:, 0])
    if num_closer_to_one / chf.shape[0] > 0.5:
        raise AssertionError(f"most ({num_closer_to_one}) hazard rates at first time point are closer to 1 than 0")


def _is_survival_estimator(x):
    return (
        inspect.isclass(x)
        and issubclass(x, BaseEstimator)
        and not issubclass(x, TransformerMixin)
        and x.__module__.startswith("sksurv.")
        and not x.__name__.startswith("_")
        and x.__module__.split(".", 2)[1] not in {"metrics", "nonparametric"}
    )


def all_survival_estimators():
    root = str(Path(sksurv.__file__).parent)
    all_classes = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(path=[root], prefix="sksurv."):
        # meta-estimators require base estimators
        if modname.startswith("sksurv.meta"):
            continue
        module = import_module(modname)
        for _name, cls in inspect.getmembers(module, _is_survival_estimator):
            if inspect.isabstract(cls):
                continue
            all_classes.append(cls)
    return set(all_classes)


class FixtureParameterFactory:
    def get_cases(self):
        cases = []
        for name, func in inspect.getmembers(self):
            if name.startswith("data_"):
                values = func()
                cases.append(pytest.param(*values, id=name))
        return cases
