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

from numpy.testing import assert_almost_equal, assert_array_equal

import sksurv
from sksurv.base import SurvivalAnalysisMixin
from sksurv.metrics import concordance_index_censored


def assert_cindex_almost_equal(event_indicator, event_time, estimate, expected):
    result = concordance_index_censored(event_indicator, event_time, estimate)
    assert_array_equal(result[1:], expected[1:])
    concordant, discordant, tied_risk = result[1:4]
    cc = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
    assert_almost_equal(result[0], cc)
    assert_almost_equal(result[0], expected[0])


def _is_survival_mixin(x):
    return inspect.isclass(x) and x is not SurvivalAnalysisMixin and issubclass(x, SurvivalAnalysisMixin)


def all_survival_estimators():
    root = str(Path(sksurv.__file__).parent)
    all_classes = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(path=[root], prefix="sksurv."):
        # meta-estimators require base estimators
        if modname.startswith("sksurv.meta"):
            continue
        module = import_module(modname)
        for _name, cls in inspect.getmembers(module, _is_survival_mixin):
            if inspect.isabstract(cls):
                continue
            all_classes.append(cls)
    return set(all_classes)
