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
from numpy.testing import assert_array_equal, assert_almost_equal
from sksurv.metrics import concordance_index_censored


def assert_cindex_almost_equal(event_indicator, event_time, estimate, expected):
    result = concordance_index_censored(event_indicator, event_time, estimate)
    assert_array_equal(result[1:], expected[1:])
    concordant, discordant, tied_risk = result[1:4]
    cc = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
    assert_almost_equal(result[0], cc)
    assert_almost_equal(result[0], expected[0])
