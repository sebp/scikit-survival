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


class SurvivalAnalysisMixin(object):

    def score(self, X, y):
        from .metrics import concordance_index_censored
        name_event, name_time = y.dtype.names

        result = concordance_index_censored(y[name_event], y[name_time], self.predict(X))
        return result[0]
