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
from scipy.io.arff import loadarff as scipy_loadarff
import pandas

__all__ = ["loadarff"]


def _to_pandas(data, meta):
    data_dict = {}
    attrnames = meta.names()
    for name in attrnames:
        tp, attr_format = meta[name]
        if tp == "nominal":
            raw = []
            for b in data[name]:
                # replace missing values with NaN
                if b == b'?':
                    raw.append(numpy.nan)
                else:
                    raw.append(b.decode())

            data_dict[name] = pandas.Categorical(raw, categories=attr_format, ordered=False)
        else:
            arr = data[name]
            p = pandas.Series(arr, dtype=arr.dtype)
            data_dict[name] = p

    # currently, this step converts all pandas.Categorial columns back to pandas.Series
    return pandas.DataFrame(data_dict)


def loadarff(filename):
    """Load ARFF file

    Parameters
    ----------
    filename : string
        Path to ARFF file

    Returns
    -------
    data_frame : :class:`pandas.DataFrame`
        DataFrame containing data of ARFF file
    """
    data, meta = scipy_loadarff(filename)
    return _to_pandas(data, meta)
