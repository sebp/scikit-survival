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
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff as scipy_loadarff

__all__ = ["loadarff"]


def _to_pandas(data, meta):
    data_dict = {}
    attrnames = sorted(meta.names())
    for name in attrnames:
        tp, attr_format = meta[name]
        if tp == "nominal":
            raw = []
            for b in data[name]:
                # replace missing values with NaN
                if b == b"?":
                    raw.append(np.nan)
                else:
                    raw.append(b.decode())

            data_dict[name] = pd.Categorical(raw, categories=attr_format, ordered=False)
        else:
            arr = data[name]
            p = pd.Series(arr, dtype=arr.dtype)
            data_dict[name] = p

    # currently, this step converts all pandas.Categorial columns back to pandas.Series
    return pd.DataFrame.from_dict(data_dict)


def loadarff(filename):
    """Load ARFF file.

    Parameters
    ----------
    filename : str or file-like
        Path to ARFF file, or file-like object to read from.

    Returns
    -------
    data_frame : :class:`pandas.DataFrame`
        DataFrame containing data of ARFF file

    See Also
    --------
    scipy.io.arff.loadarff : The underlying function that reads the ARFF file.

    Examples
    --------
    >>> from io import StringIO
    >>> from sksurv.io import loadarff
    >>>
    >>> # Create a dummy ARFF file
    >>> arff_content = '''
    ... @relation test_data
    ... @attribute feature1 numeric
    ... @attribute feature2 numeric
    ... @attribute class {A,B,C}
    ... @data
    ... 1.0,2.0,A
    ... 3.0,4.0,B
    ... 5.0,6.0,C
    ... '''
    >>>
    >>> # Load the ARFF file
    >>> with StringIO(arff_content) as f:
    ...     data = loadarff(f)
    >>>
    >>> print(data)
      class  feature1  feature2
    0     A       1.0       2.0
    1     B       3.0       4.0
    2     C       5.0       6.0
    """
    data, meta = scipy_loadarff(filename)
    return _to_pandas(data, meta)
