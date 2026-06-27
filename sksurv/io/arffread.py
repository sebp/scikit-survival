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
from pandas.api.types import is_string_dtype
from scipy.io.arff import loadarff as scipy_loadarff

__all__ = ["loadarff"]


def _to_pandas_dataframe(data, meta):
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
            dtype = "str" if is_string_dtype(arr.dtype) else arr.dtype
            p = pd.Series(arr, dtype=dtype)
            data_dict[name] = p

    # This step converts any pandas.Categorical columns back to pandas.Series.
    return pd.DataFrame.from_dict(data_dict)


def _decode_nominal_value(b):
    if b != b"?":
        return b.decode()


def _to_polars_dataframe(data, meta):
    # Constructed with the polars API directly (rather than via narwhals)
    # because ARFF nominal attributes declare a closed category list in
    # the header that may include labels absent from the data, and
    # ``pl.Enum(list(attr_format))`` is the canonical way to preserve
    # that declared list verbatim. Routing the same construction through
    # ``nw.new_series(..., dtype=nw.Enum(...))`` does not guarantee that
    # declared-but-unseen labels survive the round-trip,
    # which would corrupt ARFF write-back. Keeping the polars import
    # localized here also prevents the dataframe boundary modules in
    # ``sksurv._dataframe`` from acquiring backend-specific dependencies.
    import polars as pl

    columns = []
    for name in sorted(meta.names()):
        tp, attr_format = meta[name]
        if tp == "nominal":
            values = [_decode_nominal_value(b) for b in data[name]]
            dtype = pl.Enum(list(attr_format))
        else:
            arr = data[name]
            if is_string_dtype(arr.dtype):
                # scipy returns bytes for string-typed attributes; decode + null-map.
                values = [_decode_nominal_value(b) for b in arr]
                dtype = pl.String
            else:
                values = arr
                dtype = None
        columns.append(pl.Series(name, values, dtype=dtype))
    return pl.DataFrame(columns)


def loadarff(filename, *, output_type="pandas"):
    """Load ARFF file.

    Parameters
    ----------
    filename : str or file-like
        Path to ARFF file, or file-like object to read from.
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library for the returned frame. Nominal columns become
        ``pd.Categorical(ordered=False)`` (pandas) or
        ``pl.Enum(declared_categories)`` (polars), preserving the full
        declared category list including labels absent from the data.

    Returns
    -------
    data_frame : :class:`pandas.DataFrame` or :class:`polars.DataFrame`
        DataFrame containing data of the ARFF file. The dataframe library follows
        ``output_type``.

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
    >>> # Load the ARFF file as pandas (default)
    >>> with StringIO(arff_content) as f:
    ...     data = loadarff(f)
    >>>
    >>> print(data)
      class  feature1  feature2
    0     A       1.0       2.0
    1     B       3.0       4.0
    2     C       5.0       6.0

    Load as polars; nominal columns become ``pl.Enum``:

    >>> with StringIO(arff_content) as f:
    ...     data_pl = loadarff(f, output_type="polars")
    >>> data_pl["class"].dtype
    Enum(categories=['A', 'B', 'C'])
    """
    data, meta = scipy_loadarff(filename)
    if output_type == "pandas":
        return _to_pandas_dataframe(data, meta)
    if output_type == "polars":
        return _to_polars_dataframe(data, meta)
    raise ValueError(f"output_type must be 'pandas' or 'polars', got {output_type!r}")
