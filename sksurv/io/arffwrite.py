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
import os.path
import re

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_string_dtype

from .._dataframe import ensure_eager_dataframe, polars_inputs

_ILLEGAL_CHARACTER_PAT = re.compile(r"[^-_=\w\d\(\)<>\.]")


def _prepare_polars_for_arff_write(data):
    """Convert a polars DataFrame to pandas, preserving ``pl.Enum``
    declared categories. Column-by-column to avoid an undeclared ``pyarrow``
    dependency (``nw_df.to_pandas()`` would dispatch Categorical/Enum through Arrow).
    """
    nw_df = nw.from_native(data)
    columns = {}
    for col_name in nw_df.columns:
        col = nw_df.get_column(col_name)
        dtype = col.dtype
        if isinstance(dtype, nw.Enum):
            categories = col.cat.get_categories().to_list()
            columns[col_name] = pd.Categorical(col.to_list(), categories=categories)
        elif isinstance(dtype, nw.Categorical):
            categories = sorted(col.drop_nulls().unique())
            columns[col_name] = pd.Categorical(col.to_list(), categories=categories)
        else:
            columns[col_name] = col.to_list()
    return pd.DataFrame(columns)


def writearff(data, filename, relation_name=None, index=True):
    """Write ARFF file

    Parameters
    ----------
    data : :class:`pandas.DataFrame` or :class:`polars.DataFrame`
        Polars input is converted to pandas internally; ``pl.Enum`` columns
        keep their declared categories (incl. unseen labels) in the ARFF header.

    filename : str or file-like object
        Path to ARFF file or file-like object. In the latter case,
        the handle is closed by calling this function.

    relation_name : str, optional, default: 'pandas'
        Name of relation in ARFF file.

    index : boolean, optional, default: True
        Write row names (index). Only relevant for pandas input; other
        dataframe libraries have no row-index concept, so the value is ignored.

    See Also
    --------
    loadarff : Function to read ARFF files.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sksurv.io import writearff
    >>>
    >>> # Create a dummy DataFrame
    >>> data = pd.DataFrame({
    ...     'feature1': [1.0, 3.0, 5.0],
    ...     'feature2': [2.0, np.nan, 6.0],
    ...     'class': ['A', 'B', 'C']
    ... }, index=['One', 'Two', 'Three'])
    >>>
    >>> # Write to a temporary directory so the CWD stays clean.
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "data.arff"
    ...     writearff(data, str(path), relation_name='test_data')
    ...     print(path.read_text())
    @relation test_data
    <BLANKLINE>
    @attribute index        {One,Three,Two}
    @attribute feature1     real
    @attribute feature2     real
    @attribute class        {A,B,C}
    <BLANKLINE>
    @data
    One,1.0,2.0,A
    Two,3.0,?,B
    Three,5.0,6.0,C

    Polars input is accepted as well. ``pl.Enum`` columns preserve their
    declared category list (including labels absent from the data) in the
    resulting ARFF header.

    >>> import polars as pl
    >>> data_pl = pl.DataFrame({
    ...     'feature1': [1.0, 3.0, 5.0],
    ...     'class': pl.Series(['A', 'B', 'C'], dtype=pl.Enum(['A', 'B', 'C'])),
    ... })
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "data.arff"
    ...     writearff(data_pl, str(path), relation_name='test_data')
    """
    data = ensure_eager_dataframe(data)
    if polars_inputs.LIBRARY.is_dataframe(data):
        data = _prepare_polars_for_arff_write(data)
        # Polars frames have no meaningful row index; suppress it so the
        # output does not gain a spurious ``index`` attribute.
        index = False

    if isinstance(filename, str):
        fp = open(filename, "w")

        if relation_name is None:
            relation_name = os.path.basename(filename)
    else:
        fp = filename

        if relation_name is None:
            relation_name = "pandas"

    try:
        data = _write_header(data, fp, relation_name, index)
        fp.write("\n")
        _write_data(data, fp)
    finally:
        fp.close()


def _write_header(data, fp, relation_name, index):
    """Write header containing attribute names and types"""
    fp.write(f"@relation {relation_name}\n\n")

    if index:
        data = data.reset_index()

    attribute_names = _sanitize_column_names(data)

    for column, series in data.items():
        name = attribute_names[column]
        fp.write(f"@attribute {name}\t")

        if isinstance(series.dtype, CategoricalDtype) or is_string_dtype(series.dtype):
            _write_attribute_categorical(series, fp)
        elif np.issubdtype(series.dtype, np.floating):
            fp.write("real")
        elif np.issubdtype(series.dtype, np.integer):
            fp.write("integer")
        elif np.issubdtype(series.dtype, np.datetime64):
            fp.write("date 'yyyy-MM-dd HH:mm:ss'")
        else:
            raise TypeError(f"unsupported type {series.dtype}")

        fp.write("\n")
    return data


def _sanitize_column_names(data):
    """Replace illegal characters with underscore"""
    new_names = {}
    for name in data.columns:
        new_names[name] = _ILLEGAL_CHARACTER_PAT.sub("_", name)
    return new_names


def _check_str_value(x):
    """If string has a space, wrap it in double quotes and remove/escape illegal characters"""
    if isinstance(x, str):
        # remove commas, and single quotation marks since loadarff cannot deal with it
        x = x.replace(",", ".").replace(chr(0x2018), "'").replace(chr(0x2019), "'")

        # put string in double quotes
        if " " in x:
            if x[0] in ('"', "'"):
                x = x[1:]
            if x[-1] in ('"', "'"):
                x = x[: len(x) - 1]
            x = '"' + x.replace('"', '\\"') + '"'
    return str(x)


_check_str_array = np.frompyfunc(_check_str_value, 1, 1)


def _write_attribute_categorical(series, fp):
    """Write categories of a categorical/nominal attribute"""
    if isinstance(series.dtype, CategoricalDtype):
        categories = series.cat.categories
        string_values = _check_str_array(categories)
    else:
        categories = series.dropna().unique()
        string_values = sorted(_check_str_array(categories), key=lambda x: x.strip('"'))

    values = ",".join(string_values)
    fp.write("{")
    fp.write(values)
    fp.write("}")


def _write_data(data, fp):
    """Write the data section"""
    fp.write("@data\n")

    def to_str(x):
        if pd.isna(x):
            return "?"
        return str(x)

    data = data.map(to_str)
    # The header has already been written; coerce rows to object values so
    # categorical-only frames can pass through ``_check_str_array``.
    data = data.astype(object)
    n_rows = data.shape[0]
    for i in range(n_rows):
        str_values = list(data.iloc[i, :].apply(_check_str_array))
        line = ",".join(str_values)
        fp.write(line)
        fp.write("\n")
