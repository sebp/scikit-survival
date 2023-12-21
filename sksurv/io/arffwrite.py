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

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype

_ILLEGAL_CHARACTER_PAT = re.compile(r"[^-_=\w\d\(\)<>\.]")


def writearff(data, filename, relation_name=None, index=True):
    """Write ARFF file

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        DataFrame containing data

    filename : string or file-like object
        Path to ARFF file or file-like object. In the latter case,
        the handle is closed by calling this function.

    relation_name : string, optional, default: "pandas"
        Name of relation in ARFF file.

    index : boolean, optional, default: True
        Write row names (index)
    """
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

        if isinstance(series.dtype, CategoricalDtype) or is_object_dtype(series):
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
        if pd.isnull(x):
            return "?"
        return str(x)

    data = data.applymap(to_str)
    n_rows = data.shape[0]
    for i in range(n_rows):
        str_values = list(data.iloc[i, :].apply(_check_str_array))
        line = ",".join(str_values)
        fp.write(line)
        fp.write("\n")
