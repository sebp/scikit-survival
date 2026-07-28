from io import StringIO
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sksurv.io import loadarff, writearff
from sksurv.io.arffread import _to_pandas_dataframe, _to_polars_dataframe
from sksurv.testing import FixtureParameterFactory
from sksurv.testing.dataframe import PANDAS_BACKEND, POLARS_BACKEND

EXPECTED_1 = [
    "@relation test_nominal\n",
    "\n",
    "@attribute attr_nominal\t{beer,water,wine}\n",
    '@attribute attr_nominal_spaces\t{"hard liquor",mate,"red wine"}\n',
    "\n",
    "@data\n",
    'water,"red wine"\n',
    'wine,"hard liquor"\n',
    "beer,?\n",
    "?,mate\n",
    'wine,"hard liquor"\n',
    "water,mate\n",
]


EXPECTED_NO_QUOTES = [
    "@relation test_nominal\n",
    "\n",
    "@attribute attr_nominal\t{beer,water,wine}\n",
    "@attribute attr_nominal_spaces\t{hard liquor,mate,red wine}\n",
    "\n",
    "@data\n",
    "water,red wine\n",
    "wine,hard liquor\n",
    "beer,?\n",
    "?,mate\n",
    "wine,hard liquor\n",
    "water,mate\n",
]


EXPECTED_DATETIME = [
    "@relation test_datetime\n",
    "\n",
    "@attribute attr_datetime\tdate 'yyyy-MM-dd HH:mm:ss'\n",
    "\n",
    "@data\n",
    '"2014-10-31 14:13:01"\n',
    '"2004-03-13 19:49:31"\n',
    '"1998-12-06 09:10:11"\n',
]


ARFF_WITH_UNSEEN_NOMINAL = """\
@relation declared_unseen
@attribute age numeric
@attribute grade {I, II, III, IV}
@data
45, I
60, II
72, III
55, I
68, II
50, III
40, I
70, II
"""


class WriteArffCases(FixtureParameterFactory):
    @property
    def nominal_values(self):
        return {
            "attr_nominal": ["water", "wine", "beer", None, "wine", "water"],
            "attr_nominal_spaces": ["red wine", "hard liquor", None, "mate", "hard liquor", "mate"],
        }

    @property
    def nominal_values_with_quotes(self):
        values = self.nominal_values
        values["attr_nominal_spaces"] = ["'red wine'", "'hard liquor'", None, "mate", "'hard liquor'", "mate"]
        return values

    def data_nominal(self, backend):
        data = backend.make_frame(self.nominal_values)
        return data, "test_nominal", EXPECTED_1.copy()

    def data_nominal_with_quotes(self, backend):
        data = backend.make_frame(self.nominal_values_with_quotes)
        return data, "test_nominal", EXPECTED_1.copy()

    def data_nominal_declared_categories(self, backend):
        categories = {
            "attr_nominal": ["beer", "water", "wine"],
            "attr_nominal_spaces": ["'hard liquor'", "'red wine'", "mate"],
        }
        data = backend.make_frame(self.nominal_values_with_quotes, categories=categories)

        expected = EXPECTED_1.copy()
        expected[3] = '@attribute attr_nominal_spaces\t{"hard liquor","red wine",mate}\n'
        return data, "test_nominal", expected

    def data_nominal_declared_unseen_category(self, backend):
        categories = {
            "attr_nominal": ["beer", "coke", "water", "wine"],
            "attr_nominal_spaces": ["'hard liquor'", "'red wine'", "mate"],
        }
        data = backend.make_frame(self.nominal_values_with_quotes, categories=categories)

        expected = EXPECTED_1.copy()
        expected[2] = "@attribute attr_nominal\t{beer,coke,water,wine}\n"
        expected[3] = '@attribute attr_nominal_spaces\t{"hard liquor","red wine",mate}\n'
        return data, "test_nominal", expected

    def data_nominal_declared_category_ordering(self, backend):
        categories = {
            "attr_nominal": ["water", "coke", "beer", "wine"],
            "attr_nominal_spaces": ["'hard liquor'", "'red wine'", "mate"],
        }
        data = backend.make_frame(self.nominal_values_with_quotes, categories=categories)

        expected = EXPECTED_1.copy()
        expected[2] = "@attribute attr_nominal\t{water,coke,beer,wine}\n"
        expected[3] = '@attribute attr_nominal_spaces\t{"hard liquor","red wine",mate}\n'
        return data, "test_nominal", expected

    def data_datetime(self, backend):
        # ms resolution: polars accepts only ms/us/ns datetime64 values
        values = np.array(["2014-10-31 14:13:01", "2004-03-13 19:49:31", "1998-12-06 09:10:11"], dtype="datetime64[ms]")
        data = backend.make_frame({"attr_datetime": values})
        return data, "test_datetime", EXPECTED_DATETIME.copy()


@pytest.mark.parametrize("case", WriteArffCases().get_cases_func())
def test_writearff(case, dataframe_backend_with_pandas_options, temp_file):
    data_frame, relation_name, expectation = case(dataframe_backend_with_pandas_options)

    writearff(data_frame, temp_file, relation_name=relation_name, index=False)

    with open(temp_file.name) as fp:
        read_data = fp.readlines()
    assert expectation == read_data


class LoadArffCases(FixtureParameterFactory):
    def data_nominal(self, backend):
        contents = "".join(EXPECTED_NO_QUOTES)
        expected = backend.make_frame(
            {
                "attr_nominal": ["water", "wine", "beer", None, "wine", "water"],
                "attr_nominal_spaces": ["red wine", "hard liquor", None, "mate", "hard liquor", "mate"],
            },
            categories={
                "attr_nominal": ["beer", "water", "wine"],
                "attr_nominal_spaces": ["hard liquor", "mate", "red wine"],
            },
        )
        return contents, expected

    def data_declared_unseen_category(self, backend):
        expected = backend.make_frame(
            {
                "age": [45.0, 60.0, 72.0, 55.0, 68.0, 50.0, 40.0, 70.0],
                "grade": ["I", "II", "III", "I", "II", "III", "I", "II"],
            },
            categories={"grade": ["I", "II", "III", "IV"]},
        )
        return ARFF_WITH_UNSEEN_NOMINAL, expected

    def data_missing_nominal(self, backend):
        contents = "@relation missing\n@attribute grade {I, II, III}\n@data\nI\n?\nII\n"
        expected = backend.make_frame({"grade": ["I", None, "II"]}, categories={"grade": ["I", "II", "III"]})
        return contents, expected


@pytest.mark.parametrize("case", LoadArffCases().get_cases_func())
def test_loadarff_frame(case, dataframe_backend_with_pandas_options):
    backend = dataframe_backend_with_pandas_options
    contents, expected = case(backend)

    with StringIO(contents) as fp:
        actual = loadarff(fp, output_type=backend.name)

    backend.assert_frame_equal(actual, expected)


def test_writearff_round_trip(dataframe_backend, temp_file):
    """The declared category list (including unseen labels) survives a
    writearff/loadarff round trip in every library."""
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df = loadarff(fp, output_type=dataframe_backend.name)

    writearff(df, temp_file, relation_name="round_trip", index=False)
    temp_file.close()

    with open(temp_file.name) as fp:
        contents = fp.read()
    assert "{I,II,III,IV}" in contents

    df_round = loadarff(temp_file.name, output_type=dataframe_backend.name)
    dataframe_backend.assert_frame_equal(df_round, df)


@pytest.mark.parametrize(
    "to_frame,backend",
    [
        pytest.param(_to_pandas_dataframe, PANDAS_BACKEND, id="pandas"),
        pytest.param(_to_polars_dataframe, POLARS_BACKEND, id="polars"),
    ],
)
def test_arff_string_attribute_decodes_missing_values(to_frame, backend):
    # SciPy's public loadarff rejects string attributes, but the converters
    # still need to handle the record-array shape SciPy uses for non-nominal
    # fields: bytes are decoded and the b"?" token becomes a missing value.
    class Meta:
        @staticmethod
        def names():
            return ["note"]

        def __getitem__(self, key):
            assert key == "note"
            return "string", None

    data = np.array([(b"hello",), (b"?",), (b"world",)], dtype=[("note", "S5")])
    df = to_frame(data, Meta())

    assert isinstance(df, backend.dataframe_type)
    values = [None if pd.isna(value) else value for value in df["note"].to_numpy()]
    assert values == ["hello", None, "world"]


def test_loadarff_invalid_output_type():
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        with pytest.raises(ValueError, match=r"output_type must be 'pandas' or 'polars'"):
            loadarff(fp, output_type="numpy")


def test_writearff_polars_lazyframe_rejected(temp_file):
    """writearff rejects a polars LazyFrame with a TypeError."""
    df_pl = pl.DataFrame({"grade": pl.Series(["I", "II"], dtype=pl.Enum(["I", "II", "III"]))})

    with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
        writearff(df_pl.lazy(), temp_file, relation_name="lazy_round", index=False)


def test_writearff_pure_categorical_round_trip(dataframe_backend, temp_file):
    """A frame whose every column is categorical must round-trip.

    A previous implementation crashed in ``_write_data`` because
    ``data.iloc[i, :]`` of an all-categorical row Series cannot pass through
    the numpy ufunc in ``_check_str_array``; ``_write_data`` therefore
    coerces to object dtype before iloc.
    """
    data = {"a": ["x", "y", "x"], "b": ["p", "q", "p"]}
    categories = {"a": ["x", "y"], "b": ["p", "q", "r"]}
    df = dataframe_backend.make_frame(data, categories=categories)

    writearff(df, temp_file, relation_name="pure_cat", index=False)
    temp_file.close()

    df_round = loadarff(temp_file.name, output_type=dataframe_backend.name)
    dataframe_backend.assert_frame_equal(df_round, dataframe_backend.make_frame(data, categories=categories))


def test_writearff_polars_categorical_dtype(temp_file):
    """``pl.Categorical`` (no declared category list) serializes with sorted
    observed categories and round-trips as the equivalent ``pl.Enum``."""
    df_pl = pl.DataFrame({"a": pl.Series(["y", "x", "y"], dtype=pl.Categorical)})

    writearff(df_pl, temp_file, relation_name="observed_cat", index=False)
    temp_file.close()

    with open(temp_file.name) as fp:
        contents = fp.read()
    assert "{x,y}" in contents

    df_round = loadarff(temp_file.name, output_type="polars")
    assert df_round["a"].to_list() == ["y", "x", "y"]
    assert df_round["a"].dtype == pl.Enum(["x", "y"])


def test_writearff_polars_no_pyarrow_dependency(temp_file, monkeypatch):
    """``writearff(polars_frame_with_categorical)`` must not require ``pyarrow``.

    The ``nw_df.to_pandas()`` path dispatches Categorical / Enum
    columns through Arrow; this test shields ``pyarrow`` to ensure the
    column-wise conversion in ``_prepare_polars_for_arff_write`` does not import
    it.
    """
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    df = pl.DataFrame(
        {
            "x": pl.Series(["a", "b"], dtype=pl.Categorical),
            "y": pl.Series(["c", "d"], dtype=pl.Enum(["c", "d", "e"])),
            "z": [1.0, 2.0],
        }
    )
    writearff(df, temp_file, relation_name="t", index=False)
    temp_file.close()
    df_round = loadarff(temp_file.name, output_type="polars")
    assert df_round["y"].dtype == pl.Enum(["c", "d", "e"])


def test_writearff_unsupported_column_type(temp_file):
    data = pd.DataFrame(
        {
            "attr_datetime": np.array([2 + 3j, 45.1 - 1j, 0 - 1j, 7 + 0j, 132 - 3j, 1 - 0.41j], dtype="complex128"),
        }
    )

    with pytest.raises(TypeError, match="unsupported type complex128"):
        writearff(data, temp_file, relation_name="test_delta", index=False)
