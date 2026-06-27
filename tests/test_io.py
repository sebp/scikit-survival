from collections import OrderedDict
from io import StringIO
import sys

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import pandas.testing as tm
import polars as pl
import pytest

from sksurv.io import loadarff, writearff
from sksurv.testing import FixtureParameterFactory, get_pandas_infer_string_context

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


class DataFrameCases(FixtureParameterFactory):
    def data_nominal(self):
        data = pd.DataFrame(
            {
                "attr_nominal": ["water", "wine", "beer", None, "wine", "water"],
                "attr_nominal_spaces": ["red wine", "hard liquor", None, "mate", "hard liquor", "mate"],
            }
        )
        return data, "test_nominal", EXPECTED_1.copy()

    def data_nominal_with_quotes(self):
        data, rel_name, expected = self.data_nominal()
        data.loc[:, "attr_nominal_spaces"] = ["'red wine'", "'hard liquor'", None, "mate", "'hard liquor'", "mate"]
        return data, rel_name, expected

    def data_nominal_as_category(self):
        data, rel_name, expected = self.data_nominal_with_quotes()
        data = data.astype(dict.fromkeys(data.keys(), "category"))

        expected[3] = '@attribute attr_nominal_spaces\t{"hard liquor","red wine",mate}\n'
        return data, rel_name, expected

    def data_nominal_as_category_extra(self):
        data, rel_name, expected = self.data_nominal_as_category()
        data = data.astype(
            {
                "attr_nominal": CategoricalDtype(
                    categories=["beer", "coke", "water", "wine"],
                    ordered=False,
                )
            }
        )
        data.loc[:, "attr_nominal"] = ["water", "wine", "beer", None, "wine", "water"]

        expected[2] = "@attribute attr_nominal\t{beer,coke,water,wine}\n"
        return data, rel_name, expected

    def data_nominal_with_category_ordering(self):
        data, rel_name, expected = self.data_nominal_with_quotes()
        data = data.astype(
            {
                "attr_nominal": CategoricalDtype(
                    categories=["water", "coke", "beer", "wine"],
                    ordered=False,
                )
            }
        )
        data.loc[:, "attr_nominal"] = ["water", "wine", "beer", None, "wine", "water"]

        expected[2] = "@attribute attr_nominal\t{water,coke,beer,wine}\n"
        return data, rel_name, expected

    def data_datetime(self):
        data = pd.DataFrame(
            {
                "attr_datetime": np.array(
                    ["2014-10-31 14:13:01", "2004-03-13 19:49:31", "1998-12-06 09:10:11"], dtype="datetime64"
                )
            }
        )
        return data, "test_datetime", EXPECTED_DATETIME.copy()


@pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
def test_loadarff_dataframe(infer_string_context):
    with infer_string_context:
        contents = "".join(EXPECTED_NO_QUOTES)
        with StringIO(contents) as fp:
            actual_df = loadarff(fp)

        expected_df = pd.DataFrame.from_dict(
            OrderedDict(
                [
                    (
                        "attr_nominal",
                        pd.Series(pd.Categorical.from_codes([1, 2, 0, -1, 2, 1], ["beer", "water", "wine"])),
                    ),
                    (
                        "attr_nominal_spaces",
                        pd.Series(pd.Categorical.from_codes([2, 0, -1, 1, 0, 1], ["hard liquor", "mate", "red wine"])),
                    ),
                ]
            )
        )

        tm.assert_frame_equal(expected_df, actual_df, check_exact=True)


@pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
@pytest.mark.parametrize("make_data_fn", DataFrameCases().get_cases_func())
def test_writearff(make_data_fn, temp_file, infer_string_context):
    with infer_string_context:
        data_frame, relation_name, expectation = make_data_fn()
        writearff(data_frame, temp_file, relation_name=relation_name, index=False)

        with open(temp_file.name) as fp:
            read_date = fp.readlines()

        assert expectation == read_date


# --- loadarff(output_type="polars") tests ---
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


def test_loadarff_polars_basic_nominal():
    """polars output: nominal column becomes pl.Enum(declared_categories).

    Declared categories include the unseen "IV" label, which must survive
    in the dtype itself.
    """
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pl = loadarff(fp, output_type="polars")

    assert isinstance(df_pl, pl.DataFrame)
    assert df_pl["grade"].dtype == pl.Enum(["I", "II", "III", "IV"])
    assert df_pl["grade"].cat.get_categories().to_list() == ["I", "II", "III", "IV"]
    assert df_pl["grade"].to_list() == ["I", "II", "III", "I", "II", "III", "I", "II"]


def test_loadarff_polars_matches_pandas_kernel():
    """polars (pl.Enum) and pandas (pd.Categorical) outputs of loadarff must
    produce numerically identical clinical_kernel results.

    This pins sksurv's semantic policy that ``pl.Enum`` is classified as
    nominal, matching pandas ``pd.Categorical(ordered=False)``.
    """
    from sksurv.kernels import clinical_kernel

    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pd = loadarff(fp)
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pl = loadarff(fp, output_type="polars")

    K_pd = clinical_kernel(df_pd)
    K_pl = clinical_kernel(df_pl)
    np.testing.assert_allclose(K_pd, K_pl, atol=1e-12)


def test_loadarff_polars_matches_pandas_onehot():
    """OneHotEncoder fit on polars (pl.Enum) preserves the same declared
    categories as the pandas path, including unseen labels."""
    from sksurv.preprocessing import OneHotEncoder

    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pd = loadarff(fp)
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pl = loadarff(fp, output_type="polars")

    enc_pd = OneHotEncoder().fit(df_pd)
    enc_pl = OneHotEncoder().fit(df_pl)

    assert list(enc_pd.categories_["grade"]) == list(enc_pl.categories_["grade"])
    assert list(enc_pd.encoded_columns_) == list(enc_pl.encoded_columns_)
    # The unseen "IV" produces a column even though no training row has it.
    assert "grade=IV" in list(enc_pl.encoded_columns_)


def test_loadarff_polars_missing_value():
    """ARFF '?' missing-value tokens become null in the polars output."""
    arff = """\
@relation missing
@attribute grade {I, II, III}
@data
I
?
II
"""
    with StringIO(arff) as fp:
        df_pl = loadarff(fp, output_type="polars")
    assert df_pl["grade"].is_null().to_list() == [False, True, False]
    assert df_pl["grade"].dtype == pl.Enum(["I", "II", "III"])


def test_loadarff_polars_numeric_columns_preserved():
    """Numeric columns are emitted as plain polars numeric dtype (not Enum)."""
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pl = loadarff(fp, output_type="polars")
    assert df_pl["age"].dtype.is_numeric()
    assert df_pl["age"].to_list() == [45.0, 60.0, 72.0, 55.0, 68.0, 50.0, 40.0, 70.0]


def test_loadarff_polars_string_attribute_decodes_missing_values():
    from sksurv.io.arffread import _to_polars_dataframe

    # SciPy's public loadarff rejects string attributes, but the converter still
    # needs to handle the record-array shape SciPy uses for non-nominal fields.
    class Meta:
        @staticmethod
        def names():
            return ["note"]

        def __getitem__(self, key):
            assert key == "note"
            return "string", None

    data = np.array([(b"hello",), (b"?",), (b"world",)], dtype=[("note", "S5")])
    df_pl = _to_polars_dataframe(data, Meta())

    assert df_pl["note"].dtype == pl.String
    assert df_pl["note"].to_list() == ["hello", None, "world"]


def test_loadarff_invalid_output_type():
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        with pytest.raises(ValueError, match=r"output_type must be 'pandas' or 'polars'"):
            loadarff(fp, output_type="numpy")


def test_writearff_polars_round_trip(temp_file):
    """writearff(polars DataFrame) preserves the declared category list
    including unseen labels, then loadarff(output_type="polars") restores it
    bit-for-bit.
    """
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pl = loadarff(fp, output_type="polars")

    writearff(df_pl, temp_file, relation_name="round_trip", index=False)
    temp_file.close()

    with open(temp_file.name) as fp:
        contents = fp.read()
    # Declared categories (incl. unseen "IV") must appear in the header
    assert "{I,II,III,IV}" in contents

    df_pl_round = loadarff(temp_file.name, output_type="polars")
    assert df_pl_round["grade"].dtype == pl.Enum(["I", "II", "III", "IV"])
    assert df_pl_round["grade"].to_list() == df_pl["grade"].to_list()


def test_writearff_polars_lazyframe_rejected(temp_file):
    """writearff rejects a polars LazyFrame with a TypeError."""
    with StringIO(ARFF_WITH_UNSEEN_NOMINAL) as fp:
        df_pl = loadarff(fp, output_type="polars")

    with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported"):
        writearff(df_pl.lazy(), temp_file, relation_name="lazy_round", index=False)


def test_writearff_polars_pure_categorical(temp_file):
    """A polars frame whose every column is categorical/Enum must round-trip.

    A previous implementation crashed in ``_write_data`` because
    ``data.iloc[i, :]`` of an all-categorical row Series cannot pass through
    the numpy ufunc in ``_check_str_array``; ``_write_data`` therefore
    coerces to object dtype before iloc.
    """
    df_pl = pl.DataFrame(
        {
            "a": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "b": pl.Series(["p", "q", "p"], dtype=pl.Enum(["p", "q", "r"])),
        }
    )
    writearff(df_pl, temp_file, relation_name="pure_cat", index=False)
    temp_file.close()
    df_round = loadarff(temp_file.name, output_type="polars")
    assert df_round["a"].to_list() == ["x", "y", "x"]
    assert df_round["b"].to_list() == ["p", "q", "p"]
    assert df_round["b"].dtype == pl.Enum(["p", "q", "r"])


def test_writearff_pandas_pure_categorical_index_false(temp_file):
    """A pandas frame whose every column is categorical, written without an
    index column, must round-trip. This pins the latent pandas-side variant
    of the all-categorical ``_write_data`` crash described above.
    """
    df_pd = pd.DataFrame(
        {
            "a": pd.Categorical(["x", "y", "x"]),
            "b": pd.Categorical(["p", "q", "p"], categories=["p", "q", "r"]),
        }
    )
    writearff(df_pd, temp_file, relation_name="pure_cat", index=False)
    temp_file.close()
    df_round = loadarff(temp_file.name)
    assert list(df_round["a"]) == ["x", "y", "x"]


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
