from collections import OrderedDict
from io import StringIO

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import pandas.testing as tm
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


def test_writearff_unsupported_column_type(temp_file):
    data = pd.DataFrame(
        {
            "attr_datetime": np.array([2 + 3j, 45.1 - 1j, 0 - 1j, 7 + 0j, 132 - 3j, 1 - 0.41j], dtype="complex128"),
        }
    )

    with pytest.raises(TypeError, match="unsupported type complex128"):
        writearff(data, temp_file, relation_name="test_delta", index=False)
