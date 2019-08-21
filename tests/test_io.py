from collections import OrderedDict
from io import StringIO

import numpy
import pandas
import pandas.util.testing as tm
import pytest

from sksurv.io import loadarff, writearff


EXPECTED_1 = ["@relation test_nominal\n",
              "\n",
              "@attribute attr_nominal\t{beer,water,wine}\n",
              "@attribute attr_nominal_spaces\t{\"hard liquor\",mate,\"red wine\"}\n",
              "\n",
              "@data\n",
              "water,\"red wine\"\n",
              "wine,\"hard liquor\"\n",
              "beer,?\n",
              "?,mate\n",
              "wine,\"hard liquor\"\n",
              "water,mate\n"]


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
    "water,mate\n"]


EXPECTED_DATETIME = ["@relation test_datetime\n",
                     "\n",
                     "@attribute attr_datetime\tdate 'yyyy-MM-dd HH:mm:ss'\n",
                     "\n",
                     "@data\n",
                     '"2014-10-31 14:13:01"\n',
                     '"2004-03-13 19:49:31"\n',
                     '"1998-12-06 09:10:11"\n']


def test_loadarff_dataframe():
    contents = "".join(EXPECTED_NO_QUOTES)
    with StringIO(contents) as fp:
        actual_df = loadarff(fp)

    expected_df = pandas.DataFrame.from_dict(OrderedDict(
        [("attr_nominal",
          pandas.Series(pandas.Categorical.from_codes(
              [1, 2, 0, -1, 2, 1],
              ["beer", "water", "wine"]))),
         ("attr_nominal_spaces",
          pandas.Series(pandas.Categorical.from_codes(
              [2, 0, -1, 1, 0, 1],
              ['hard liquor', 'mate', 'red wine'])))
         ]
    ))

    tm.assert_frame_equal(expected_df, actual_df, check_exact=True)


def test_writearff_nominal(temp_file):
    data = pandas.DataFrame({'attr_nominal': ['water', 'wine', 'beer', None, 'wine', 'water'],
                             'attr_nominal_spaces': ['red wine', 'hard liquor', None, 'mate', 'hard liquor',
                                                     'mate']})

    writearff(data, temp_file, relation_name='test_nominal', index=False)

    with open(temp_file.name, 'r') as fp:
        read_date = fp.readlines()

    assert EXPECTED_1 == read_date


def test_writearff_nominal_with_quotes(temp_file):
    data = pandas.DataFrame({'attr_nominal': ['water', 'wine', 'beer', None, 'wine', 'water'],
                             'attr_nominal_spaces': ["'red wine'", "'hard liquor'", None, 'mate', "'hard liquor'",
                                                     'mate']})

    writearff(data, temp_file, relation_name='test_nominal', index=False)

    with open(temp_file.name, 'r') as fp:
        read_date = fp.readlines()

    assert EXPECTED_1 == read_date


def test_writearff_nominal_as_category(temp_file):
    data = pandas.DataFrame({'attr_nominal': pandas.Categorical(['water', 'wine', 'beer', None, 'wine', 'water'],
                                                                ordered=False),
                             'attr_nominal_spaces': pandas.Categorical(["'red wine'", "'hard liquor'", None,
                                                                        'mate', "'hard liquor'", 'mate'],
                                                                       ordered=False)})

    writearff(data, temp_file, relation_name='test_nominal', index=False)

    with open(temp_file.name, 'r') as fp:
        read_date = fp.readlines()

    expected = EXPECTED_1.copy()
    expected[3] = "@attribute attr_nominal_spaces\t{\"hard liquor\",\"red wine\",mate}\n"
    assert expected == read_date


def test_writearff_nominal_as_category_extra(temp_file):
    data = pandas.DataFrame({'attr_nominal': pandas.Categorical(['water', 'wine', 'beer', None, 'wine', 'water'],
                                                                categories=['beer', 'coke', 'water', 'wine'],
                                                                ordered=False),
                             'attr_nominal_spaces': pandas.Categorical(["'red wine'", "'hard liquor'", None,
                                                                        'mate', "'hard liquor'", 'mate'],
                                                                       ordered=False)})

    writearff(data, temp_file, relation_name='test_nominal', index=False)

    with open(temp_file.name, 'r') as fp:
        read_date = fp.readlines()

    expected = EXPECTED_1.copy()
    expected[2] = "@attribute attr_nominal\t{beer,coke,water,wine}\n"
    expected[3] = "@attribute attr_nominal_spaces\t{\"hard liquor\",\"red wine\",mate}\n"
    assert expected == read_date


def test_writearff_nominal_with_category_ordering(temp_file):
    data = pandas.DataFrame({'attr_nominal': pandas.Categorical(['water', 'wine', 'beer', None, 'wine', 'water'],
                                                                categories=['water', 'coke', 'beer', 'wine'],
                                                                ordered=False),
                             'attr_nominal_spaces': ["'red wine'", "'hard liquor'", None, 'mate', "'hard liquor'",
                                                     'mate']})

    writearff(data, temp_file, relation_name='test_nominal', index=False)

    with open(temp_file.name, 'r') as fp:
        read_date = fp.readlines()

    expected = EXPECTED_1.copy()
    expected[2] = "@attribute attr_nominal\t{water,coke,beer,wine}\n"
    assert expected == read_date


def test_writearff_datetime(temp_file):
    data = pandas.DataFrame(
        {"attr_datetime": numpy.array(
            ["2014-10-31 14:13:01", "2004-03-13 19:49:31", "1998-12-06 09:10:11"], dtype="datetime64")})

    writearff(data, temp_file, relation_name='test_datetime', index=False)

    with open(temp_file.name, 'r') as fp:
        read_date = fp.readlines()

    assert EXPECTED_DATETIME == read_date


def test_writearff_unsupported_column_type(temp_file):
    data = pandas.DataFrame(
        {"attr_datetime": numpy.array([2+3j, 45.1-1j, 0-1j, 7+0j, 132-3j, 1-0.41j], dtype="complex128")})

    with pytest.raises(TypeError, match="unsupported type complex128"):
        writearff(data, temp_file, relation_name='test_delta', index=False)
