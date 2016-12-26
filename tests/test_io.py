from io import TextIOWrapper, StringIO
import os
import tempfile

from numpy.testing import TestCase, run_module_suite
import numpy
import pandas
import pandas.util.testing as tm

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


EXPECTED_DATETIME = ["@relation test_datetime\n",
                     "\n",
                     "@attribute attr_datetime\tdate 'yyyy-MM-dd HH:mm:ss'\n",
                     "\n",
                     "@data\n",
                     '"2014-10-31 14:13:01"\n',
                     '"2004-03-13 19:49:31"\n',
                     '"1998-12-06 09:10:11"\n']


class ReadArffTest(TestCase):
    def test_dataframe(self):
        contents = "".join(EXPECTED_1)
        with StringIO(contents) as fp:
            actual_df = loadarff(fp)

        expected_df = pandas.DataFrame.from_items(
            [("attr_nominal",
              pandas.Series(["water", "wine", "beer", None, "wine", "water"]).astype("category")),
             ("attr_nominal_spaces",
              pandas.Series(['"red wine"', '"hard liquor"', None, "mate", '"hard liquor"', "mate"]).astype("category"))
             ]
        )

        tm.assert_frame_equal(expected_df, actual_df, check_exact=True)


class WriteArffTest(TestCase):
    def test_nominal(self):
        data = pandas.DataFrame({'attr_nominal': ['water', 'wine', 'beer', None, 'wine', 'water'],
                                 'attr_nominal_spaces': ['red wine', 'hard liquor', None, 'mate', 'hard liquor',
                                                         'mate']})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                writearff(data, tfp, relation_name='test_nominal', index=False)

            with open(fp.name, 'r') as fp:
                read_date = fp.readlines()
        finally:
            os.unlink(fp.name)

        self.assertListEqual(EXPECTED_1, read_date)

    def test_nominal_with_quotes(self):
        data = pandas.DataFrame({'attr_nominal': ['water', 'wine', 'beer', None, 'wine', 'water'],
                                 'attr_nominal_spaces': ["'red wine'", "'hard liquor'", None, 'mate', "'hard liquor'",
                                                         'mate']})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                writearff(data, tfp, relation_name='test_nominal', index=False)

            with open(fp.name, 'r') as fp:
                read_date = fp.readlines()
        finally:
            os.unlink(fp.name)

        self.assertListEqual(EXPECTED_1, read_date)

    def test_nominal_as_category(self):
        data = pandas.DataFrame({'attr_nominal': pandas.Categorical(['water', 'wine', 'beer', None, 'wine', 'water'],
                                                                    ordered=False),
                                 'attr_nominal_spaces': pandas.Categorical(["'red wine'", "'hard liquor'", None,
                                                                            'mate', "'hard liquor'", 'mate'],
                                                                           ordered=False)})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                writearff(data, tfp, relation_name='test_nominal', index=False)

            with open(fp.name, 'r') as fp:
                read_date = fp.readlines()
        finally:
            os.unlink(fp.name)

        expected = EXPECTED_1.copy()
        expected[3] = "@attribute attr_nominal_spaces\t{\"hard liquor\",\"red wine\",mate}\n"
        self.assertListEqual(expected, read_date)

    def test_nominal_as_category_extra(self):
        data = pandas.DataFrame({'attr_nominal': pandas.Categorical(['water', 'wine', 'beer', None, 'wine', 'water'],
                                                                    categories=['beer', 'coke', 'water', 'wine'],
                                                                    ordered=False),
                                 'attr_nominal_spaces': pandas.Categorical(["'red wine'", "'hard liquor'", None,
                                                                            'mate', "'hard liquor'", 'mate'],
                                                                           ordered=False)})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                writearff(data, tfp, relation_name='test_nominal', index=False)

            with open(fp.name, 'r') as fp:
                read_date = fp.readlines()
        finally:
            os.unlink(fp.name)

        expected = EXPECTED_1.copy()
        expected[2] = "@attribute attr_nominal\t{beer,coke,water,wine}\n"
        expected[3] = "@attribute attr_nominal_spaces\t{\"hard liquor\",\"red wine\",mate}\n"
        self.assertListEqual(expected, read_date)

    def test_nominal_with_category_ordering(self):
        data = pandas.DataFrame({'attr_nominal': pandas.Categorical(['water', 'wine', 'beer', None, 'wine', 'water'],
                                                                    categories=['water', 'coke', 'beer', 'wine'],
                                                                    ordered=False),
                                 'attr_nominal_spaces': ["'red wine'", "'hard liquor'", None, 'mate', "'hard liquor'",
                                                         'mate']})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                writearff(data, tfp, relation_name='test_nominal', index=False)

            with open(fp.name, 'r') as fp:
                read_date = fp.readlines()
        finally:
            os.unlink(fp.name)

        expected = EXPECTED_1.copy()
        expected[2] = "@attribute attr_nominal\t{water,coke,beer,wine}\n"
        self.assertListEqual(expected, read_date)

    def test_datetime(self):
        data = pandas.DataFrame(
            {"attr_datetime": numpy.array(
                ["2014-10-31 14:13:01", "2004-03-13 19:49:31", "1998-12-06 09:10:11"], dtype="datetime64")})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                writearff(data, tfp, relation_name='test_datetime', index=False)

            with open(fp.name, 'r') as fp:
                read_date = fp.readlines()
        finally:
            os.unlink(fp.name)

        self.assertListEqual(EXPECTED_DATETIME, read_date)

    def test_unsupported_column_type(self):
        data = pandas.DataFrame(
            {"attr_datetime": numpy.array([2+3j, 45.1-1j, 0-1j, 7+0j, 132-3j, 1-0.41j], dtype="complex128")})

        fp = tempfile.NamedTemporaryFile(delete=False)
        try:
            with TextIOWrapper(fp) as tfp:
                self.assertRaisesRegex(TypeError, "unsupported type complex128",
                                       writearff, data, tfp, relation_name='test_delta', index=False)
        finally:
            os.unlink(fp.name)


if __name__ == '__main__':
    run_module_suite()