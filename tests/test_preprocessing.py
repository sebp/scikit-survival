import pandas as pd
import numpy as np
from numpy.testing import run_module_suite, TestCase
import pandas.util.testing as tm

from sksurv.preprocessing import OneHotEncoder


def create_data(n_samples=117):
    rnd = np.random.RandomState(51365192)
    data_num = pd.DataFrame(rnd.rand(n_samples, 5),
                            columns=["N%d" % i for i in range(5)])

    dat_cat = pd.DataFrame(dict(
        binary_1=pd.Categorical.from_codes(
            rnd.binomial(1, 0.6, n_samples),
            ["Yes", "No"]),
        binary_2=pd.Categorical.from_codes(
            rnd.binomial(1, 0.376, n_samples),
            ["East", "West"]),
        trinary=pd.Categorical.from_codes(
            rnd.binomial(2, 0.76, n_samples),
            ["Green", "Blue", "Red"]),
        many=pd.Categorical.from_codes(
            rnd.binomial(5, 0.47, n_samples),
            ["One", "Two", "Three", "Four", "Five", "Six"])
    ))
    data = pd.concat((data_num, dat_cat), axis=1)
    return data


def encoded_data(data):
    expected = []
    for nam, col in data.iteritems():
        if hasattr(col, "cat"):
            for cat in col.cat.categories[1:]:
                name = '{}={}'.format(nam, cat)
                s = pd.Series(col == cat, dtype=np.float64)
                expected.append((name, s))
        else:
            expected.append((nam, col))

    expected_data = pd.DataFrame.from_items(expected)
    return expected_data


class TestOneHotEncoder(TestCase):
    def test_fit(self):
        data = create_data()
        expected_data = encoded_data(data)

        t = OneHotEncoder().fit(data)

        self.assertListEqual(t.feature_names_.tolist(),
                             ['binary_1', 'binary_2', 'many', 'trinary'])
        self.assertSetEqual(set(t.encoded_columns_),
                            set(expected_data.columns))

        self.assertDictEqual(t.categories_,
                             {k: data[k].cat.categories
                              for k in ['binary_1', 'binary_2', 'many', 'trinary']})

    def test_fit_transform(self):
        data = create_data()
        expected_data = encoded_data(data)

        actual_data = OneHotEncoder().fit_transform(data)
        tm.assert_frame_equal(actual_data, expected_data)

    def test_transform(self):
        data = create_data()

        t = OneHotEncoder().fit(data)
        data = create_data(165)
        expected_data = encoded_data(data)
        actual_data = t.transform(data)
        tm.assert_frame_equal(actual_data, expected_data)

        data = pd.concat((data.iloc[:, :2], data.iloc[:, 5:], data.iloc[:, 2:5]), axis=1)
        actual_data = t.transform(data)
        tm.assert_frame_equal(actual_data, expected_data)

    def test_transform_other_columns(self):
        data = create_data()

        t = OneHotEncoder().fit(data)
        data = create_data(125)

        data_renamed = data.rename(columns={"binary_1": "renamed_1"})
        self.assertRaisesRegex(ValueError,
                               "1 features are missing from data: \['binary_1'\]",
                               t.transform, data_renamed)

        data_dropped = data.drop('trinary', axis=1)
        self.assertRaisesRegex(ValueError,
                               "1 features are missing from data: \['trinary'\]",
                               t.transform, data_dropped)

        data_renamed = data.rename(columns={"binary_1": "renamed_1", "many": "too_many"})
        self.assertRaisesRegex(ValueError,
                               "2 features are missing from data: \['binary_1', 'many'\]",
                               t.transform, data_renamed)


if __name__ == '__main__':
    run_module_suite()
