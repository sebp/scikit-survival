from collections import OrderedDict

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pandas.testing as tm
import pytest

from sksurv.preprocessing import OneHotEncoder


def _encoded_data(data):
    expected = []
    for nam, col in data.iteritems():
        if hasattr(col, "cat"):
            for cat in col.cat.categories[1:]:
                name = '{}={}'.format(nam, cat)
                s = pd.Series(col == cat, dtype=np.float64)
                expected.append((name, s))
        else:
            expected.append((nam, col))

    expected_data = pd.DataFrame.from_dict(OrderedDict(expected))
    return expected_data


@pytest.fixture()
def create_data():
    def _create_data(n_samples=117):
        rnd = np.random.RandomState(51365192)
        data_num = pd.DataFrame(rnd.rand(n_samples, 5),
                                columns=["N%d" % i for i in range(5)])

        dat_cat = pd.DataFrame(OrderedDict([
            ("binary_1", pd.Categorical.from_codes(
                rnd.binomial(1, 0.6, n_samples),
                ["Yes", "No"])),
            ("binary_2", pd.Categorical.from_codes(
                rnd.binomial(1, 0.376, n_samples),
                ["East", "West"])),
            ("trinary", pd.Categorical.from_codes(
                rnd.binomial(2, 0.76, n_samples),
                ["Green", "Blue", "Red"])),
            ("many", pd.Categorical.from_codes(
                rnd.binomial(5, 0.47, n_samples),
                ["One", "Two", "Three", "Four", "Five", "Six"]))
        ]))
        data = pd.concat((data_num, dat_cat), axis=1)
        return data, _encoded_data(data)

    return _create_data


class TestOneHotEncoder:
    @staticmethod
    def test_fit(create_data):
        data, expected_data = create_data()

        t = OneHotEncoder().fit(data)

        assert t.feature_names_.tolist() == ['binary_1', 'binary_2', 'trinary', 'many']
        assert set(t.encoded_columns_) == set(expected_data.columns)

        assert t.categories_ == {k: data[k].cat.categories
                                 for k in ['binary_1', 'binary_2', 'trinary', 'many']}

    @staticmethod
    def test_fit_transform(create_data):
        data, expected_data = create_data()

        actual_data = OneHotEncoder().fit_transform(data)
        tm.assert_frame_equal(actual_data, expected_data)

    @staticmethod
    def test_transform(create_data):
        data, _ = create_data()

        t = OneHotEncoder().fit(data)
        data, expected_data = create_data(165)
        actual_data = t.transform(data)
        tm.assert_frame_equal(actual_data, expected_data)

        data = pd.concat((data.iloc[:, :2], data.iloc[:, 5:], data.iloc[:, 2:5]), axis=1)
        actual_data = t.transform(data)
        tm.assert_frame_equal(actual_data, expected_data)

    @staticmethod
    def test_get_feature_names_out(create_data):
        data, expected_data = create_data()

        t = OneHotEncoder()
        t.fit(data)

        out_names = t.get_feature_names_out()
        assert_array_equal(out_names, expected_data.columns.values)

    @staticmethod
    def test_get_feature_names_out_shuffled(create_data):
        data, _ = create_data()
        order = np.array(['binary_1', 'N0', 'N3', 'trinary', 'binary_2', 'N1', 'N2', 'many'])
        expected_columns = np.array([
            'binary_1=No',
            'N0',
            'N3',
            'trinary=Blue',
            'trinary=Red',
            'binary_2=West',
            'N1',
            'N2',
            'many=Two',
            'many=Three',
            'many=Four',
            'many=Five',
            'many=Six',
        ])

        t = OneHotEncoder()
        t.fit(data.loc[:, order])

        out_names = t.get_feature_names_out()
        assert_array_equal(out_names, expected_columns)

        with pytest.raises(ValueError, match="input_features is not equal to feature_names_in_"):
            t.get_feature_names_out(data.columns.tolist())

    @staticmethod
    def test_transform_other_columns(create_data):
        data, _ = create_data()

        t = OneHotEncoder().fit(data)
        data, _ = create_data(125)

        data_renamed = data.rename(columns={"binary_1": "renamed_1"})
        with pytest.raises(ValueError, match=r"1 features are missing from data: \['binary_1'\]"):
            t.transform(data_renamed)

        data_dropped = data.drop('trinary', axis=1)
        error_msg = "X has 8 features, but OneHotEncoder is expecting 9 features as input"
        with pytest.raises(ValueError, match=error_msg):
            t.transform(data_dropped)

        data_renamed = data.rename(columns={"binary_1": "renamed_1", "many": "too_many"})
        with pytest.raises(ValueError, match=r"2 features are missing from data: \['binary_1', 'many'\]"):
            t.transform(data_renamed)
