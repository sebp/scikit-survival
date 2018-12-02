from numpy.testing import assert_array_almost_equal

import numpy
import pandas
import pytest
from sklearn.base import clone
from sklearn.metrics.pairwise import pairwise_kernels

from sksurv.kernels import clinical_kernel, ClinicalKernelTransform


def _get_expected_matrix(with_ordinal=True, with_nominal=True, with_continuous=True):
    mat_age = numpy.array([[1., 0.9625, 0.925, 0.575, 0.],
                           [0.9625, 1., 0.9625, 0.6125, 0.0375],
                           [0.925, 0.9625, 1., 0.6500, 0.075],
                           [0.575, 0.6125, 0.6500, 1., 0.425],
                           [0., 0.0375, 0.075, 0.425, 1.]])

    mat_node_size = numpy.array([[1., 2/3, 2/3, 1/3, 2/3],
                                [2/3, 1., 1/3, 0., 1.],
                                [2/3, 1/3, 1., 2/3, 1/3],
                                [1/3, 0., 2/3, 1., 0.],
                                [2/3, 1., 1/3, 0., 1.]])

    mat_node_spread = numpy.array([[1., 0., 1., 0.5, 0.],
                                   [0., 1., 0., 0.5, 1.],
                                   [1., 0., 1., 0.5, 0.],
                                   [0.5, 0.5, 0.5, 1., 0.5],
                                   [0., 1., 0., 0.5, 1.]])

    mat_metastasis = numpy.array([[1, 0, 1, 1, 0],
                                  [0, 1, 0, 0, 1],
                                  [1, 0, 1, 1, 0],
                                  [1, 0, 1, 1, 0],
                                  [0, 1, 0, 0, 1]], dtype=float)

    included = []
    if with_continuous:
        included.append(mat_age)
    if with_ordinal:
        included.append(mat_node_size)
        included.append(mat_node_spread)
    if with_nominal:
        included.append(mat_metastasis)

    expected = included[0]
    for i in range(1, len(included)):
        expected += included[i]
    expected /= len(included)

    return expected


@pytest.fixture
def make_data():
    data = {'age': [20, 23, 26, 54, 100],
            'lymph node size': [2, 1, 3, 4, 1],
            'lymph node spread': ['distant', 'none', 'distant', 'close', 'none'],
            'metastasis': ['yes', 'no', 'yes', 'yes', 'no']}

    def _make_data(with_ordinal=True, with_nominal=True, with_continuous=True):
        data_s = {}
        if with_continuous:
            data_s['age'] = data['age']

        if with_ordinal:
            data_s['lymph node size'] = pandas.Categorical(data['lymph node size'],
                                                           categories=[1, 2, 3, 4],
                                                           ordered=True)
            data_s['lymph node spread'] = pandas.Categorical(data['lymph node spread'],
                                                             categories=['none', 'close', 'distant'],
                                                             ordered=True)
        if with_nominal:
            data_s['metastasis'] = pandas.Categorical(data['metastasis'],
                                                      categories=['no', 'yes'],
                                                      ordered=False)
        expected = _get_expected_matrix(
            with_ordinal=with_ordinal,
            with_nominal=with_nominal,
            with_continuous=with_continuous)

        return pandas.DataFrame(data_s), expected

    return _make_data


class TestClinicalKernel(object):

    @staticmethod
    def test_clinical_kernel_1(make_data):
        data, expected = make_data()
        mat = clinical_kernel(data)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_ordinal(make_data):
        data, expected = make_data(with_ordinal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_nominal(make_data):
        data, expected = make_data(with_nominal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_no_continuous(make_data):
        data, expected = make_data(with_continuous=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_only_nominal(make_data):
        data, expected = make_data(with_continuous=False, with_ordinal=False)
        mat = clinical_kernel(data)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_clinical_kernel_x_and_y(make_data):
        data, m = make_data()
        mat = clinical_kernel(data.iloc[:3, :], data.iloc[3:, :])
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_fit_error_ndim():
        t = ClinicalKernelTransform()

        with pytest.raises(ValueError, match="expected 2d array, but got 1"):
            t.fit(numpy.random.randn(31))

        with pytest.raises(ValueError, match="expected 2d array, but got 3"):
            t.fit(numpy.random.randn(31, 20, 2))

    @staticmethod
    def test_kernel_transform(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform()

        t.fit(data)
        mat = t.transform(t.X_fit_)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_kernel_transform_x_and_y(make_data):
        data, m = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)
        x_num = t.X_fit_.copy()

        t.fit(x_num[:3, :])
        mat = t.transform(x_num[3:, :])

        expected = m[:3, 3:].T

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_kernel_transform_feature_mismatch(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        with pytest.raises(ValueError, match='expected array with 4 features, but got 17'):
            t.transform(numpy.zeros((2, 17), dtype=float))

    @staticmethod
    def test_pairwise(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        mat = pairwise_kernels(t.X_fit_, t.X_fit_,
                               metric=t.pairwise_kernel, n_jobs=1)

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_pairwise_x_and_y(make_data):
        data, m = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        mat = pairwise_kernels(t.X_fit_[:3, :], t.X_fit_[3:, :],
                               metric=t.pairwise_kernel, n_jobs=1)

        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_pairwise_x_and_y_error_shape(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        with pytest.raises(ValueError, match="X and Y have different number of features"):
            t.pairwise_kernel(data.iloc[0, :], data.iloc[1, :2])

    @staticmethod
    def test_pairwise_no_nominal(make_data):
        data, expected = make_data(with_nominal=False)
        t = ClinicalKernelTransform()
        t.fit(data)

        mat = pairwise_kernels(t.X_fit_[:3, :], t.X_fit_[3:, :],
                               metric=t.pairwise_kernel, n_jobs=1)

        assert_array_almost_equal(expected[:3:, 3:], mat, 4)

    @staticmethod
    def test_call_function(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        mat = t(t.X_fit_, t.X_fit_)
        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_call_function_x_and_y(make_data):
        data, m = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        mat = t(t.X_fit_[:3, :], t.X_fit_[3:, :])
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    @staticmethod
    def test_pairwise_feature_mismatch(make_data):
        data, _ = make_data()
        t = ClinicalKernelTransform()
        t.fit(data)

        with pytest.raises(ValueError, match=r'Incompatible dimension for X and Y matrices: '
                                             r'X.shape\[1\] == 4 while Y.shape\[1\] == 17'):
            pairwise_kernels(t.X_fit_, numpy.zeros((2, 17), dtype=float),
                             metric=t.pairwise_kernel, n_jobs=1)

    @staticmethod
    def test_prepare(make_data):
        data, expected = make_data()
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(data)

        copy = clone(t).fit(t.X_fit_)
        mat = copy.transform(t.X_fit_[:4, :])

        assert_array_almost_equal(expected[:4, :], mat, 4)

    @staticmethod
    def test_prepare_error_fit_once(make_data):
        data = make_data()
        t = ClinicalKernelTransform(fit_once=False)

        with pytest.raises(ValueError, match="prepare can only be used if fit_once parameter is set to True"):
            t.prepare(data)

    @staticmethod
    def test_prepare_error_type():
        t = ClinicalKernelTransform(fit_once=True)

        with pytest.raises(TypeError, match='X must be a pandas DataFrame'):
            t.prepare([[0, 1], [1, 2], [4, 3], [6, 5]])

    @staticmethod
    def test_prepare_error_dtype():
        t = ClinicalKernelTransform(fit_once=True)
        data = pandas.DataFrame({"age": [12, 61, 18, 21, 57, 17],
                                 "date": numpy.array(
                                     ["2016-01-01", "1954-06-30", "1999-03-01", "2005-02-25", "2112-12-31",
                                      "1731-09-16"], dtype='datetime64')})

        with pytest.raises(TypeError, match=r'unsupported dtype: dtype\(.+\)'):
            t.prepare(data)

    @staticmethod
    def test_feature_mismatch(make_data):
        data, _ = make_data()
        x = data.iloc[:, :2]
        y = data.iloc[:, 2:]
        with pytest.raises(ValueError, match='columns do not match'):
            clinical_kernel(x, y)

        y = numpy.zeros((10, 17))
        with pytest.raises(ValueError, match='x and y have different number of features'):
            clinical_kernel(x, y)
