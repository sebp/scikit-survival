from numpy.testing import TestCase, run_module_suite, assert_array_almost_equal

import numpy
import pandas
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


class TestClinicalKernel(TestCase):
    def setUp(self):
        data = pandas.DataFrame({'age': [20, 23, 26, 54, 100],
                                 'lymph node size': [2, 1, 3, 4, 1],
                                 'lymph node spread': ['distant', 'none', 'distant', 'close', 'none'],
                                 'metastasis': ['yes', 'no', 'yes', 'yes', 'no']})
        data['lymph node size'] = pandas.Categorical(data['lymph node size'],
                                                     categories=[1, 2, 3, 4],
                                                     ordered=True)
        data['lymph node spread'] = pandas.Categorical(data['lymph node spread'],
                                                       categories=['none', 'close', 'distant'],
                                                       ordered=True)
        data['metastasis'] = pandas.Categorical(data['metastasis'],
                                                categories=['no', 'yes'],
                                                ordered=False)
        self.data = data

    def test_clinical_kernel_1(self):
        mat = clinical_kernel(self.data)
        expected = _get_expected_matrix()

        assert_array_almost_equal(expected, mat, 4)

    def test_clinical_kernel_no_ordinal(self):
        mat = clinical_kernel(self.data.drop(['lymph node size', 'lymph node spread'], axis=1))
        expected = _get_expected_matrix(with_ordinal=False)
        assert_array_almost_equal(expected, mat, 4)

    def test_clinical_kernel_no_nominal(self):
        mat = clinical_kernel(self.data.drop('metastasis', axis=1))
        expected = _get_expected_matrix(with_nominal=False)
        assert_array_almost_equal(expected, mat, 4)

    def test_clinical_kernel_no_continuous(self):
        mat = clinical_kernel(self.data.drop('age', axis=1))
        expected = _get_expected_matrix(with_continuous=False)
        assert_array_almost_equal(expected, mat, 4)

    def test_clinical_kernel_only_nominal(self):
        mat = clinical_kernel(self.data.drop(['age', 'lymph node size', 'lymph node spread'], axis=1))
        expected = _get_expected_matrix(with_continuous=False, with_ordinal=False)
        assert_array_almost_equal(expected, mat, 4)

    def test_clinical_kernel_x_and_y(self):
        mat = clinical_kernel(self.data.iloc[:3, :], self.data.iloc[3:, :])
        m = _get_expected_matrix()
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    def test_fit_error_ndim(self):
        t = ClinicalKernelTransform()

        self.assertRaisesRegex(ValueError, "expected 2d array, but got 1",
                               t.fit, numpy.random.randn(31))

        self.assertRaisesRegex(ValueError, "expected 2d array, but got 3",
                               t.fit, numpy.random.randn(31, 20, 2))

    def test_kernel_transform(self):
        t = ClinicalKernelTransform()

        t.fit(self.data)
        mat = t.transform(t.X_fit_)

        expected = _get_expected_matrix()

        assert_array_almost_equal(expected, mat, 4)

    def test_kernel_transform_x_and_y(self):
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(self.data)
        x_num = t.X_fit_.copy()

        t.fit(x_num[:3, :])
        mat = t.transform(x_num[3:, :])

        m = _get_expected_matrix()
        expected = m[:3, 3:].T

        assert_array_almost_equal(expected, mat, 4)

    def test_kernel_transform_feature_mismatch(self):
        t = ClinicalKernelTransform()
        t.fit(self.data)

        self.assertRaisesRegex(ValueError, 'expected array with 4 features, but got 17',
                               t.transform, numpy.zeros((2, 17), dtype=float))

    def test_pairwise(self):
        t = ClinicalKernelTransform()
        t.fit(self.data)

        mat = pairwise_kernels(t.X_fit_, t.X_fit_,
                               metric=t.pairwise_kernel, n_jobs=1)

        expected = _get_expected_matrix()

        assert_array_almost_equal(expected, mat, 4)

    def test_pairwise_x_and_y(self):
        t = ClinicalKernelTransform()
        t.fit(self.data)

        mat = pairwise_kernels(t.X_fit_[:3, :], t.X_fit_[3:, :],
                               metric=t.pairwise_kernel, n_jobs=1)

        m = _get_expected_matrix()
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    def test_pairwise_x_and_y_error_shape(self):
        t = ClinicalKernelTransform()
        t.fit(self.data)

        self.assertRaisesRegex(ValueError, "X and Y have different number of features",
                               t.pairwise_kernel, self.data.iloc[0, :], self.data.iloc[1, :2])

    def test_pairwise_no_nominal(self):
        t = ClinicalKernelTransform()
        t.fit(self.data.drop('metastasis', axis=1))

        mat = pairwise_kernels(t.X_fit_[:3, :], t.X_fit_[3:, :],
                               metric=t.pairwise_kernel, n_jobs=1)

        expected = _get_expected_matrix(with_nominal=False)
        assert_array_almost_equal(expected[:3:, 3:], mat, 4)

    def test_call_function(self):
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(self.data)

        mat = t(t.X_fit_, t.X_fit_)
        expected = _get_expected_matrix()
        assert_array_almost_equal(expected, mat, 4)

    def test_call_function_x_and_y(self):
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(self.data)

        mat = t(t.X_fit_[:3, :], t.X_fit_[3:, :])
        m = _get_expected_matrix()
        expected = m[:3:, 3:]

        assert_array_almost_equal(expected, mat, 4)

    def test_pairwise_feature_mismatch(self):
        t = ClinicalKernelTransform()
        t.fit(self.data)

        self.assertRaisesRegex(ValueError, 'Incompatible dimension for X and Y matrices: '
                                           'X.shape\[1\] == 4 while Y.shape\[1\] == 17',
                               pairwise_kernels, t.X_fit_, numpy.zeros((2, 17), dtype=float),
                               metric=t.pairwise_kernel, n_jobs=1)

    def test_prepare(self):
        t = ClinicalKernelTransform(fit_once=True)
        t.prepare(self.data)

        copy = clone(t).fit(t.X_fit_)
        mat = copy.transform(t.X_fit_[:4, :])

        expected = _get_expected_matrix()
        assert_array_almost_equal(expected[:4, :], mat, 4)

    def test_prepare_error_fit_once(self):
        t = ClinicalKernelTransform(fit_once=False)

        self.assertRaisesRegex(ValueError, "prepare can only be used if fit_once parameter is set to True",
                               t.prepare, self.data)

    def test_prepare_error_type(self):
        t = ClinicalKernelTransform(fit_once=True)

        self.assertRaisesRegex(TypeError, 'X must be a pandas DataFrame',
                               t.prepare, [[0, 1], [1, 2], [4, 3], [6, 5]])

    def test_prepare_error_dtype(self):
        t = ClinicalKernelTransform(fit_once=True)
        data = pandas.DataFrame({"age": [12, 61, 18, 21, 57, 17],
                                 "date": numpy.array(
                                     ["2016-01-01", "1954-06-30", "1999-03-01", "2005-02-25", "2112-12-31",
                                      "1431-09-16"], dtype='datetime64')})

        self.assertRaisesRegex(TypeError, 'unsupported dtype: dtype\(.+\)',
                               t.prepare, data)

    def test_feature_mismatch(self):
        x = self.data.iloc[:, :2]
        y = self.data.iloc[:, 2:]
        self.assertRaisesRegex(ValueError, 'columns do not match',
                               clinical_kernel, x, y)

        y = numpy.zeros((10, 17))
        self.assertRaisesRegex(ValueError, 'x and y have different number of features',
                               clinical_kernel, x, y)


if __name__ == '__main__':
    run_module_suite()
