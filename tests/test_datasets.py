import os
import tempfile
import warnings
from io import StringIO

import numpy
import pandas
import pandas.util.testing as tm
from numpy.testing import TestCase, assert_array_equal, assert_array_almost_equal, run_module_suite

from sksurv.datasets import *
from sksurv.io import writearff

ARFF_CATEGORICAL_INDEX_1 = """@relation arff_categorical_index
@attribute index {SampleOne,SampleTwo,SampleThree,SampleFour}
@attribute value real
@attribute label {no,yes}
@attribute size {small,medium,large}
@data
SampleOne,15.1,yes,medium
SampleTwo,13.8,no,large
SampleThree,-0.2,yes,small
SampleFour,2.453,yes,large
"""

ARFF_CATEGORICAL_INDEX_2 = """@relation arff_categorical_index
@attribute index {ASampleOne,ASampleTwo,ASampleThree,ASampleFour,ASampleFive}
@attribute value real
@attribute label {yes,no}
@attribute size {small,medium,large}
@data
ASampleOne,1.51,no,small
ASampleTwo,1.38,no,small
ASampleThree,-20,yes,large
ASampleFour,245.3,yes,small
ASampleFive,3.14,no,large
"""


def _make_features(n_samples, n_features, seed):
    rnd = numpy.random.RandomState(seed)
    return rnd.randn(n_samples, n_features)


def _make_survival_data(n_samples, n_features, seed):
    rnd = numpy.random.RandomState(seed)

    x = _make_features(n_samples, n_features, seed)
    event = rnd.binomial(1, 0.2, n_samples)
    time = rnd.exponential(25, size=n_samples)
    return x, event, time


def _make_classification_data(n_samples, n_features, n_classes, seed):
    rnd = numpy.random.RandomState(seed)

    x = _make_features(n_samples, n_features, seed)
    y = rnd.binomial(n_classes - 1, 0.2, 100)
    return x, y


class TestGetXy(TestCase):
    def test_get_x_y_survival(self):
        x, event, time = _make_survival_data(100, 10, 0)
        columns = ["V{}".format(i) for i in range(10)] + ["event", "time"]
        dataset = pandas.DataFrame(numpy.column_stack((x, event, time)), columns=columns)

        attr_labels = ["event", "time"]

        x_test, y_test = get_x_y(dataset, attr_labels, pos_label=1, survival=True)

        self.assertTupleEqual(y_test.dtype.names, ("event", "time"))

        assert_array_equal(y_test["event"].astype(numpy.uint32),
                           event.astype(numpy.uint32))
        assert_array_almost_equal(y_test["time"], time)

        assert_array_equal(x, x_test)

    def test_get_x_y_survival_no_label(self):
        x = _make_features(100, 10, 0)

        columns = ["V{}".format(i) for i in range(10)]
        dataset = pandas.DataFrame(x, columns=columns)

        attr_labels = [None, None]

        x_test, y_test = get_x_y(dataset, attr_labels, pos_label=1, survival=True)

        self.assertEqual(y_test, None)
        assert_array_equal(x, x_test)

    def test_get_x_y_survival_too_many_labels(self):
        x, event, time = _make_survival_data(100, 10, 0)

        columns = ["V{}".format(i) for i in range(10)] + ["event", "time"]
        dataset = pandas.DataFrame(numpy.column_stack((x, event, time)), columns=columns)

        attr_labels = ["event", "time", "random"]
        self.assertRaisesRegex(ValueError, "expected sequence of length two for attr_labels, but got 3",
                               get_x_y, dataset, attr_labels, pos_label=1, survival=True)

    def test_get_x_y_survival_too_little_labels(self):
        x, event, time = _make_survival_data(100, 10, 0)

        columns = ["V{}".format(i) for i in range(10)] + ["event", "time"]
        dataset = pandas.DataFrame(numpy.column_stack((x, event, time)), columns=columns)

        self.assertRaisesRegex(ValueError, "expected sequence of length two for attr_labels, but got 1",
                               get_x_y, dataset, ["event"], pos_label=1, survival=True)

        self.assertRaisesRegex(ValueError, "expected sequence of length two for attr_labels, but got 0",
                               get_x_y, dataset, [], pos_label=1, survival=True)

    def test_get_x_y_survival_no_pos_label(self):
        x, event, time = _make_survival_data(100, 10, 0)

        columns = ["V{}".format(i) for i in range(10)] + ["event", "time"]
        dataset = pandas.DataFrame(numpy.column_stack((x, event, time)), columns=columns)

        self.assertRaisesRegex(ValueError, "pos_label needs to be specified if survival=True",
                               get_x_y, dataset, ["event", "time"], survival=True)

    def test_get_x_y_classification(self):
        x, label = _make_classification_data(100, 10, 6, 0)

        columns = ["V{}".format(i) for i in range(10)] + ["class_label"]
        dataset = pandas.DataFrame(numpy.column_stack((x, label)), columns=columns)

        attr_labels = ["class_label"]

        x_test, y_test = get_x_y(dataset, attr_labels, survival=False)

        self.assertEqual(y_test.ndim, 2)
        assert_array_equal(y_test.values.ravel(), label)
        assert_array_equal(x_test, x)

    def test_get_x_y_classification_no_label(self):
        x = _make_features(100, 10, 0)

        columns = ["V{}".format(i) for i in range(10)]
        dataset = pandas.DataFrame(x, columns=columns)

        x_test, y_test = get_x_y(dataset, None, survival=False)

        self.assertEqual(y_test, None)
        assert_array_equal(x_test, x)


class TestLoadDatasets(TestCase):

    def assert_structured_array_dtype(self, arr, event, time, num_events):
        self.assertTupleEqual(arr.dtype.names, (event, time))
        self.assertTrue(numpy.issubdtype(arr.dtype.fields[event][0], numpy.bool_))
        self.assertTrue(numpy.issubdtype(arr.dtype.fields[time][0], numpy.float_))
        self.assertEqual(arr[event].sum(), num_events)

    def test_load_whas500(self):
        x, y = load_whas500()
        self.assertTupleEqual(x.shape, (500, 14))
        self.assertTupleEqual(y.shape, (500,))
        self.assert_structured_array_dtype(y, 'fstat', 'lenfol', 215)

    def test_load_gbsg2(self):
        x, y = load_gbsg2()
        self.assertTupleEqual(x.shape, (686, 8))
        self.assertTupleEqual(y.shape, (686,))
        self.assert_structured_array_dtype(y, 'cens', 'time', 299)

    def test_load_veterans_lung_cancer(self):
        x, y = load_veterans_lung_cancer()
        self.assertTupleEqual(x.shape, (137, 6))
        self.assertTupleEqual(y.shape, (137,))
        self.assert_structured_array_dtype(y, 'Status', 'Survival_in_days', 128)

    def test_load_aids(self):
        x, y = load_aids(endpoint="aids")
        self.assertTupleEqual(x.shape, (1151, 11))
        self.assertTupleEqual(y.shape, (1151,))
        self.assert_structured_array_dtype(y, 'censor', 'time', 96)
        self.assertFalse("censor_d" in x.columns)
        self.assertFalse("time_d" in x.columns)

        x, y = load_aids(endpoint="death")
        self.assertTupleEqual(x.shape, (1151, 11))
        self.assertTupleEqual(y.shape, (1151,))
        self.assert_structured_array_dtype(y, 'censor_d', 'time_d', 26)
        self.assertFalse("censor" in x.columns)
        self.assertFalse("time" in x.columns)

        self.assertRaisesRegex(ValueError, "endpoint must be 'aids' or 'death'",
                               load_aids, endpoint="foobar")

    def test_load_breast_cancer(self):
        x, y = load_breast_cancer()
        self.assertTupleEqual(x.shape, (198, 80))
        self.assertTupleEqual(y.shape, (198,))
        self.assert_structured_array_dtype(y, 'e.tdm', 't.tdm', 51)


def _make_and_write_data(fp, n_samples, n_features, with_index, with_labels, seed, column_prefix="V"):
    x, event, time = _make_survival_data(n_samples, n_features, seed)

    columns = ["{}{}".format(column_prefix, i) for i in range(n_features)]
    if with_labels:
        columns += ["event", "time"]
        arr = numpy.column_stack((x, event, time))
    else:
        arr = x

    if with_index:
        index = numpy.arange(n_samples, dtype=numpy.float_)
        numpy.random.RandomState(0).shuffle(index)
    else:
        index = None

    dataset = pandas.DataFrame(arr, index=index, columns=columns)
    dataset.index.name = "index"

    writearff(dataset, fp, index=with_index)
    return dataset


class TestLoadArffFile(TestCase):

    def assert_x_equal(self, x_true, x_train):
        tm.assert_index_equal(x_true.columns, x_train.columns, exact=True)
        tm.assert_index_equal(x_true.index, x_train.index, exact=True)

        tm.assert_frame_equal(x_true, x_train,
                              check_index_type=False,
                              check_column_type=True,
                              check_names=False,
                              check_less_precise=True)

    def assert_y_equal(self, y_true, y_train):
        self.assertTupleEqual(y_train.dtype.names, ("event", "time"))

        assert_array_equal(y_train["event"].astype(numpy.uint32),
                           y_true["event"].values.astype(numpy.uint32))
        assert_array_almost_equal(y_train["time"], y_true["time"].values)

    def test_load_with_index(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        try:
            dataset = _make_and_write_data(tmp, 100, 10, True, True, 0)

            x_train, y_train, x_test, y_test = load_arff_files_standardized(
                tmp.name, ["event", "time"], 1, survival=True,
                standardize_numeric=False, to_numeric=False)

            self.assertEqual(x_test, None)
            self.assertEqual(y_test, None)

            cols = ["event", "time"]
            x_true = dataset.drop(cols, axis=1)

            self.assert_x_equal(x_true, x_train)
            self.assert_y_equal(dataset, y_train)
        finally:
            os.unlink(tmp.name)

    def test_load_with_categorical_index_1(self):
        fp = StringIO(ARFF_CATEGORICAL_INDEX_1)
        x_train, y_train, x_test, y_test = load_arff_files_standardized(
            fp, ["label"], pos_label="yes", survival=False,
            standardize_numeric=False, to_numeric=False)

        self.assertEqual(x_test, None)
        self.assertEqual(y_test, None)

        self.assertTupleEqual(x_train.shape, (4, 2))
        self.assertTupleEqual(y_train.shape, (4, 1))

        index = pandas.Index(['SampleOne', 'SampleTwo', 'SampleThree', 'SampleFour'],
                             name='index', dtype=object)
        tm.assert_index_equal(x_train.index, index, exact=True)

        label = pandas.Series(pandas.Categorical(["yes", "no", "yes", "yes"], categories=["no", "yes"], ordered=False),
                              name="label", index=index)
        tm.assert_series_equal(y_train["label"], label, check_exact=True)

        value = pandas.Series([15.1, 13.8, -0.2, 2.453], name="value", index=index)
        tm.assert_series_equal(x_train["value"], value, check_exact=True)

        size = pandas.Series(pandas.Categorical(["medium", "large", "small", "large"],
                                                categories=["small", "medium", "large"], ordered=False),
                             name="size", index=index)
        tm.assert_series_equal(x_train["size"], size, check_exact=True)

    def test_load_with_categorical_index_2(self):
        fp = StringIO(ARFF_CATEGORICAL_INDEX_2)
        x_train, y_train, x_test, y_test = load_arff_files_standardized(
            fp, ["label"], pos_label="yes", survival=False,
            standardize_numeric=False, to_numeric=False)

        self.assertEqual(x_test, None)
        self.assertEqual(y_test, None)

        self.assertTupleEqual(x_train.shape, (5, 2))
        self.assertTupleEqual(y_train.shape, (5, 1))

        index = pandas.Index(['ASampleOne', 'ASampleTwo', 'ASampleThree', 'ASampleFour', 'ASampleFive'],
                             name='index', dtype=object)
        tm.assert_index_equal(x_train.index, index, exact=True)

        label = pandas.Series(pandas.Categorical(["no", "no", "yes", "yes", "no"], categories=["yes", "no"], ordered=False),
                              name="label", index=index)
        tm.assert_series_equal(y_train["label"], label, check_exact=True)

        value = pandas.Series([1.51, 1.38, -20, 245.3, 3.14], name="value", index=index)
        tm.assert_series_equal(x_train["value"], value, check_exact=True)

        size = pandas.Series(pandas.Categorical(["small", "small", "large", "small", "large"],
                                                categories=["small", "medium", "large"], ordered=False),
                             name="size", index=index)
        tm.assert_series_equal(x_train["size"], size, check_exact=True)

    def test_load_train_and_test_with_labels(self):
        tmp_train = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        tmp_test = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        try:
            train_dataset = _make_and_write_data(tmp_train, 100, 10, True, True, 0)
            test_dataset = _make_and_write_data(tmp_test, 20, 10, True, True, 0)

            x_train, y_train, x_test, y_test = load_arff_files_standardized(
                tmp_train.name, ["event", "time"], 1, path_testing=tmp_test.name,
                survival=True, standardize_numeric=False, to_numeric=False)

            cols = ["event", "time"]

            x_true = train_dataset.drop(cols, axis=1)
            self.assert_x_equal(x_true, x_train)
            self.assert_y_equal(train_dataset, y_train)

            x_true = test_dataset.drop(cols, axis=1)
            self.assert_x_equal(x_true, x_test)
            self.assert_y_equal(test_dataset, y_test)
        finally:
            os.unlink(tmp_train.name)
            os.unlink(tmp_test.name)

    def test_load_train_and_test_with_categorical_index(self):
        fp_1 = StringIO(ARFF_CATEGORICAL_INDEX_1)
        fp_2 = StringIO(ARFF_CATEGORICAL_INDEX_2)
        x_train, y_train, x_test, y_test = load_arff_files_standardized(
            fp_1, ["label"], pos_label="yes", path_testing=fp_2, survival=False,
            standardize_numeric=False, to_numeric=False)

        self.assertTupleEqual(x_train.shape, (4, 2))
        self.assertTupleEqual(x_test.shape, (5, 2))
        self.assertTupleEqual(y_train.shape, (4, 1))
        self.assertTupleEqual(y_test.shape, (5, 1))

        # Check train data
        train_index = pandas.Index(['SampleOne', 'SampleTwo', 'SampleThree', 'SampleFour'],
                                   name='index', dtype=object)
        tm.assert_index_equal(x_train.index, train_index, exact=True)

        train_label = pandas.Series(
            pandas.Categorical(["yes", "no", "yes", "yes"], categories=["no", "yes"], ordered=False),
            name="label", index=train_index)
        tm.assert_series_equal(y_train["label"], train_label, check_exact=True)

        train_value = pandas.Series([15.1, 13.8, -0.2, 2.453], name="value", index=train_index)
        tm.assert_series_equal(x_train["value"], train_value, check_exact=True)

        train_size = pandas.Series(pandas.Categorical(["medium", "large", "small", "large"],
                                                      categories=["small", "medium", "large"], ordered=False),
                                   name="size", index=train_index)
        tm.assert_series_equal(x_train["size"], train_size, check_exact=True)

        # Check test data
        test_index = pandas.Index(['ASampleOne', 'ASampleTwo', 'ASampleThree', 'ASampleFour', 'ASampleFive'],
                                  name='index', dtype=object)
        tm.assert_index_equal(x_test.index, test_index, exact=True)

        test_label = pandas.Series(
            pandas.Categorical(["no", "no", "yes", "yes", "no"], categories=["yes", "no"], ordered=False),
            name="label", index=test_index)
        tm.assert_series_equal(y_test["label"], test_label, check_exact=True)

        test_value = pandas.Series([1.51, 1.38, -20, 245.3, 3.14], name="value", index=test_index)
        tm.assert_series_equal(x_test["value"], test_value, check_exact=True)

        test_size = pandas.Series(pandas.Categorical(["small", "small", "large", "small", "large"],
                                                     categories=["small", "medium", "large"], ordered=False),
                                  name="size", index=test_index)
        tm.assert_series_equal(x_test["size"], test_size, check_exact=True)

    def test_load_train_and_test_no_labels(self):
        tmp_train = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        tmp_test = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        try:
            train_dataset = _make_and_write_data(tmp_train, 100, 10, True, True, 0)
            test_dataset = _make_and_write_data(tmp_test, 20, 10, True, False, 0)

            x_train, y_train, x_test, y_test = load_arff_files_standardized(
                tmp_train.name, ["event", "time"], 1, path_testing=tmp_test.name,
                survival=True, standardize_numeric=False, to_numeric=False)

            cols = ["event", "time"]

            x_true = train_dataset.drop(cols, axis=1)
            self.assert_x_equal(x_true, x_train)
            self.assert_y_equal(train_dataset, y_train)

            self.assert_x_equal(test_dataset, x_test)
            self.assertEqual(y_test, None)
        finally:
            os.unlink(tmp_train.name)
            os.unlink(tmp_test.name)

    def test_load_train_and_test_with_different_columns(self):
        tmp_train = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        tmp_test = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        try:
            _make_and_write_data(tmp_train, 100, 19, False, True, 0)
            _make_and_write_data(tmp_test, 20, 11, False, True, 0)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                load_arff_files_standardized(tmp_train.name, ["event", "time"], 1,
                                             path_testing=tmp_test.name,
                                             survival=True,
                                             standardize_numeric=False, to_numeric=False)

                self.assertEqual(1, len(w))
                self.assertEqual("Restricting columns to intersection between training and testing data",
                                 str(w[0].message))

        finally:
            os.unlink(tmp_train.name)
            os.unlink(tmp_test.name)

    def test_load_train_and_test_columns_dont_intersect(self):
        tmp_train = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        tmp_test = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
        try:
            _make_and_write_data(tmp_train, 100, 19, True, True, 0, column_prefix="A")
            _make_and_write_data(tmp_test, 20, 11, True, True, 0, column_prefix="B")

            self.assertRaisesRegex(ValueError, "columns of training and test data do not intersect",
                                   load_arff_files_standardized, tmp_train.name, ["event", "time"], 1,
                                   path_testing=tmp_test.name,
                                   survival=True,
                                   standardize_numeric=False, to_numeric=False)

        finally:
            os.unlink(tmp_train.name)
            os.unlink(tmp_test.name)


if __name__ == '__main__':
    run_module_suite()
