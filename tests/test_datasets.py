from contextlib import ExitStack
from contextlib import nullcontext as does_not_raise
from io import StringIO
from pathlib import Path
import tempfile

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas as pd
import pandas.testing as tm
import pytest

import sksurv.datasets as sdata
from sksurv.io import writearff
from sksurv.testing import FixtureParameterFactory

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


class Skip:
    pass


@pytest.fixture()
def temp_file_pair():
    tmp_train = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)
    tmp_test = tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False)

    yield tmp_train, tmp_test

    Path(tmp_train.name).unlink()
    Path(tmp_test.name).unlink()


def _make_features(n_samples, n_features, seed):
    return np.random.RandomState(seed).randn(n_samples, n_features)


def _make_survival_data(n_samples, n_features, seed):
    rnd = np.random.RandomState(seed)

    x = _make_features(n_samples, n_features, seed)
    event = rnd.binomial(1, 0.2, n_samples)
    time = rnd.exponential(25, size=n_samples)
    return x, event, time


def _make_classification_data(n_samples, n_features, n_classes, seed):
    rnd = np.random.RandomState(seed)

    x = _make_features(n_samples, n_features, seed)
    y = rnd.binomial(n_classes - 1, 0.2, 100)
    return x, y


class GetXyCases(FixtureParameterFactory):
    @property
    def survival_data(self):
        return _make_survival_data(100, 10, 0)

    @property
    def features(self):
        return _make_features(100, 10, 0)

    @property
    def attr_labels(self):
        return ["event", "time"]

    def _to_data_frame(self, data, columns):
        if isinstance(data, tuple | list):
            data = np.column_stack(data)
        return pd.DataFrame(data, columns=columns)

    @property
    def columns(self):
        return [f"V{i}" for i in range(10)]

    def data_survival_data(self):
        x, event, time = self.survival_data
        attr_labels = self.attr_labels
        dataset = self._to_data_frame((x, event, time), self.columns + attr_labels)

        args = (dataset, attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        return args, kwargs, x, (event, time), does_not_raise()

    def data_no_label(self):
        x = self.features
        attr_labels = [None, None]
        dataset = self._to_data_frame(x, self.columns)

        args = (dataset, attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        return args, kwargs, x, None, does_not_raise()

    def data_too_many_labels(self):
        x, event, time = self.survival_data
        attr_labels = self.attr_labels + ["random"]
        dataset = self._to_data_frame((x, event, time), self.columns + self.attr_labels)

        args = (dataset, attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        error = pytest.raises(
            ValueError,
            match="expected sequence of length two for attr_labels, but got 3",
        )
        return args, kwargs, Skip(), Skip(), error

    def data_too_little_labels_0(self):
        x, event, time = self.survival_data
        attr_labels = self.attr_labels[:1]
        dataset = self._to_data_frame((x, event, time), self.columns + self.attr_labels)

        args = (dataset, attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        error = pytest.raises(
            ValueError,
            match="expected sequence of length two for attr_labels, but got 1",
        )
        return args, kwargs, Skip(), Skip(), error

    def data_too_little_labels_1(self):
        x, event, time = self.survival_data
        attr_labels = []
        dataset = self._to_data_frame((x, event, time), self.columns + self.attr_labels)

        args = (dataset, attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        error = pytest.raises(
            ValueError,
            match="expected sequence of length two for attr_labels, but got 0",
        )
        return args, kwargs, Skip(), Skip(), error

    def data_no_pos_label(self):
        x, event, time = self.survival_data
        attr_labels = self.attr_labels
        dataset = self._to_data_frame((x, event, time), self.columns + attr_labels)

        args = (dataset, attr_labels)
        kwargs = {"survival": True}
        error = pytest.raises(
            ValueError,
            match="pos_label needs to be specified if survival=True",
        )
        return args, kwargs, Skip(), Skip(), error

    def data_classification(self):
        x, label = _make_classification_data(100, 10, 6, 0)
        attr_labels = ["class_label"]
        dataset = self._to_data_frame((x, label), self.columns + attr_labels)

        args = (dataset, attr_labels)
        kwargs = {"survival": False}
        return args, kwargs, x, label, does_not_raise()

    def data_classification_no_label(self):
        x = self.features
        attr_labels = None
        dataset = self._to_data_frame(x, self.columns)

        args = (dataset, attr_labels)
        kwargs = {"survival": False}
        return args, kwargs, x, None, does_not_raise()


@pytest.mark.parametrize("args,kwargs,x_expected,y_expected,error_expected", GetXyCases().get_cases())
def test_get_xy(args, kwargs, x_expected, y_expected, error_expected):
    with error_expected:
        x_test, y_test = sdata.get_x_y(*args, **kwargs)

    if not isinstance(x_expected, Skip):
        assert_array_equal(x_test, x_expected)

    if not isinstance(y_expected, Skip):
        if y_expected is None:
            assert y_test is None
        elif isinstance(y_expected, tuple):
            assert y_test.dtype.names == ("event", "time")
            event, time = y_expected
            assert_array_equal(y_test["event"].astype(np.uint32), event.astype(np.uint32))
            assert_array_almost_equal(y_test["time"], time)
        else:
            assert y_test.ndim == 2
            assert_array_equal(y_test.values.ravel(), y_expected)


def assert_structured_array_dtype(arr, event, time, num_events):
    assert arr.dtype.names == (event, time)
    assert np.issubdtype(arr.dtype.fields[event][0], np.bool_)
    assert np.issubdtype(arr.dtype.fields[time][0], np.float64)
    assert arr[event].sum() == num_events


class TestLoadDatasets:
    @staticmethod
    def test_load_whas500():
        x, y = sdata.load_whas500()
        assert x.shape == (500, 14)
        assert y.shape == (500,)
        assert_structured_array_dtype(y, "fstat", "lenfol", 215)

    @staticmethod
    def test_load_gbsg2():
        x, y = sdata.load_gbsg2()
        assert x.shape == (686, 8)
        assert y.shape == (686,)
        assert_structured_array_dtype(y, "cens", "time", 299)

    @staticmethod
    def test_load_veterans_lung_cancer():
        x, y = sdata.load_veterans_lung_cancer()
        assert x.shape == (137, 6)
        assert y.shape == (137,)
        assert_structured_array_dtype(y, "Status", "Survival_in_days", 128)

    @staticmethod
    def test_load_aids():
        x, y = sdata.load_aids(endpoint="aids")
        assert x.shape == (1151, 11)
        assert y.shape == (1151,)
        assert_structured_array_dtype(y, "censor", "time", 96)
        assert "censor_d" not in x.columns
        assert "time_d" not in x.columns

        x, y = sdata.load_aids(endpoint="death")
        assert x.shape == (1151, 11)
        assert y.shape == (1151,)
        assert_structured_array_dtype(y, "censor_d", "time_d", 26)
        assert "censor" not in x.columns
        assert "time" not in x.columns

        with pytest.raises(ValueError, match="endpoint must be 'aids' or 'death'"):
            sdata.load_aids(endpoint="foobar")

    @staticmethod
    def test_load_breast_cancer():
        x, y = sdata.load_breast_cancer()
        assert x.shape == (198, 80)
        assert y.shape == (198,)
        assert_structured_array_dtype(y, "e.tdm", "t.tdm", 51)

    @staticmethod
    def test_load_flchain():
        x, y = sdata.load_flchain()
        assert x.shape == (7874, 9)
        assert y.shape == (7874,)
        assert_structured_array_dtype(y, "death", "futime", 2169)


def _make_and_write_data(fp, n_samples, n_features, with_index, with_labels, seed, column_prefix="V"):
    x, event, time = _make_survival_data(n_samples, n_features, seed)

    columns = [f"{column_prefix}{i}" for i in range(n_features)]
    if with_labels:
        columns += ["event", "time"]
        arr = np.column_stack((x, event, time))
    else:
        arr = x

    if with_index:
        index = np.arange(n_samples, dtype=float)
        np.random.RandomState(0).shuffle(index)
    else:
        index = None

    dataset = pd.DataFrame(arr, index=index, columns=columns)
    dataset.index.name = "index"

    writearff(dataset, fp, index=with_index)
    return dataset


def assert_x_equal(x_true, x_train):
    tm.assert_index_equal(x_true.columns, x_train.columns, exact=True)
    tm.assert_index_equal(x_true.index, x_train.index, exact=True)

    tm.assert_frame_equal(
        x_true,
        x_train,
        check_index_type=False,
        check_column_type=True,
        check_names=False,
    )


def assert_y_equal(y_true, y_train):
    assert y_train.dtype.names == ("event", "time")

    assert_array_equal(
        y_train["event"].astype(np.uint32),
        y_true["event"].values.astype(np.uint32),
    )
    assert_array_almost_equal(y_train["time"], y_true["time"].values)


class LoadArffFilesCases(FixtureParameterFactory):
    @property
    def arff_1(self):
        return StringIO(ARFF_CATEGORICAL_INDEX_1)

    @property
    def arff_2(self):
        return StringIO(ARFF_CATEGORICAL_INDEX_2)

    def data_with_categorical_index_1(self):
        values = ["SampleOne", "SampleTwo", "SampleThree", "SampleFour"]
        index = pd.Index(values, name="index", dtype=object)
        x = pd.DataFrame.from_dict(
            {
                "size": pd.Series(
                    pd.Categorical(
                        ["medium", "large", "small", "large"],
                        categories=["small", "medium", "large"],
                        ordered=False,
                    ),
                    name="size",
                ),
                "value": pd.Series([15.1, 13.8, -0.2, 2.453], name="value"),
            }
        )
        x.index = index

        y = pd.DataFrame.from_dict(
            {
                "label": pd.Series(
                    pd.Categorical(["yes", "no", "yes", "yes"], categories=["no", "yes"], ordered=False), name="label"
                )
            }
        )
        y.index = index

        args = (self.arff_1, ["label"])
        kwargs = {
            "pos_label": "yes",
            "survival": False,
            "standardize_numeric": False,
            "to_numeric": False,
        }

        return args, kwargs, x, y, None, None

    def data_with_categorical_index_2(self):
        values = ["ASampleOne", "ASampleTwo", "ASampleThree", "ASampleFour", "ASampleFive"]
        index = pd.Index(values, name="index", dtype=object)

        y = pd.DataFrame.from_dict(
            {
                "label": pd.Series(
                    pd.Categorical(["no", "no", "yes", "yes", "no"], categories=["yes", "no"], ordered=False),
                    name="label",
                )
            }
        )
        y.index = index

        x = pd.DataFrame.from_dict(
            {
                "size": pd.Series(
                    pd.Categorical(
                        ["small", "small", "large", "small", "large"],
                        categories=["small", "medium", "large"],
                        ordered=False,
                    ),
                    name="size",
                ),
                "value": pd.Series([1.51, 1.38, -20, 245.3, 3.14], name="value"),
            }
        )
        x.index = index

        args = (self.arff_2, ["label"])
        kwargs = {
            "pos_label": "yes",
            "survival": False,
            "standardize_numeric": False,
            "to_numeric": False,
        }

        return args, kwargs, x, y, None, None

    def data_with_categorical_index(self):
        _, _, x_train, y_train, _, _ = self.data_with_categorical_index_1()
        _, _, x_test, y_test, _, _ = self.data_with_categorical_index_2()

        args = (self.arff_1, ["label"])
        kwargs = {
            "pos_label": "yes",
            "path_testing": self.arff_2,
            "survival": False,
            "standardize_numeric": False,
            "to_numeric": False,
        }

        return args, kwargs, x_train, y_train, x_test, y_test


@pytest.mark.parametrize(
    "args,kwargs,x_train_expected,y_train_expected,x_test_expected,y_test_expected",
    LoadArffFilesCases().get_cases(),
)
def test_load_arff_files(
    args,
    kwargs,
    x_train_expected,
    y_train_expected,
    x_test_expected,
    y_test_expected,
):
    x_train, y_train, x_test, y_test = sdata.load_arff_files_standardized(
        *args,
        **kwargs,
    )

    tm.assert_frame_equal(x_train, x_train_expected, check_exact=True)
    tm.assert_frame_equal(y_train, y_train_expected, check_exact=True)

    if x_test_expected is None:
        assert x_test is None
    else:
        tm.assert_frame_equal(x_test, x_test_expected, check_exact=True)

    if y_test_expected is None:
        assert y_test is None
    else:
        tm.assert_frame_equal(y_test, y_test_expected, check_exact=True)


class LoadArffFilesWithTempFileCases(FixtureParameterFactory):
    def data_with_index(self):
        args_train = (100, 10, True, True, 0)
        args_test = None
        return args_train, {}, args_test, {}, (does_not_raise(),)

    def data_train_and_test_with_labels(self):
        args_train = (100, 10, True, True, 0)
        args_test = (20, 10, True, True, 0)
        return args_train, {}, args_test, {}, (does_not_raise(),)

    def data_train_and_test_no_labels(self):
        args_train = (100, 10, True, True, 0)
        args_test = (20, 10, True, False, 0)
        return args_train, {}, args_test, {}, (does_not_raise(),)

    def data_train_and_test_with_different_columns(self):
        args_train = (100, 19, False, True, 0)
        args_test = (20, 11, False, True, 0)
        error = pytest.warns(
            UserWarning,
            match="Restricting columns to intersection between training and testing data",
        )
        return args_train, {}, args_test, {}, (error,)

    def data_train_and_test_columns_dont_intersect(self):
        args_train = (100, 19, True, True, 0)
        kwargs_train = {"column_prefix": "A"}
        args_test = (20, 11, True, True, 0)
        kwargs_test = {"column_prefix": "B"}
        error = pytest.raises(
            ValueError,
            match="columns of training and test data do not intersect",
        )
        warning = pytest.warns(
            UserWarning,
            match="Restricting columns to intersection between training and testing data",
        )
        return (
            args_train,
            kwargs_train,
            args_test,
            kwargs_test,
            (
                error,
                warning,
            ),
        )


@pytest.mark.parametrize(
    "args_train,kwargs_train,args_test,kwargs_test,errors_expected", LoadArffFilesWithTempFileCases().get_cases()
)
def test_load_from_temp_file(args_train, kwargs_train, args_test, kwargs_test, errors_expected, temp_file_pair):
    tmp_train, tmp_test = temp_file_pair

    train_dataset = _make_and_write_data(tmp_train, *args_train, **kwargs_train)
    if args_test is not None:
        test_dataset = _make_and_write_data(tmp_test, *args_test, **kwargs_test)
        path_testing = tmp_test.name
        check_y_test = args_test[-2]  # with_label
    else:
        test_dataset = None
        path_testing = None
        check_y_test = False
        tmp_test.close()

    with ExitStack() as stack:
        for error_expected in errors_expected:
            stack.enter_context(error_expected)
        x_train, y_train, x_test, y_test = sdata.load_arff_files_standardized(
            tmp_train.name,
            ["event", "time"],
            1,
            path_testing=path_testing,
            survival=True,
            standardize_numeric=False,
            to_numeric=False,
        )

    if all(not isinstance(err, does_not_raise) for err in errors_expected):
        return

    cols = ["event", "time"]

    x_true = train_dataset.drop(cols, axis=1)
    assert_x_equal(x_true, x_train)
    assert_y_equal(train_dataset, y_train)

    if test_dataset is not None:
        x_true = test_dataset
        if check_y_test:
            assert_y_equal(test_dataset, y_test)
            x_true = test_dataset.drop(cols, axis=1)
        else:
            assert y_test is None
        assert_x_equal(x_true, x_test)
    else:
        assert x_test is None
        assert y_test is None
