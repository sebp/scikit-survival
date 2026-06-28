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
from sksurv.testing import FixtureParameterFactory, get_pandas_infer_string_context

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
    return np.random.default_rng(seed).standard_normal((n_samples, n_features))


def _make_survival_data(n_samples, n_features, seed):
    rnd = np.random.default_rng(seed)

    x = _make_features(n_samples, n_features, seed)
    event = rnd.binomial(1, 0.2, n_samples).astype(bool)
    time = rnd.exponential(25, size=n_samples)
    return x, event, time


def _make_classification_data(n_samples, n_features, n_classes, seed):
    rnd = np.random.default_rng(seed)

    x = _make_features(n_samples, n_features, seed)
    y = rnd.binomial(n_classes - 1, 0.2, 100)
    return x, y[:, np.newaxis]


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

    def _to_data_frame(self, data_arrays, columns):
        if not isinstance(data_arrays, tuple | list):
            data_arrays = (data_arrays,)

        df = [pd.DataFrame(data_array) for data_array in data_arrays]

        data = pd.concat(df, axis=1).set_axis(columns, axis=1)
        return data

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
        assert_array_equal(x_test, x_expected, strict=True)

    if not isinstance(y_expected, Skip):
        if y_expected is None:
            assert y_test is None
        elif isinstance(y_expected, tuple):
            assert y_test.dtype.names == ("event", "time")
            event, time = y_expected
            assert_array_equal(y_test["event"], event, strict=True)
            assert_array_almost_equal(y_test["time"], time)
        else:
            assert y_test.ndim == 2
            assert_array_equal(y_test.to_numpy(), y_expected, strict=True)


def assert_structured_array_dtype(arr, event, time, num_events):
    assert arr.dtype.names == (event, time)
    assert np.issubdtype(arr.dtype.fields[event][0], np.bool_)
    assert np.issubdtype(arr.dtype.fields[time][0], np.float64)
    assert arr[event].sum() == num_events


class TestLoadDatasets:
    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_load_whas500(infer_string_context):
        with infer_string_context:
            x, y = sdata.load_whas500()
            assert x.shape == (500, 14)
            assert y.shape == (500,)
            assert_structured_array_dtype(y, "fstat", "lenfol", 215)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_load_gbsg2(infer_string_context):
        with infer_string_context:
            x, y = sdata.load_gbsg2()
            assert x.shape == (686, 8)
            assert y.shape == (686,)
            assert_structured_array_dtype(y, "cens", "time", 299)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_load_veterans_lung_cancer(infer_string_context):
        with infer_string_context:
            x, y = sdata.load_veterans_lung_cancer()
            assert x.shape == (137, 6)
            assert y.shape == (137,)
            assert_structured_array_dtype(y, "Status", "Survival_in_days", 128)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_load_aids(infer_string_context):
        with infer_string_context:
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

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_load_breast_cancer(infer_string_context):
        with infer_string_context:
            x, y = sdata.load_breast_cancer()
            assert x.shape == (198, 80)
            assert y.shape == (198,)
            assert_structured_array_dtype(y, "e.tdm", "t.tdm", 51)

    @pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
    @staticmethod
    def test_load_flchain(infer_string_context):
        with infer_string_context:
            x, y = sdata.load_flchain()
            assert x.shape == (7874, 9)
            assert y.shape == (7874,)
            assert_structured_array_dtype(y, "death", "futime", 2169)


_POLARS_LOADERS = [
    ("whas500", sdata.load_whas500, {}, (500, 14), "fstat", "lenfol", 215),
    ("gbsg2", sdata.load_gbsg2, {}, (686, 8), "cens", "time", 299),
    ("veterans", sdata.load_veterans_lung_cancer, {}, (137, 6), "Status", "Survival_in_days", 128),
    ("aids", sdata.load_aids, {}, (1151, 11), "censor", "time", 96),
    ("aids_death", sdata.load_aids, {"endpoint": "death"}, (1151, 11), "censor_d", "time_d", 26),
    ("breast_cancer", sdata.load_breast_cancer, {}, (198, 80), "e.tdm", "t.tdm", 51),
    ("flchain", sdata.load_flchain, {}, (7874, 9), "death", "futime", 2169),
]


class TestLoadDatasetsPolars:
    """Pin output_type='polars' shape and dtype parity for every loader.

    For each loader: features come back as ``polars.DataFrame`` with the same
    shape as the pandas path, and the structured-array ``y`` is identical.
    The polars features carry ``pl.Enum`` for ARFF nominal columns (with the
    declared categories preserved, even ones absent from the data), while
    pandas features carry the equivalent ``pd.Categorical``.
    """

    @staticmethod
    @pytest.mark.parametrize(
        "name,loader,kwargs,shape,event_name,time_name,n_events",
        _POLARS_LOADERS,
        ids=[t[0] for t in _POLARS_LOADERS],
    )
    def test_polars_shape_and_y(name, loader, kwargs, shape, event_name, time_name, n_events):
        import polars as pl

        x, y = loader(output_type="polars", **kwargs)
        assert isinstance(x, pl.DataFrame), f"{name}: x is {type(x).__name__}"
        assert x.shape == shape
        assert y.shape == (shape[0],)
        assert_structured_array_dtype(y, event_name, time_name, n_events)

    @staticmethod
    def test_polars_matches_pandas_columns():
        """Polars and pandas paths produce identical column lists (modulo
        dataframe-library container differences) for every loader."""
        import polars as pl

        for name, loader, kwargs, _shape, *_ in _POLARS_LOADERS:
            x_pd, _ = loader(**kwargs)
            x_pl, _ = loader(output_type="polars", **kwargs)
            assert list(x_pl.columns) == list(x_pd.columns), f"column mismatch for {name}"
            assert isinstance(x_pl, pl.DataFrame)

    @staticmethod
    def test_polars_nominal_columns_are_enum():
        """ARFF nominal columns must surface as pl.Enum in the polars output,
        preserving the declared category list from the ARFF schema."""
        import polars as pl

        x, _ = sdata.load_gbsg2(output_type="polars")
        # GBSG2 has nominal columns "horTh", "menostat", "tgrade"
        for col in ("horTh", "menostat", "tgrade"):
            assert isinstance(x.schema[col], pl.Enum), f"{col}: expected pl.Enum, got {x.schema[col]!r}"

    @staticmethod
    def test_load_bmt_polars():
        import polars as pl

        x, _ = sdata.load_bmt(output_type="polars")
        assert isinstance(x, pl.DataFrame)
        assert x.shape == (35, 1)
        assert x["dis"].dtype == pl.Enum(["0", "1"])

    @staticmethod
    def test_load_cgvhd_polars():
        import polars as pl

        x, _ = sdata.load_cgvhd(output_type="polars")
        assert isinstance(x, pl.DataFrame)
        assert x.shape == (100, 4)
        assert list(x.columns) == ["dx", "tx", "extent", "age"]
        # Nominal columns must surface as pl.Enum (declared categories
        # preserved). Plain ``pl.from_pandas`` would have demoted them to
        # pl.Categorical, which silently loses the ARFF schema's declared
        # category list.
        for col in ("dx", "tx", "extent"):
            assert isinstance(x.schema[col], pl.Enum), f"{col}: expected pl.Enum, got {x.schema[col]!r}"

    @staticmethod
    def test_load_arff_files_standardized_polars_drops_index_column(tmp_path):
        """ARFF files with an ``index`` attribute must not leak that column
        into the polars feature set.

        Regression for a silent-divergence bug: the pandas branch promotes
        ``index`` to the row index (so it is excluded from features), but
        the earlier polars branch left it as a regular column, silently
        feeding downstream ``standardize`` / ``categorical_to_numeric`` /
        row concatenation with an extra (and possibly string-typed) feature.
        """
        import polars as pl

        arff = (
            "@relation test\n"
            "@attribute index numeric\n"
            "@attribute x1 numeric\n"
            "@attribute x2 numeric\n"
            "@attribute event {0, 1}\n"
            "@attribute time numeric\n"
            "@data\n"
            "0,1.0,10.0,0,100\n"
            "1,2.0,20.0,1,80\n"
            "2,3.0,30.0,1,60\n"
            "3,4.0,40.0,0,150\n"
            "4,5.0,50.0,1,90\n"
        )
        path = tmp_path / "with_index.arff"
        path.write_text(arff)

        x_pd, y_pd, _, _ = sdata.load_arff_files_standardized(
            str(path),
            attr_labels=["event", "time"],
            pos_label="1",
        )
        x_pl, y_pl, _, _ = sdata.load_arff_files_standardized(
            str(path),
            attr_labels=["event", "time"],
            pos_label="1",
            output_type="polars",
        )
        assert isinstance(x_pl, pl.DataFrame)
        assert list(x_pl.columns) == list(x_pd.columns)
        assert "index" not in x_pl.columns
        np.testing.assert_allclose(x_pl.to_numpy(), x_pd.to_numpy())
        assert (y_pd == y_pl).all()

    @staticmethod
    def test_load_arff_files_standardized_polars_preserves_enum(tmp_path):
        """``load_arff_files_standardized(output_type='polars')`` keeps every
        ARFF nominal column as pl.Enum (declared categories preserved).

        Internally the function does its standardization / numeric conversion
        in pandas; the conversion back to polars must not demote pd.Categorical
        to pl.Categorical.
        """
        import polars as pl

        arff = (
            "@relation test\n"
            "@attribute index numeric\n"
            "@attribute feat numeric\n"
            "@attribute grade {I, II, III, IV}\n"
            "@attribute event {0, 1}\n"
            "@attribute time numeric\n"
            "@data\n"
            "0,1.0,I,0,10\n"
            "1,2.0,II,1,20\n"
            "2,3.0,III,1,15\n"
            "3,4.0,I,0,25\n"
            "4,5.0,II,1,30\n"
        )
        path = tmp_path / "tiny.arff"
        path.write_text(arff)

        x_train, _, _, _ = sdata.load_arff_files_standardized(
            str(path),
            attr_labels=["event", "time"],
            pos_label="1",
            output_type="polars",
            to_numeric=False,
            standardize_numeric=False,
        )
        assert isinstance(x_train, pl.DataFrame)
        assert isinstance(x_train.schema["grade"], pl.Enum), f"grade dtype: {x_train.schema['grade']!r}"
        assert x_train["grade"].cat.get_categories().to_list() == ["I", "II", "III", "IV"]

    @staticmethod
    @pytest.mark.parametrize("loader,kwargs", [(t[1], t[2]) for t in _POLARS_LOADERS])
    def test_polars_invalid_output_type(loader, kwargs):
        with pytest.raises(ValueError, match="output_type must be 'pandas' or 'polars'"):
            loader(output_type="numpy", **kwargs)


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
        np.random.default_rng(0).shuffle(index)
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
        y_true["event"].to_numpy(dtype=np.uint32),
    )
    assert_array_almost_equal(y_train["time"], y_true["time"].to_numpy())


class LoadArffFilesCases(FixtureParameterFactory):
    @property
    def arff_1(self):
        return StringIO(ARFF_CATEGORICAL_INDEX_1)

    @property
    def arff_2(self):
        return StringIO(ARFF_CATEGORICAL_INDEX_2)

    def data_with_categorical_index_1(self):
        values = ["SampleOne", "SampleTwo", "SampleThree", "SampleFour"]
        index = pd.Index(values, name="index", dtype="str")
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
        index = pd.Index(values, name="index", dtype="str")

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


@pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
@pytest.mark.parametrize("make_data_fn", LoadArffFilesCases().get_cases_func())
def test_load_arff_files(make_data_fn, infer_string_context):
    with infer_string_context:
        (
            args,
            kwargs,
            x_train_expected,
            y_train_expected,
            x_test_expected,
            y_test_expected,
        ) = make_data_fn()
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


@pytest.mark.parametrize("infer_string_context", get_pandas_infer_string_context())
@pytest.mark.parametrize("make_data_fn", LoadArffFilesWithTempFileCases().get_cases_func())
def test_load_from_temp_file(make_data_fn, temp_file_pair, infer_string_context):
    with infer_string_context:
        (
            args_train,
            kwargs_train,
            args_test,
            kwargs_test,
            errors_expected,
        ) = make_data_fn()
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


@pytest.mark.parametrize("output_type", ["pandas", "polars"])
def test_load_arff_files_standardized_testing_without_labels_and_attr_labels_none(tmp_path, output_type):
    train_path = tmp_path / "train.arff"
    test_path = tmp_path / "test.arff"
    train_path.write_text(
        "@relation train\n" "@attribute x numeric\n" "@attribute group {a,b}\n" "@data\n" "1,a\n" "2,b\n"
    )
    test_path.write_text(
        "@relation test\n" "@attribute x numeric\n" "@attribute group {a,b}\n" "@data\n" "3,a\n" "4,b\n"
    )

    x_train, y_train, x_test, y_test = sdata.load_arff_files_standardized(
        str(train_path),
        attr_labels=None,
        path_testing=str(test_path),
        survival=False,
        standardize_numeric=False,
        to_numeric=False,
        output_type=output_type,
    )

    assert y_train is None
    assert y_test is None
    assert set(x_train.columns) == {"x", "group"}
    assert list(x_test.columns) == list(x_train.columns)
