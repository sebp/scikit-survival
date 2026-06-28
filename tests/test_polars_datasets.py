"""Tests for ``sksurv.datasets`` (get_x_y) with polars input."""

from contextlib import nullcontext as does_not_raise

from dataframe_test_utils import to_polars_dataframe
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import polars as pl
import pytest

import sksurv.datasets as sdata
from sksurv.testing import FixtureParameterFactory


class _Skip:
    """Sentinel to skip the corresponding assertion in a parametrized test."""


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
    y = rnd.binomial(n_classes - 1, 0.2, n_samples)
    return x, y[:, np.newaxis]


def _make_competing_risks_data(n_samples, n_features, seed):
    rnd = np.random.default_rng(seed)

    x = _make_features(n_samples, n_features, seed)
    event = rnd.integers(0, 3, n_samples)
    time = rnd.exponential(25, size=n_samples)
    return x, event, time


class GetXyPolarsCases(FixtureParameterFactory):
    """Test cases for ``sksurv.datasets.get_x_y`` with polars eager/lazy input.

    Output policy: container-preserving. For polars input, ``x_frame`` comes
    back as ``polars.DataFrame``; ``y`` is a numpy structured array
    (``survival=True``) or a polars container (``survival=False``).
    """

    n_samples = 100
    n_features = 10

    @property
    def columns(self):
        return [f"V{i}" for i in range(self.n_features)]

    @property
    def attr_labels(self):
        return ["event", "time"]

    def _make_polars_eager(self, data_arrays, columns):
        import pandas as pd

        if not isinstance(data_arrays, tuple | list):
            data_arrays = (data_arrays,)

        df = [pd.DataFrame(data_array) for data_array in data_arrays]

        data = pd.concat(df, axis=1).set_axis(columns, axis=1)
        return to_polars_dataframe(data)

    def data_polars_eager_survival(self):
        x, event, time = _make_survival_data(self.n_samples, self.n_features, 0)
        df = self._make_polars_eager((x, event, time), self.columns + self.attr_labels)
        args = (df, self.attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        return args, kwargs, x, (event, time), does_not_raise()

    def data_polars_eager_no_label(self):
        x = _make_features(self.n_samples, self.n_features, 0)
        df = self._make_polars_eager(x, self.columns)
        args = (df, [None, None])
        kwargs = {"pos_label": 1, "survival": True}
        return args, kwargs, x, None, does_not_raise()

    def data_polars_eager_competing_risks(self):
        x, event, time = _make_competing_risks_data(self.n_samples, self.n_features, 0)
        df = self._make_polars_eager((x, event, time), self.columns + self.attr_labels)
        args = (df, self.attr_labels)
        kwargs = {"competing_risks": True, "survival": True}
        return args, kwargs, x, (event, time), does_not_raise()

    def data_polars_eager_classification(self):
        x, label = _make_classification_data(self.n_samples, self.n_features, 6, 0)
        df = self._make_polars_eager((x, label), self.columns + ["class_label"])
        args = (df, ["class_label"])
        kwargs = {"survival": False}
        return args, kwargs, x, label, does_not_raise()

    def data_polars_eager_classification_no_label(self):
        x = _make_features(self.n_samples, self.n_features, 0)
        df = self._make_polars_eager(x, self.columns)
        args = (df, None)
        kwargs = {"survival": False}
        return args, kwargs, x, None, does_not_raise()

    def data_polars_wrong_class(self):
        x = _make_features(self.n_samples, self.n_features, 0)
        args = (x, self.attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        err = pytest.raises(
            TypeError,
            match=r"expected pandas\.DataFrame or polars\.DataFrame",
        )
        return args, kwargs, _Skip(), _Skip(), err

    def data_polars_series_wrong_class(self):
        s = pl.Series("foo", [1, 2, 3])
        args = (s, self.attr_labels)
        kwargs = {"pos_label": 1, "survival": True}
        err = pytest.raises(
            TypeError,
            match=r"expected pandas\.DataFrame or polars\.DataFrame",
        )
        return args, kwargs, _Skip(), _Skip(), err


@pytest.mark.parametrize("args,kwargs,x_expected,y_expected,error_expected", GetXyPolarsCases().get_cases())
def test_get_xy_polars(args, kwargs, x_expected, y_expected, error_expected):
    with error_expected:
        x_test, y_test = sdata.get_x_y(*args, **kwargs)

    if not isinstance(x_expected, _Skip):
        assert isinstance(x_test, pl.DataFrame), f"expected polars.DataFrame, got {type(x_test)!r}"
        assert_array_equal(x_test.to_numpy(), x_expected, strict=True)

    if not isinstance(y_expected, _Skip):
        if y_expected is None:
            assert y_test is None
        elif isinstance(y_expected, tuple):
            assert y_test.dtype.names == ("event", "time")
            event, time = y_expected
            assert_array_equal(y_test["event"], event, strict=True)
            assert_array_almost_equal(y_test["time"], time)
        else:
            assert isinstance(y_test, pl.DataFrame), f"expected polars.DataFrame, got {type(y_test)!r}"
            assert y_test.shape[1] == 1
            assert_array_equal(y_test.to_numpy(), y_expected, strict=True)


class TestGetXYLazyFrameRejected:
    """``get_x_y(LazyFrame, ...)`` must reject the lazy frame with the same
    actionable ``TypeError`` as every other entry point.
    """

    @staticmethod
    def test_get_x_y_lazyframe_rejected():
        from sksurv.datasets import get_x_y

        lf = pl.LazyFrame({"e": [True, False], "t": [1.0, 2.0], "x": [1, 2]})
        with pytest.raises(TypeError, match=r"polars\.LazyFrame is not supported; call \.collect\(\)"):
            get_x_y(lf, None, survival=False)

    @staticmethod
    def test_get_x_y_string_label_returns_series_pandas():
        """``get_x_y(pd.DataFrame, "label", survival=False)`` historically
        returned a ``pd.Series`` (via ``dataset.loc[:, "label"]``). Preserve
        that contract.
        """
        import pandas as pd

        from sksurv.datasets import get_x_y

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        _, y = get_x_y(df, "a", survival=False)
        assert isinstance(y, pd.Series), f"expected pd.Series, got {type(y).__name__}"

    @staticmethod
    def test_get_x_y_string_label_returns_series_polars():
        """Mirror the pandas contract on polars: string label returns
        ``polars.Series``.
        """
        from sksurv.datasets import get_x_y

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        _, y = get_x_y(df, "a", survival=False)
        assert isinstance(y, pl.Series), f"expected pl.Series, got {type(y).__name__}"
