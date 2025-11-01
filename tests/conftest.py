from collections import namedtuple
from pathlib import Path
import tempfile

import numpy as np
from packaging.version import Version
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

from sksurv.column import categorical_to_numeric, encode_categorical, standardize
from sksurv.datasets import load_breast_cancer, load_whas500
from sksurv.util import Surv

DataSet = namedtuple("DataSet", ["x", "y"])
DataSetWithNames = namedtuple("DataSetWithNames", ["x", "y", "names", "x_data_frame"])
SparseDataSet = namedtuple("SparseDataSet", ["x_dense", "x_sparse", "y"])

if Version(pd.__version__) >= Version("2.3.0"):
    pd.set_option("mode.copy_on_write", True)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks test as slow (deselect with '-m \"not slow\"')")


@pytest.fixture()
def fake_data():
    rng = np.random.default_rng()
    x = rng.standard_normal((100, 11))
    y = Surv.from_arrays(np.ones(100, dtype=bool), np.arange(1, 101, dtype=float))
    return x, y


@pytest.fixture()
def breast_cancer():
    X_str, y = load_breast_cancer()
    X_num = encode_categorical(X_str)
    return X_num, y


@pytest.fixture()
def make_whas500():
    """Load and standardize WHAS500 data."""

    def _make_whas500(with_mean=True, with_std=True, to_numeric=False):
        x, y = load_whas500()
        if with_mean:
            x = standardize(x, with_std=with_std)
        if to_numeric:
            x = categorical_to_numeric(x)
        names = ["(Intercept)"] + x.columns.tolist()
        return DataSetWithNames(x=x.to_numpy(), y=y, names=names, x_data_frame=x)

    return _make_whas500


@pytest.fixture()
def whas500_sparse_data():
    x, y = load_whas500()
    x_dense = categorical_to_numeric(x.select_dtypes(exclude=[float]))

    data = []
    index_i = []
    index_j = []
    for j, (_, col) in enumerate(x_dense.items()):
        idx = np.flatnonzero(col.values)
        data.extend([1] * len(idx))
        index_i.extend(idx)
        index_j.extend([j] * len(idx))

    x_sparse = coo_matrix((data, (index_i, index_j)))
    return SparseDataSet(x_dense=x_dense, x_sparse=x_sparse, y=y)


@pytest.fixture()
def whas500_uncomparable(make_whas500):
    whas500 = make_whas500(to_numeric=True)
    i = np.argmax(whas500.y["lenfol"])
    whas500.y["fstat"][:] = False
    whas500.y["fstat"][i] = True
    return whas500


@pytest.fixture()
def rossi():
    """Load rossi.csv"""
    p = Path(__file__)
    f = p.parent / "data" / "rossi.csv"
    data = pd.read_csv(f)
    y = Surv.from_dataframe("arrest", "week", data)
    x = data.drop(["arrest", "week"], axis=1)
    return DataSet(x=x, y=y)


@pytest.fixture(params=[np.inf, -np.inf, np.nan])
def non_finite_value(request):
    """Inf/-Inf/NaN value."""
    return request.param


@pytest.fixture()
def temp_file():
    f = tempfile.NamedTemporaryFile(mode="w", delete=False)
    fp = Path(f.name)
    yield f
    fp.unlink()
