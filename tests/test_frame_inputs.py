import numpy as np
from numpy.testing import assert_array_equal
import pytest

from sksurv.datasets import load_whas500
from sksurv.testing import all_survival_estimators


@pytest.fixture(scope="module")
def whas500_numeric_columns():
    X, y = load_whas500()
    columns = ["age", "bmi", "diasbp", "hr"]
    data = {name: X[name].to_numpy(dtype=float)[:50] for name in columns}
    return data, y[:50]


@pytest.mark.filterwarnings("ignore:NaiveSurvivalSVM is deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:The 'ecos' solver will be removed in a future release.:FutureWarning")
@pytest.mark.parametrize("estimator_cls", all_survival_estimators())
def test_frame_inputs(estimator_cls, dataframe_backend, whas500_numeric_columns):
    data, y = whas500_numeric_columns
    X_df = dataframe_backend.make_frame(data)
    X_np = np.column_stack(list(data.values()))

    estimator = estimator_cls()
    if "kernel" in estimator.get_params():
        estimator.set_params(kernel="rbf")
    estimator.fit(X_df, y)
    assert hasattr(estimator, "feature_names_in_")
    assert_array_equal(estimator.feature_names_in_, np.asarray(list(data), dtype=object), strict=True)
    estimator.predict(X_df)

    msg = "Feature names must be in the same order as they were in fit"
    X_reordered = dataframe_backend.make_frame(dict(zip(reversed(list(data)), data.values(), strict=True)))
    with pytest.raises(ValueError, match=msg):
        estimator.predict(X_reordered)

    # warns when fitted on a dataframe and predicting from an ndarray
    msg = f"X does not have valid feature names, but {estimator_cls.__name__} was fitted with feature names"
    with pytest.warns(UserWarning, match=msg):
        estimator.predict(X_np)

    # warns when fitted on an ndarray and predicting from a dataframe
    msg = f"X has feature names, but {estimator_cls.__name__} was fitted without feature names"
    estimator.fit(X_np, y)
    with pytest.warns(UserWarning, match=msg):
        estimator.predict(X_df)
