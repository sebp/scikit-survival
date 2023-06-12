import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from sksurv.datasets import load_whas500
from sksurv.testing import all_survival_estimators


@pytest.mark.parametrize("estimator_cls", all_survival_estimators())
def test_pandas_inputs(estimator_cls):
    X, y = load_whas500()
    X = X.iloc[:50]
    y = y[:50]
    X_df = X.loc[:, ["age", "bmi", "chf", "gender"]].astype(float)
    X_np = X_df.values

    estimator = estimator_cls()
    if "kernel" in estimator.get_params():
        estimator.set_params(kernel="rbf")
    estimator.fit(X_df, y)
    assert hasattr(estimator, "feature_names_in_")
    assert_array_equal(estimator.feature_names_in_, np.asarray(X_df.columns, dtype=object))
    estimator.predict(X_df)

    msg = "Feature names must be in the same order as they were in fit"
    X_bad = pd.DataFrame(X_np, columns=X_df.columns.tolist()[::-1])
    with pytest.raises(ValueError, match=msg):
        estimator.predict(X_bad)

    # warns when fitted on dataframe and transforming a ndarray
    msg = f"X does not have valid feature names, but {estimator_cls.__name__} was fitted with feature names"
    with pytest.warns(UserWarning, match=msg):
        estimator.predict(X_np)

    # warns when fitted on a ndarray and transforming dataframe
    msg = f"X has feature names, but {estimator_cls.__name__} was fitted without feature names"
    estimator.fit(X_np, y)
    with pytest.warns(UserWarning, match=msg):
        estimator.predict(X_df)
