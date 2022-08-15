import numpy
from numpy.testing import assert_array_equal
import pandas
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
    assert_array_equal(estimator.feature_names_in_, numpy.asarray(X_df.columns, dtype=object))
    estimator.predict(X_df)

    msg = "The feature names should match those that were passed"
    X_bad = pandas.DataFrame(X_np, columns=X_df.columns.tolist()[::-1])
    with pytest.warns(FutureWarning, match=msg):
        estimator.predict(X_bad)

    # warns when fitted on dataframe and transforming a ndarray
    msg = "X does not have valid feature names, but {} was fitted with feature names".format(
        estimator_cls.__name__
    )
    with pytest.warns(UserWarning, match=msg):
        estimator.predict(X_np)

    # warns when fitted on a ndarray and transforming dataframe
    msg = "X has feature names, but {} was fitted without feature names".format(estimator_cls.__name__)
    estimator.fit(X_np, y)
    with pytest.warns(UserWarning, match=msg):
        estimator.predict(X_df)
