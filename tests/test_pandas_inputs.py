from importlib import import_module
import inspect
from os.path import dirname
import pkgutil

import pytest

import sksurv
from sksurv.base import SurvivalAnalysisMixin
from sksurv.datasets import load_whas500


def is_survival_mixin(x):
    return inspect.isclass(x) and x is not SurvivalAnalysisMixin and issubclass(x, SurvivalAnalysisMixin)


def all_survival_estimators():
    root = dirname(sksurv.__file__)
    all_classes = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(path=[root], prefix="sksurv."):
        # meta-estimators require base estimators
        if modname.startswith("sksurv.meta"):
            continue
        module = import_module(modname)
        for _name, cls in inspect.getmembers(module, is_survival_mixin):
            if inspect.isabstract(cls):
                continue
            all_classes.append(cls)
    return set(all_classes)


@pytest.mark.parametrize("estimator_cls", all_survival_estimators())
def test_pandas_inputs(estimator_cls):
    X, y = load_whas500()
    X = X.iloc[:50]
    y = y[:50]
    X = X.loc[:, ["age", "bmi", "chf", "gender"]].astype(float)

    estimator = estimator_cls()
    if "kernel" in estimator.get_params():
        estimator.set_params(kernel="rbf")
    estimator.fit(X, y)
    estimator.predict(X)
