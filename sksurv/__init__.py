from importlib.metadata import PackageNotFoundError, version
import platform
import sys

from sklearn.pipeline import Pipeline, _final_estimator_has
from sklearn.utils.metaestimators import available_if

from .util import property_available_if


def _get_version(name):
    try:
        pkg_version = version(name)
    except ImportError:
        pkg_version = None
    return pkg_version


def show_versions():
    sys_info = {
        "Platform": platform.platform(),
        "Python version": f"{platform.python_implementation()} {platform}",
        "Python interpreter": sys.executable,
    }

    deps = [
        "scikit-survival",
        "scikit-learn",
        "numpy",
        "scipy",
        "pandas",
        "numexpr",
        "ecos",
        "osqp",
        "joblib",
        "matplotlib",
        "pytest",
        "sphinx",
        "Cython",
        "pip",
        "setuptools",
    ]
    minwidth = max(
        max(map(len, deps)),
        max(map(len, sys_info.keys())),
    )
    fmt = "{0:<%ds}: {1}" % minwidth

    print("SYSTEM")
    print("------")
    for name, version_string in sys_info.items():
        print(fmt.format(name, version_string))

    print("\nDEPENDENCIES")
    print("------------")
    for dep in deps:
        version_string = _get_version(dep)
        print(fmt.format(dep, version_string))


@available_if(_final_estimator_has("predict_cumulative_hazard_function"))
def predict_cumulative_hazard_function(self, X, **kwargs):
    """Predict cumulative hazard function.

    The cumulative hazard function for an individual
    with feature vector :math:`x` is defined as

    .. math::

        H(t \\mid x) = \\exp(x^\\top \\beta) H_0(t) ,

    where :math:`H_0(t)` is the baseline hazard function,
    estimated by Breslow's estimator.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    Returns
    -------
    cum_hazard : ndarray, shape = (n_samples,)
        Predicted cumulative hazard functions.
    """
    Xt = X
    for _, _, transform in self._iter(with_final=False):
        Xt = transform.transform(Xt)
    return self.steps[-1][-1].predict_cumulative_hazard_function(Xt, **kwargs)


@available_if(_final_estimator_has("predict_survival_function"))
def predict_survival_function(self, X, **kwargs):
    """Predict survival function.

    The survival function for an individual
    with feature vector :math:`x` is defined as

    .. math::

        S(t \\mid x) = S_0(t)^{\\exp(x^\\top \\beta)} ,

    where :math:`S_0(t)` is the baseline survival function,
    estimated by Breslow's estimator.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Data matrix.

    Returns
    -------
    survival : ndarray, shape = (n_samples,)
        Predicted survival functions.
    """
    Xt = X
    for _, _, transform in self._iter(with_final=False):
        Xt = transform.transform(Xt)
    return self.steps[-1][-1].predict_survival_function(Xt, **kwargs)


@property_available_if(_final_estimator_has("_predict_risk_score"))
def _predict_risk_score(self):
    return self.steps[-1][-1]._predict_risk_score


def patch_pipeline():
    Pipeline.predict_survival_function = predict_survival_function
    Pipeline.predict_cumulative_hazard_function = predict_cumulative_hazard_function
    Pipeline._predict_risk_score = _predict_risk_score


try:
    __version__ = version("scikit-survival")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"

patch_pipeline()
