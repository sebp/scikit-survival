import importlib
import platform
import sys

from pkg_resources import DistributionNotFound, get_distribution
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method


def _get_version(name):
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None

    if name == "osqp":
        version = module.OSQP().version()
    else:
        version = getattr(module, "__version__", None)

    if version is None:  # pragma: no cover
        raise ImportError("Can't determine version for {}".format(
            module.__name__))
    return version


def show_versions():
    sys_info = {
        "Platform": platform.platform(),
        "Python version": "{} {}".format(
            platform.python_implementation(),
            platform.python_version()),
        "Python interpreter": sys.executable,
    }

    deps = [
        "sksurv",
        "sklearn",
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
    for name, version in sys_info.items():
        print(fmt.format(name, version))

    print("\nDEPENDENCIES")
    print("------------")
    for dep in deps:
        version = _get_version(dep)
        print(fmt.format(dep, version))


@if_delegate_has_method(delegate='_final_estimator')
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


@if_delegate_has_method(delegate='_final_estimator')
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


def patch_pipeline():
    Pipeline.predict_survival_function = predict_survival_function
    Pipeline.predict_cumulative_hazard_function = predict_cumulative_hazard_function


try:
    __version__ = get_distribution('scikit-survival').version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    __version__ = 'unknown'

patch_pipeline()
