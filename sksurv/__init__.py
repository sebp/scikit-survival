from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution('scikit-survival').version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    __version__ = 'unknown'
