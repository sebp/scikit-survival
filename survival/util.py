# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy
from sklearn.utils import check_consistent_length, check_array, safe_indexing


__all__ = ['check_arrays_survival']


def check_arrays_survival(X, y, force_all_finite=True):
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    X : array-like
        Data matrix containing feature vectors.

    y : structured array with two fields
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    X : array, shape=[n_samples, n_features]
        Feature vectors.

    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    if not isinstance(y, numpy.ndarray) or y.dtype.fields is None or len(y.dtype.fields) != 2:
        raise ValueError('y must be a structured array with the first field'
                         ' being a binary class event indicator and the second field'
                         ' the time of the event/censoring')

    event_field, time_field = y.dtype.names

    X = check_array(X, dtype=float, force_all_finite=force_all_finite)
    event = check_array(y[event_field], ensure_2d=False)
    if not numpy.issubdtype(event.dtype, numpy.bool_):
        raise ValueError('elements of event indicator must be boolean, but found {0}'.format(event.dtype))

    if not numpy.any(event):
        raise ValueError('all samples are censored')

    if not numpy.issubdtype(y[time_field].dtype, numpy.number):
        raise ValueError('time must be numeric, but found {0}'.format(y[time_field].dtype))

    time = check_array(y[time_field], dtype=float, ensure_2d=False)
    check_consistent_length(X, event, time)
    return X, event, time
