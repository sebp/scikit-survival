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
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.utils.validation import check_array, check_consistent_length

__all__ = ["check_array_survival", "check_y_survival", "safe_concat", "Surv"]


class Surv:
    """A helper class to create a structured array for survival analysis.

    This class provides helper functions to create a structured array that
    encapsulates the event indicator and the observed time. The resulting
    structured array is the recommended format for the ``y`` argument in
    scikit-survival's estimators.
    """

    @staticmethod
    def from_arrays(event, time, name_event=None, name_time=None):
        """Create structured array from event indicator and time arrays.

        Parameters
        ----------
        event : array-like, shape=(n_samples,)
            Event indicator. A boolean array or array with values 0/1,
            where ``True`` or 1 indicates an event and ``False`` or 0
            indicates right-censoring.
        time : array-like, shape=(n_samples,)
            Observed time. Time to event or time of censoring.
        name_event : str, optional, default: 'event'
            Name of the event field in the structured array.
        name_time : str, optional, default: 'time'
            Name of the observed time field in the structured array.

        Returns
        -------
        y : numpy.ndarray
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event and ``False`` indicates right-censoring.
            The second field is a float with the time of event or time of censoring.
            The names of the fields are set to the values of `name_event` and `name_time`.

        Examples
        --------
        >>> from sksurv.util import Surv
        >>>
        >>> y = Surv.from_arrays(event=[True, False, True],
        ...                      time=[10, 25, 15])
        >>> y
        array([( True, 10.), (False, 25.), ( True, 15.)],
            dtype=[('event', '?'), ('time', '<f8')])
        >>> y['event']
        array([ True, False,  True])
        >>> y['time']
        array([10., 25., 15.])
        """
        name_event = name_event or "event"
        name_time = name_time or "time"
        if name_time == name_event:
            raise ValueError("name_time must be different from name_event")

        time = np.asanyarray(time, dtype=float)
        y = np.empty(time.shape[0], dtype=[(name_event, bool), (name_time, float)])
        y[name_time] = time

        event = np.asanyarray(event)
        check_consistent_length(time, event)

        if np.issubdtype(event.dtype, np.bool_):
            y[name_event] = event
        else:
            events = np.unique(event)
            events.sort()
            if len(events) != 2:
                raise ValueError("event indicator must be binary")

            if np.all(events == np.array([0, 1], dtype=events.dtype)):
                y[name_event] = event.astype(bool)
            else:
                raise ValueError("non-boolean event indicator must contain 0 and 1 only")

        return y

    @staticmethod
    def from_dataframe(event, time, data):
        """Create structured array from columns in a pandas DataFrame.

        Parameters
        ----------
        event : str
            Name of the column in ``data`` containing the event indicator.
            It must be a boolean or have values 0/1,
            where ``True`` or 1 indicates an event and ``False`` or 0
            indicates right-censoring.
        time : str
            Name of the column in ``data`` containing the observed time
            (time to event or time of censoring).
        data : pandas.DataFrame
            A DataFrame with columns for event and time.

        Returns
        -------
        y : numpy.ndarray
            A structured array with two fields. The first field is a boolean
            where ``True`` indicates an event and ``False`` indicates right-censoring.
            The second field is a float with the time of event or time of censoring.
            The names of the fields are the respective column names.

        Examples
        --------
        >>> import pandas as pd
        >>> from sksurv.util import Surv
        >>>
        >>> df = pd.DataFrame({
        ...     'status': [True, False, True],
        ...     'followup_time': [10, 25, 15],
        ... })
        >>> y = Surv.from_dataframe(
        ...     event='status', time='followup_time', data=df,
        ... )
        >>> y
        array([( True, 10.), (False, 25.), ( True, 15.)],
            dtype=[('status', '?'), ('followup_time', '<f8')])
        >>> y['status']
        array([ True, False,  True])
        >>> y['followup_time']
        array([10., 25., 15.])
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"expected pandas.DataFrame, but got {type(data)!r}")

        return Surv.from_arrays(
            data.loc[:, event].values, data.loc[:, time].values, name_event=str(event), name_time=str(time)
        )


def check_y_survival(y_or_event, *args, allow_all_censored=False, allow_time_zero=True, competing_risks=False):
    """Check that array correctly represents an outcome for survival analysis.

    Parameters
    ----------
    y_or_event : structured array with two fields, or boolean array
        If a structured array, it must contain the binary event indicator
        as first field, and time of event or time of censoring as
        second field. Otherwise, it is assumed that a boolean array
        representing the event indicator is passed.
        If `competing_risks` is `True`, it should be a non-negative valued integer array,
        also all risks must appear at least once in the event array.

    *args : list of array-likes
        Any number of array-like objects representing time information.
        Elements that are `None` are passed along in the return value.

    allow_all_censored : bool, optional, default: False
        Whether to allow all events to be censored.

    allow_time_zero : bool, optional, default: True
        Whether to allow event times to be zero.

    competing_risks : bool, optional, default: False
        Whether there are multiple risks. (See y_or_event)

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    if len(args) == 0:
        y = y_or_event

        if not isinstance(y, np.ndarray) or y.dtype.fields is None or len(y.dtype.fields) != 2:
            raise ValueError(
                "y must be a structured array with the first field"
                " being a binary class event indicator and the second field"
                " the time of the event/censoring"
            )

        event_field, time_field = y.dtype.names
        y_event = y[event_field]
        time_args = (y[time_field],)
    else:
        y_event = np.asanyarray(y_or_event)
        time_args = args

    event = check_array(y_event, ensure_2d=False)
    check_event_dtype(event, competing_risks)
    if competing_risks and not np.all(np.isin(range(1, np.max(event) + 1), event)):
        raise ValueError("Some risks do not appear in the event array.")

    if not (allow_all_censored or np.any(event)):
        raise ValueError("all samples are censored")

    return_val = [event]
    for i, yt in enumerate(time_args):
        if yt is None:
            return_val.append(yt)
            continue

        yt = check_array(yt, ensure_2d=False)
        if not np.issubdtype(yt.dtype, np.number):
            raise ValueError(f"time must be numeric, but found {yt.dtype} for argument {i + 2}")

        if allow_time_zero:
            cond = yt < 0
            msg = "observed time contains values smaller zero"
        else:
            cond = yt <= 0
            msg = "observed time contains values smaller or equal to zero"
        if np.any(cond):
            raise ValueError(msg)

        return_val.append(yt)

    return tuple(return_val)


def check_event_dtype(event, competing_risks=False):
    """Check that the event array has the correct dtype.

    For single-event survival analysis, the event indicator must be a
    boolean array. For competing risk analysis, it must be an integer
    array.

    Parameters
    ----------
    event : ndarray, shape=(n_samples,), dtype=bool | int
        Array containing the event indicator.

    competing_risks : bool, optional, default: False
        Whether `event` is for a competing risks analysis.
    """
    if competing_risks:
        if not np.issubdtype(event.dtype, np.integer):
            raise ValueError(f"Elements of event indicator must be integer, but found {event.dtype}")
        if np.any(event < 0):
            raise ValueError("Elements of event indicator must be non-negative")
        return

    if not np.issubdtype(event.dtype, np.bool_):
        raise ValueError(f"elements of event indicator must be boolean, but found {event.dtype}")


def check_array_survival(X, y, **kwargs):
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    X : array-like
        Data matrix containing feature vectors.

    y : structured array with two fields
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    kwargs : dict
        Additional arguments passed to :func:`check_y_survival`.

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    event, time = check_y_survival(y, **kwargs)
    check_consistent_length(X, event, time)
    return event, time


def safe_concat(objs, *args, **kwargs):
    """Alternative to :func:`pandas.concat` that preserves categorical variables.

    Parameters
    ----------
    objs : a sequence or mapping of Series, DataFrame, or Panel objects
        If a dict is passed, the sorted keys will be used as the `keys`
        argument, unless it is passed, in which case the values will be
        selected (see below). Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised
    axis : {0, 1, ...}, default 0
        The axis to concatenate along
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis(es)
    join_axes : list of Index objects
        Specific indexes to use for the other n - 1 axes instead of performing
        inner/outer set logic
    verify_integrity : boolean, default False
        Check whether the new concatenated axis contains duplicates. This can
        be very expensive relative to the actual data concatenation
    keys : sequence, default None
        If multiple levels passed, should contain tuples. Construct
        hierarchical index using the passed keys as the outermost level
    levels : list of sequences, default None
        Specific levels (unique values) to use for constructing a
        MultiIndex. Otherwise they will be inferred from the keys
    names : list, default None
        Names for the levels in the resulting hierarchical index
    ignore_index : boolean, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note that the index values on the other
        axes are still respected in the join.
    copy : boolean, default True
        If False, do not copy data unnecessarily

    Notes
    -----
    The keys, levels, and names arguments are all optional

    Returns
    -------
    concatenated : type of objects
    """
    axis = kwargs.pop("axis", 0)
    categories = {}
    for df in objs:
        if isinstance(df, pd.Series):
            if isinstance(df.dtype, CategoricalDtype):
                categories[df.name] = {"categories": df.cat.categories, "ordered": df.cat.ordered}
        else:
            dfc = df.select_dtypes(include=["category"])
            new_dtypes = {}
            for name, s in dfc.items():
                if name in categories:
                    if axis == 1:
                        raise ValueError(f"duplicate columns {name}")
                    if not categories[name]["categories"].equals(s.cat.categories):
                        raise ValueError(f"categories for column {name} do not match")
                else:
                    categories[name] = {"categories": s.cat.categories, "ordered": s.cat.ordered}
                new_dtypes[name] = "str"
            df = df.astype(new_dtypes)

    concatenated = pd.concat(objs, *args, axis=axis, **kwargs)

    concatenated = concatenated.astype({name: pd.CategoricalDtype(**params) for name, params in categories.items()})

    return concatenated


class _PropertyAvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol based on the property decorator.

    The corresponding class in scikit-learn (`_AvailableIfDescriptor`) only supports callables.
    This class adopts the property decorator as described in the descriptor guide in the offical Python documentation.

    See also
    --------
    https://docs.python.org/3/howto/descriptor.html
        Descriptor HowTo Guide

    :class:`sklearn.utils.available_if._AvailableIfDescriptor`
        The original class in scikit-learn.
    """

    def __init__(self, check, fget, doc=None):
        self.check = check
        self.fget = fget
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc
        self._name = ""

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        attr_err = AttributeError(f"This {obj!r} has no attribute {self._name!r}")
        if not self.check(obj):
            raise attr_err

        if self.fget is None:
            raise AttributeError(f"property '{self._name}' has no getter")
        return self.fget(obj)


def property_available_if(check):
    """A property attribute that is available only if check returns a truthy value.

    Only supports getting an attribute value, setting or deleting an attribute value are not supported.

    Parameters
    ----------
    check : callable
        When passed the object of the decorated method, this should return
        `True` if the property attribute is available, and either return `False`
        or raise an `AttributeError` if not available.

    Returns
    -------
    callable
        Callable makes the decorated property available if `check` returns
        `True`, otherwise the decorated property is unavailable.
    """
    return lambda fn: _PropertyAvailableIfDescriptor(check=check, fget=fn)
