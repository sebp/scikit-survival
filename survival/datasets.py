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
import warnings

import numpy
import pandas

from .io import loadarff
from .column import standardize, categorical_to_numeric
from .util import safe_concat

__all__ = ["get_x_y", "load_arff_file"]


def _get_x_y_survival(dataset, col_event, col_time, val_outcome):
    if col_event is None or col_time is None:
        y = None
        x_frame = dataset
    else:
        y = numpy.empty(dtype=[(col_event, numpy.bool), (col_time, numpy.float64)],
                        shape=dataset.shape[0])
        y[col_event] = (dataset[col_event] == val_outcome).values
        y[col_time] = dataset[col_time].values

        x_frame = dataset.drop([col_event, col_time], axis=1)

    return x_frame, y


def _get_x_y_other(dataset, col_label):
    if col_label is None:
        y = None
        x_frame = dataset
    else:
        y = dataset.loc[:, col_label]
        x_frame = dataset.drop(col_label, axis=1)

    return x_frame, y


def get_x_y(data_frame, attr_labels, pos_label=None, survival=True):
    """Split data frame into features and labels.

    Parameters
    ----------
    data_frame : pandas.DataFrame, shape = [n_samples, n_columns]
        A data frame.

    attr_labels : sequence of str or None
        A list of one or more columns that are considered the label.
        If `survival` is `True`, then attr_labels has two elements:
        1) the name of the column denoting the event indicator, and
        2) the name of the column denoting the survival time.
        If the sequence contains `None`, then labels are not retrieved
        and only a data frame with features is returned.

    pos_label : any, optional
        Which value of the event indicator column denotes that a
        patient experienced an event. This value is ignored if
        `survival` is `False`.

    survival : bool, optional, default: True
        Whether to return `y` that can be used for survival analysis.

    Returns
    -------
    X : pandas.DataFrame, shape = [n_samples, n_columns - len(attr_labels)]
        Data frame containing features.

    y : None or pandas.DataFrame, shape = [n_samples, len(attr_labels])
        Data frame containing columns with supervised information.
        If `survival` was `True`, then the column denoting the event
        indicator will be boolean and survival times will be float.
        If `attr_labels` contains `None`, y is set to `None`.
    """
    if survival:
        if len(attr_labels) != 2:
            raise ValueError("expected sequence of length two for attr_labels, but got %d" % len(attr_labels))
        if pos_label is None:
            raise ValueError("pos_label needs to be specified if survival=True")
        return _get_x_y_survival(data_frame, attr_labels[0], attr_labels[1], pos_label)

    return _get_x_y_other(data_frame, attr_labels)


def load_arff_file(path_training, attr_labels, pos_label=None, path_testing=None, survival=True,
                   standardize_numeric=True, to_numeric=True):
    """Load dataset in ARFF format.

    Parameters
    ----------
    path_training : str
        Path to ARFF file containing data.

    attr_labels : sequence of str
        Names of attributes denoting dependent variables.
        If ``survival`` is set, it must be a sequence with two items:
        the name of the event indicator and the name of the survival/censoring time.

    pos_label : any type, optional
        Value corresponding to an event in survival analysis.
        Only considered if ``survival`` is ``True``.

    path_testing : str, optional
        Path to ARFF file containing hold-out data. Only columns that are available in both
        training and testing are considered (excluding dependent variables).
        If ``standardize_numeric`` is set, data is normalized by considering both training
        and testing data.

    survival : bool, optional, default=True
        Whether the dependent variables denote event indicator and survival/censoring time.

    standardize_numeric : bool, optional, default=True
        Whether to standardize data to zero mean and unit variance.
        See :func:`survival.column.standardize`.

    to_numeric : boo, optional, default=True
        Whether to convert categorical variables to numeric values.
        See :func:`survival.column.categorical_to_numeric`.

    Returns
    -------
    x_train : pandas.DataFrame, shape = [n_train, n_features]
        Training data.

    y_train : pandas.DataFrame, shape = [n_train, n_labels]
        Dependent variables of training data.

    x_test : None or pandas.DataFrame, shape = [n_train, n_features]
        Testing data if `path_testing` was provided.

    y_test : None or pandas.DataFrame, shape = [n_train, n_labels]
        Dependent variables of testing data if `path_testing` was provided.
    """
    dataset = loadarff(path_training)
    if "index" in dataset.columns:
        dataset.index = dataset["index"].astype(object)
        dataset.drop("index", axis=1, inplace=True)

    x_train, y_train = get_x_y(dataset, attr_labels, pos_label, survival)

    if path_testing is not None:
        test_dataset = loadarff(path_testing)
        if "index" in test_dataset.columns:
            test_dataset.index = test_dataset["index"].astype(object)
            test_dataset.drop("index", axis=1, inplace=True)

        has_labels = pandas.Index(attr_labels).isin(test_dataset.columns).all()
        if not has_labels:
            if survival:
                attr_labels = [None, None]
            else:
                attr_labels = None
        x_test, y_test = get_x_y(test_dataset, attr_labels, pos_label, survival)

        if len(x_train.columns.symmetric_difference(x_test.columns)) > 0:
            warnings.warn("Restricting columns to intersection between training and testing data",
                          stacklevel=2)

            cols = x_train.columns.intersection(x_test.columns)
            if len(cols) == 0:
                raise ValueError("columns of training and test data do not intersect")

            x_train = x_train.loc[:, cols]
            x_test = x_test.loc[:, cols]

        x = safe_concat((x_train, x_test), axis=0)
        if standardize_numeric:
            x = standardize(x)
        if to_numeric:
            x = categorical_to_numeric(x)

        n_train = x_train.shape[0]
        x_train = x.iloc[:n_train, :]
        x_test = x.iloc[n_train:, :]
    else:
        if standardize_numeric:
            x_train = standardize(x_train)
        if to_numeric:
            x_train = categorical_to_numeric(x_train)

        x_test = None
        y_test = None

    return x_train, y_train, x_test, y_test
