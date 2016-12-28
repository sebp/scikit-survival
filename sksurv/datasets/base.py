from pkg_resources import resource_filename
import warnings

import numpy
import pandas

from ..column import standardize, categorical_to_numeric
from ..io import loadarff
from ..util import safe_concat

__all__ = ["get_x_y",
           "load_arff_files_standardized",
           "load_aids",
           "load_breast_cancer",
           "load_gbsg2",
           "load_whas500",
           "load_veterans_lung_cancer"]


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


def load_arff_files_standardized(path_training, attr_labels, pos_label=None, path_testing=None, survival=True,
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
        See :func:`sksurv.column.standardize`.

    to_numeric : boo, optional, default=True
        Whether to convert categorical variables to numeric values.
        See :func:`sksurv.column.categorical_to_numeric`.

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


def load_whas500():
    """Load and return the Worcester Heart Attack Study dataset

    The dataset has 500 samples and 14 features.
    The endpoint is death, which occurred for 215 patients (43.0%).

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *fstat*: boolean indicating whether the endpoint has been reached
        or the event time is right censored.

        *lenfol*: total length of follow-up (days from hospital admission date
        to date of last follow-up)

    References
    ----------
    .. [1] http://www.umass.edu/statdata/statdata/data/

    .. [2] Hosmer, D., Lemeshow, S., May, S.:
        "Applied Survival Analysis: Regression Modeling of Time to Event Data."
        John Wiley & Sons, Inc. (2008)
    """
    fn = resource_filename(__name__, 'data/whas500.arff')
    return get_x_y(loadarff(fn), attr_labels=['fstat', 'lenfol'], pos_label='1')


def load_gbsg2():
    """Load and return the German Breast Cancer Study Group 2 dataset

    The dataset has 686 samples and 8 features.
    The endpoint is recurrence free survival, which occurred for 299 patients (43.6%).

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *cens*: boolean indicating whether the endpoint has been reached
        or the event time is right censored.

        *time*: total length of follow-up

    References
    ----------
    .. [1] http://ascopubs.org/doi/abs/10.1200/jco.1994.12.10.2086

    .. [2] Schumacher, M., Basert, G., Bojar, H., et al.
        "Randomized 2 × 2 trial evaluating hormonal treatment and the duration of chemotherapy
        in node-positive breast cancer patients."
        Journal of Clinical Oncology 12, 2086–2093. (1994)
    """
    fn = resource_filename(__name__, 'data/GBSG2.arff')
    return get_x_y(loadarff(fn), attr_labels=['cens', 'time'], pos_label='1')


def load_veterans_lung_cancer():
    """Load and return the Worcester Heart Attack Study dataset

    The dataset has 137 samples and 6 features.
    The endpoint is death, which occurred for 128 patients (93.4%).

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *Status*: boolean indicating whether the endpoint has been reached
        or the event time is right censored.

        *Survival_in_days*: total length of follow-up

    References
    ----------
    .. [1] Kalbfleisch, J.D., Prentice, R.L.:
        "The Statistical Analysis of Failure Time Data." John Wiley & Sons, Inc. (2002)
    """
    fn = resource_filename(__name__, 'data/veteran.arff')
    return get_x_y(loadarff(fn), attr_labels=['Status', 'Survival_in_days'], pos_label="'dead'")


def load_aids(endpoint="aids"):
    """Load and return the AIDS Clinical Trial dataset

    The dataset has 1,151 samples and 11 features.
    The dataset has 2 endpoints:

    1. AIDS defining event, which occurred for 96 patients (8.3%)
    2. Death, which occurred for 26 patients (2.3%)

    Parameters
    ----------
    endpoint : aids|death
        The endpoint

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *censor*: boolean indicating whether the endpoint has been reached
        or the event time is right censored.

        *time*: total length of follow-up

        If ``endpoint`` is death, the fields are named *censor_d* and *time_d*.

    References
    ----------
    .. [1] http://www.umass.edu/statdata/statdata/data/

    .. [2] Hosmer, D., Lemeshow, S., May, S.:
        "Applied Survival Analysis: Regression Modeling of Time to Event Data."
        John Wiley & Sons, Inc. (2008)
    """
    labels_aids = ['censor', 'time']
    labels_death = ['censor_d', 'time_d']
    if endpoint == "aids":
        attr_labels = labels_aids
        drop_columns = labels_death
    elif endpoint == "death":
        attr_labels = labels_death
        drop_columns = labels_aids
    else:
        raise ValueError("endpoint must be 'aids' or 'death'")

    fn = resource_filename(__name__, 'data/actg320.arff')
    x, y = get_x_y(loadarff(fn), attr_labels=attr_labels, pos_label='1')
    x.drop(drop_columns, axis=1, inplace=True)
    return x, y


def load_breast_cancer():
    """Load and return the breast cancer dataset

    The dataset has 198 samples and 80 features.
    The endpoint is the presence of distance metastases, which occurred for 51 patients (25.8%).

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *e.tdm*: boolean indicating whether the endpoint has been reached
        or the event time is right censored.

        *t.tdm*: time to distant metastasis (days)

    References
    ----------
    .. [1] https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7390

    .. [2] Desmedt, C., Piette, F., Loi et al.:
        "Strong Time Dependence of the 76-Gene Prognostic Signature for Node-Negative Breast Cancer
        Patients in the TRANSBIG Multicenter Independent Validation Series."
        Clin. Cancer Res. 13(11), 3207–14 (2007)
    """
    fn = resource_filename(__name__, 'data/breast_cancer_GSE7390-metastasis.arff')
    return get_x_y(loadarff(fn), attr_labels=['e.tdm', 't.tdm'], pos_label="'1'")
