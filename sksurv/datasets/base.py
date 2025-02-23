import warnings

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ..column import categorical_to_numeric, standardize
from ..io import loadarff
from ..util import safe_concat

__all__ = [
    "get_x_y",
    "load_arff_files_standardized",
    "load_aids",
    "load_bmt",
    "load_cgvhd",
    "load_breast_cancer",
    "load_flchain",
    "load_gbsg2",
    "load_whas500",
    "load_veterans_lung_cancer",
]


def _get_data_path(name):
    from importlib.resources import files

    return files(__package__) / "data" / name


def _get_x_y_survival(dataset, col_event, col_time, val_outcome, competing_risks=False):
    if col_event is None or col_time is None:
        y = None
        x_frame = dataset
    else:
        event_type = np.int64 if competing_risks else bool
        y = np.empty(dtype=[(col_event, event_type), (col_time, np.float64)], shape=dataset.shape[0])
        if competing_risks:
            y[col_event] = dataset[col_event].values
        else:
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


def get_x_y(data_frame, attr_labels, pos_label=None, survival=True, competing_risks=False):
    """Split data frame into features and labels.

    Parameters
    ----------
    data_frame : pandas.DataFrame, shape = (n_samples, n_columns)
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

    competing_risks : bool, optional, default: False
        Whether `y` refers to competing risks situation. Only used if `survival` is True

    Returns
    -------
    X : pandas.DataFrame, shape = (n_samples, n_columns - len(attr_labels))
        Data frame containing features.

    y : None or pandas.DataFrame, shape = (n_samples, len(attr_labels))
        Data frame containing columns with supervised information.
        If `survival` was `True`, then the column denoting the event
        indicator will be boolean and survival times will be float.
        If `attr_labels` contains `None`, y is set to `None`.
    """
    if survival:
        if len(attr_labels) != 2:
            raise ValueError(f"expected sequence of length two for attr_labels, but got {len(attr_labels)}")
        if pos_label is None and not competing_risks:
            raise ValueError("pos_label needs to be specified if survival=True")
        return _get_x_y_survival(data_frame, attr_labels[0], attr_labels[1], pos_label, competing_risks)

    return _get_x_y_other(data_frame, attr_labels)


def _loadarff_with_index(filename):
    dataset = loadarff(filename)
    if "index" in dataset.columns:
        if isinstance(dataset["index"].dtype, CategoricalDtype):
            # concatenating categorical index may raise TypeError
            # see https://github.com/pandas-dev/pandas/issues/14586
            dataset["index"] = dataset["index"].astype(object)
        dataset.set_index("index", inplace=True)
    return dataset


def load_arff_files_standardized(
    path_training,
    attr_labels,
    pos_label=None,
    path_testing=None,
    survival=True,
    standardize_numeric=True,
    to_numeric=True,
):
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

    survival : bool, optional, default: True
        Whether the dependent variables denote event indicator and survival/censoring time.

    standardize_numeric : bool, optional, default: True
        Whether to standardize data to zero mean and unit variance.
        See :func:`sksurv.column.standardize`.

    to_numeric : boo, optional, default: True
        Whether to convert categorical variables to numeric values.
        See :func:`sksurv.column.categorical_to_numeric`.

    Returns
    -------
    x_train : pandas.DataFrame, shape = (n_train, n_features)
        Training data.

    y_train : pandas.DataFrame, shape = (n_train, n_labels)
        Dependent variables of training data.

    x_test : None or pandas.DataFrame, shape = (n_train, n_features)
        Testing data if `path_testing` was provided.

    y_test : None or pandas.DataFrame, shape = (n_train, n_labels)
        Dependent variables of testing data if `path_testing` was provided.
    """
    dataset = _loadarff_with_index(path_training)

    x_train, y_train = get_x_y(dataset, attr_labels, pos_label, survival)

    if path_testing is not None:
        x_test, y_test = _load_arff_testing(path_testing, attr_labels, pos_label, survival)

        if len(x_train.columns.symmetric_difference(x_test.columns)) > 0:
            warnings.warn("Restricting columns to intersection between training and testing data", stacklevel=2)

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


def _load_arff_testing(path_testing, attr_labels, pos_label, survival):
    test_dataset = _loadarff_with_index(path_testing)

    has_labels = pd.Index(attr_labels).isin(test_dataset.columns).all()
    if not has_labels:
        if survival:
            attr_labels = [None, None]
        else:
            attr_labels = None
    return get_x_y(test_dataset, attr_labels, pos_label, survival)


def load_whas500():
    """Load and return the Worcester Heart Attack Study dataset

    The dataset has 500 samples and 14 features.
    The endpoint is death, which occurred for 215 patients (43.0%).

    See [1]_, [2]_ for further description.

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
    .. [1] https://web.archive.org/web/20170114043458/http://www.umass.edu/statdata/statdata/data/

    .. [2] Hosmer, D., Lemeshow, S., May, S.:
        "Applied Survival Analysis: Regression Modeling of Time to Event Data."
        John Wiley & Sons, Inc. (2008)
    """
    fn = _get_data_path("whas500.arff")
    return get_x_y(loadarff(fn), attr_labels=["fstat", "lenfol"], pos_label="1")


def load_gbsg2():
    """Load and return the German Breast Cancer Study Group 2 dataset

    The dataset has 686 samples and 8 features.
    The endpoint is recurrence free survival, which occurred for 299 patients (43.6%).

    See [1]_, [2]_ for further description.

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
    fn = _get_data_path("GBSG2.arff")
    return get_x_y(loadarff(fn), attr_labels=["cens", "time"], pos_label="1")


def load_veterans_lung_cancer():
    """Load and return data from the Veterans' Administration
    Lung Cancer Trial

    The dataset has 137 samples and 6 features.
    The endpoint is death, which occurred for 128 patients (93.4%).

    See [1]_ for further description.

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
    fn = _get_data_path("veteran.arff")
    return get_x_y(loadarff(fn), attr_labels=["Status", "Survival_in_days"], pos_label="dead")


def load_aids(endpoint="aids"):
    """Load and return the AIDS Clinical Trial dataset

    The dataset has 1,151 samples and 11 features.
    The dataset has 2 endpoints:

    1. AIDS defining event, which occurred for 96 patients (8.3%)
    2. Death, which occurred for 26 patients (2.3%)

    See [1]_, [2]_ for further description.

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
    .. [1] https://web.archive.org/web/20170114043458/http://www.umass.edu/statdata/statdata/data/

    .. [2] Hosmer, D., Lemeshow, S., May, S.:
        "Applied Survival Analysis: Regression Modeling of Time to Event Data."
        John Wiley & Sons, Inc. (2008)
    """
    labels_aids = ["censor", "time"]
    labels_death = ["censor_d", "time_d"]
    if endpoint == "aids":
        attr_labels = labels_aids
        drop_columns = labels_death
    elif endpoint == "death":
        attr_labels = labels_death
        drop_columns = labels_aids
    else:
        raise ValueError("endpoint must be 'aids' or 'death'")

    fn = _get_data_path("actg320.arff")
    x, y = get_x_y(loadarff(fn), attr_labels=attr_labels, pos_label="1")
    x.drop(drop_columns, axis=1, inplace=True)
    return x, y


def load_breast_cancer():
    """Load and return the breast cancer dataset

    The dataset has 198 samples and 80 features.
    The endpoint is the presence of distance metastases, which occurred for 51 patients (25.8%).

    See [1]_, [2]_ for further description.

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
    fn = _get_data_path("breast_cancer_GSE7390-metastasis.arff")
    return get_x_y(loadarff(fn), attr_labels=["e.tdm", "t.tdm"], pos_label="1")


def load_flchain():
    """Load and return assay of serum free light chain for 7874 subjects.

    The dataset has 7874 samples and 9 features:

        1. age: age in years
        2. sex: F=female, M=male
        3. sample.yr: the calendar year in which a blood sample was obtained
        4. kappa: serum free light chain, kappa portion
        5. lambda: serum free light chain, lambda portion
        6. flc.grp: the serum free light chain group for the subject, as used in the original analysis
        7. creatinine: serum creatinine
        8. mgus: whether the subject had been diagnosed with monoclonal gammapothy (MGUS)
        9. chapter: for those who died, a grouping of their primary cause of death by chapter headings
           of the International Code of Diseases ICD-9

    The endpoint is death, which occurred for 2169 subjects (27.5%).

    See [1]_, [2]_ for further description.

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *death*: boolean indicating whether the subject died
        or the event time is right censored.

        *futime*: total length of follow-up or time of death.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.mayocp.2012.03.009

    .. [2] Dispenzieri, A., Katzmann, J., Kyle, R., Larson, D., Therneau, T., Colby, C., Clark, R.,
           Mead, G., Kumar, S., Melton III, LJ. and Rajkumar, SV.
           Use of nonclonal serum immunoglobulin free light chains to predict overall survival in
           the general population, Mayo Clinic Proceedings 87:512-523. (2012)
    """
    fn = _get_data_path("flchain.arff")
    return get_x_y(loadarff(fn), attr_labels=["death", "futime"], pos_label="dead")


def load_bmt():
    """Load and return response to hematopoietic stem cell transplantation (HSCT) for acute leukemia patients.

    The dataset has 35 samples and 1 feature "dis" indicating the type of leukemia::

        0=ALL (Acute Lymphoblastic Leukemia)
        1=AML (Acute Myeloid Leukemia)

    The endpoint (status) is defined as

    +-------+------------------------------------+---------------------+
    | Value | Description                        | Count (%)           |
    +=======+====================================+=====================+
    | 0     | Survival (Right-censored data)     | 11 patients (31.4%) |
    +-------+------------------------------------+---------------------+
    | 1     | Transplant related mortality (TRM) | 9 events (25.7%)    |
    +-------+------------------------------------+---------------------+
    | 2     | Relapse                            | 15 events (42.8%)   |
    +-------+------------------------------------+---------------------+

    See [1]_ for further description and [2]_ for the dataset.

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *status*: Integer indicating the endpoint: 0-(survival i.e. right censored data), 1-(TRM), 2-(relapse)

        *ftime*: total length of follow-up or time of event.

    References
    ----------
    .. [1] https://doi.org/10.1038/sj.bmt.1705727
           Scrucca, L., Santucci, A. & Aversa, F.:
           "Competing risk analysis using R: an easy guide for clinicians. Bone Marrow Transplant 40, 381–387 (2007)"

    .. [2] https://luca-scr.github.io/R/bmt.csv
    """
    full_path = _get_data_path("bmt.arff")
    data = loadarff(full_path)
    data["ftime"] = data["ftime"].astype(int)
    return get_x_y(data, attr_labels=["status", "ftime"], competing_risks=True)


def load_cgvhd():
    r"""Load and return data from multicentre randomized clinical trial
    initiated for patients with a myeloid malignancy who were to
    undergo an allogeneic bone marrow transplant.

    The dataset is a 100 patient subsample of the full data set. See [2]_ for further details.

    +-------+------------+----------------------------------------------+-------------------------------------------+
    | Index | Name       | Description                                  | Encoding                                  |
    +=======+============+==============================================+===========================================+
    | 1     | dx         | Diagnosis                                    | | AML=acute myeloid leukaemia             |
    |       |            |                                              | | CML=chronic myeloid leukaemia           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 2     | tx         | Randomized treatment                         | | BM=cell harvested from the bone marrow  |
    |       |            |                                              | | PB=cell harvested from peripheral blood |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 3     | extent     | Extent of disease                            | L=limited, E=extensive                    |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 4     | agvhdgd    | Grade of acute GVHD                          |                                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 5     | age        | Age                                          | Years                                     |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 6     | survtime   | Time from date of transplant to death        | Years                                     |
    |       |            | or last follow-up                            |                                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 7     | reltime    | Time from date of transplant to relapse      | Years                                     |
    |       |            | or last follow-up                            |                                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 8     | agvhtime   | Time from date of transplant to acute GVHD   | Years                                     |
    |       |            | or last follow-up                            |                                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 9     | cgvhtime   | Time from date of transplant to chronic GVHD | Years                                     |
    |       |            | or last follow-up                            |                                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 10    | stat       | Status                                       | 1=Dead, 0=Alive                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 11    | rcens      | Relapse                                      | 1=Yes, 0=No                               |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 12    | agvh       | Acute GVHD                                   | 1=Yes, 0=No                               |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 13    | cgvh       | Chronic GVHD                                 | 1=Yes, 0=No                               |
    +-------+------------+----------------------------------------------+-------------------------------------------+
    | 14    | stnum      | patient ID                                   |                                           |
    +-------+------------+----------------------------------------------+-------------------------------------------+

    Columns 6,7 and 9 contain the time to death, relapse and CGVHD
    calculated in years (survtime, reltime, cgvhtime) and the
    respective indicator variables are in columns 10,11 and 13 (stat,
    rcens, cgvh). The earliest time that any of these events happened
    is calculated by taking the minimum of the observed times. The
    censoring variable cens is coded as 0 when no events were
    observed, 1 if CGVHD was observed as first event, 2 if a relapse
    was observed as the first event and 3 if death occurred before
    either of the events: The endpoint (status) is therefore defined as

    +-------+-------------------------------------------+-----------------+
    | Value | Description                               | Count (%)       |
    +=======+===========================================+=================+
    | 0     | Survival (Right-censored data)            | 4 patients (4%) |
    +-------+-------------------------------------------+-----------------+
    | 1     | Chronic graft versus host disease (CGVHD) | 86 events (86%) |
    +-------+-------------------------------------------+-----------------+
    | 2     | Relapse (TRM)                             | 5 events (5%)   |
    +-------+-------------------------------------------+-----------------+
    | 3     | Death                                     | 5 events (5%)   |
    +-------+-------------------------------------------+-----------------+

    The dataset has been obtained from [1]_.

    Returns
    -------
    x : pandas.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *status*: Integer indicating the endpoint: 0: right censored data; 1: CGVHD; 2: relapse; 3: death.

        *ftime*: total length of follow-up or time of event.

    References
    ----------
    .. [1] https://sites.google.com/view/melaniapintiliemscstatistics/home/statistics

    .. [2] Melania Pintilie: "Competing Risks: A Practical Perspective". John Wiley & Sons, 2006
    """
    full_path = _get_data_path("cgvhd.arff")
    data = loadarff(full_path)
    data["ftime"] = data[["survtime", "reltime", "cgvhtime"]].min(axis=1)
    data["status"] = (
        ((data["ftime"] == data["cgvhtime"]) & (data["cgvh"] == "1")).astype(int)
        + 2 * ((data["ftime"] == data["reltime"]) & (data["rcens"] == "1")).astype(int)
        + 3 * ((data["ftime"] == data["survtime"]) & (data["stat"] == "1")).astype(int)
    )
    data = data[["ftime", "status", "dx", "tx", "extent", "age"]]

    return get_x_y(data, attr_labels=["status", "ftime"], competing_risks=True)
