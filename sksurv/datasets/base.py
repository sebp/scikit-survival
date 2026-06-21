import warnings

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .._dataframe import (
    ensure_eager_dataframe,
    is_supported_dataframe,
    unsupported_dataframe_error,
)
from ..column import categorical_to_numeric, standardize
from ..io import loadarff

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
        return dataset, None

    nw_data = nw.from_native(dataset)

    event_type = np.int64 if competing_risks else bool
    n_samples = nw_data.shape[0]
    y = np.empty(dtype=[(col_event, event_type), (col_time, np.float64)], shape=n_samples)
    event_arr = nw_data.get_column(col_event).to_numpy()
    time_arr = nw_data.get_column(col_time).to_numpy()
    if competing_risks:
        y[col_event] = event_arr
    else:
        y[col_event] = event_arr == val_outcome
    y[col_time] = time_arr

    x_frame = nw_data.drop([col_event, col_time]).to_native()
    return x_frame, y


def _get_x_y_other(dataset, col_label):
    if col_label is None:
        return dataset, None

    nw_data = nw.from_native(dataset)

    if isinstance(col_label, str):
        # A single label name yields a Series in the input dataframe library;
        # a sequence of label names yields a DataFrame.
        y = nw_data.get_column(col_label).to_native()
        x_frame = nw_data.drop([col_label]).to_native()
    else:
        col_label_list = list(col_label)
        y = nw_data.select(col_label_list).to_native()
        x_frame = nw_data.drop(col_label_list).to_native()
    return x_frame, y


def get_x_y(data_frame, attr_labels, pos_label=None, survival=True, competing_risks=False):
    """Split data frame into features and labels.

    Parameters
    ----------
    data_frame : pandas.DataFrame or polars.DataFrame, shape = (n_samples, n_columns)
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
        Whether `y` refers to competing risks situation. Only used if `survival` is `True`.

    Returns
    -------
    X : pandas.DataFrame or polars.DataFrame, shape = (n_samples, n_features)
        Data frame containing features. The output dataframe library matches the input.

    y : structured array, Series, DataFrame, or None
        If `survival` is `True`, a structured array of shape ``(n_samples,)`` with
        two fields.
        The first field is a boolean where ``True`` indicates an event and ``False``
        indicates right-censoring. The second field is a float with the time of
        event or time of censoring.

        If `survival` is `False` and `attr_labels` is a single column name, a Series
        in the input dataframe library; if it is a sequence of column names, a
        DataFrame with those columns.

        If `survival` is `False` and `attr_labels` is `None`, `y` is set to `None`.
    """
    # Reject a polars LazyFrame first so it gets the actionable
    # "call .collect()" error instead of the generic unsupported-type one.
    data_frame = ensure_eager_dataframe(data_frame)
    if not is_supported_dataframe(data_frame):
        raise unsupported_dataframe_error(data_frame)

    if survival:
        if len(attr_labels) != 2:
            raise ValueError(f"expected sequence of length two for attr_labels, but got {len(attr_labels)}")
        if pos_label is None and not competing_risks:
            raise ValueError("pos_label needs to be specified if survival=True")
        return _get_x_y_survival(data_frame, attr_labels[0], attr_labels[1], pos_label, competing_risks)

    return _get_x_y_other(data_frame, attr_labels)


def _loadarff_with_index(filename, output_type="pandas"):
    dataset = loadarff(filename, output_type=output_type)
    if output_type == "polars":
        # polars has no row-index; drop the ARFF "index" column so the column
        # set matches the pandas branch (which absorbs it as the row index).
        if "index" in dataset.columns:
            dataset = nw.from_native(dataset).drop("index").to_native()
        return dataset
    if "index" in dataset.columns:
        if isinstance(dataset["index"].dtype, CategoricalDtype):
            # concatenating categorical index may raise TypeError
            # see https://github.com/pandas-dev/pandas/issues/14586
            dataset = dataset.astype({"index": "str"})
        dataset.set_index("index", inplace=True)
    return dataset


def _concat_rows(frames):
    frames = list(frames)
    if not frames:
        raise ValueError("No objects to concatenate")
    if nw.dependencies.is_pandas_dataframe(frames[0]):
        return pd.concat(frames, axis=0)
    return nw.concat([nw.from_native(frame) for frame in frames], how="vertical").to_native()


def load_arff_files_standardized(
    path_training,
    attr_labels,
    pos_label=None,
    path_testing=None,
    survival=True,
    standardize_numeric=True,
    to_numeric=True,
    *,
    output_type="pandas",
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

    to_numeric : bool, optional, default: True
        Whether to convert categorical variables to numeric values.
        See :func:`sksurv.column.categorical_to_numeric`.

    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned ``x_train`` / ``x_test``.
        All derivations (concatenation, standardization, numeric conversion)
        run through narwhals in the requested dataframe library; there is no
        intermediate pandas conversion when ``output_type="polars"``.

    Returns
    -------
    x_train : pandas.DataFrame or polars.DataFrame, shape = (n_train, n_features)
        Training data.

    y_train : structured array, Series, DataFrame, or None
        Dependent variables of training data.

        If `survival` is `True`, a structured array with two fields.
        The first field is a boolean where ``True`` indicates an event and ``False``
        indicates right-censoring. The second field is a float with the time of
        event or time of censoring.

        If `survival` is `False` and `attr_labels` is a single column name, a Series
        in the ``output_type`` dataframe library; if it is a sequence of column names,
        a DataFrame with those columns.

        If `survival` is `False` and `attr_labels` is `None`, `y_train` is set to `None`.

    x_test : None, or pandas.DataFrame / polars.DataFrame of shape (n_test, n_features)
        Testing data if `path_testing` was provided. Dataframe library matches ``output_type``.

    y_test : structured array, Series, DataFrame, or None
        Dependent variables of testing data if `path_testing` was provided.

        If `survival` is `True`, a structured array with two fields.
        The first field is a boolean where ``True`` indicates an event and ``False``
        indicates right-censoring. The second field is a float with the time of
        event or time of censoring.

        If `survival` is `False` and `attr_labels` is a single column name, a Series
        in the ``output_type`` dataframe library; if it is a sequence of column names,
        a DataFrame with those columns.

        If `survival` is `False` and `attr_labels` is `None`, `y_test` is set to `None`.
    """
    dataset = _loadarff_with_index(path_training, output_type=output_type)

    x_train, y_train = get_x_y(dataset, attr_labels, pos_label, survival)

    if path_testing is not None:
        x_test, y_test = _load_arff_testing(path_testing, attr_labels, pos_label, survival, output_type=output_type)

        train_cols = nw.from_native(x_train).columns
        test_cols = nw.from_native(x_test).columns
        train_col_set = set(train_cols)
        test_col_set = set(test_cols)
        if train_col_set != test_col_set:
            warnings.warn("Restricting columns to intersection between training and testing data", stacklevel=2)

            cols = [c for c in train_cols if c in test_col_set]
            if not cols:
                raise ValueError("columns of training and test data do not intersect")

            x_train = nw.from_native(x_train).select(cols).to_native()
            x_test = nw.from_native(x_test).select(cols).to_native()

        x = _concat_rows((x_train, x_test))
        if standardize_numeric:
            x = standardize(x)
        if to_numeric:
            x = categorical_to_numeric(x)

        n_train = nw.from_native(x_train).shape[0]
        nw_x = nw.from_native(x)
        x_train = nw_x.head(n_train).to_native()
        x_test = nw_x.tail(nw_x.shape[0] - n_train).to_native()
    else:
        if standardize_numeric:
            x_train = standardize(x_train)
        if to_numeric:
            x_train = categorical_to_numeric(x_train)

        x_test = None
        y_test = None

    return x_train, y_train, x_test, y_test


def _load_arff_testing(path_testing, attr_labels, pos_label, survival, output_type="pandas"):
    test_dataset = _loadarff_with_index(path_testing, output_type=output_type)

    test_columns = set(nw.from_native(test_dataset).columns)
    has_labels = attr_labels is not None and all(lbl in test_columns for lbl in attr_labels)
    if not has_labels:
        if survival:
            attr_labels = [None, None]
        else:
            attr_labels = None
    return get_x_y(test_dataset, attr_labels, pos_label, survival)


def load_whas500(*, output_type="pandas"):
    """Load and return the Worcester Heart Attack Study dataset

    The dataset has 500 samples and 14 features.
    The endpoint is death, which occurred for 215 patients (43.0%).

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *fstat*: boolean indicating whether the endpoint has been reached
        or the event time is right-censored.

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
    return get_x_y(loadarff(fn, output_type=output_type), attr_labels=["fstat", "lenfol"], pos_label="1")


def load_gbsg2(*, output_type="pandas"):
    """Load and return the German Breast Cancer Study Group 2 dataset

    The dataset has 686 samples and 8 features.
    The endpoint is recurrence free survival, which occurred for 299 patients (43.6%).

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *cens*: boolean indicating whether the endpoint has been reached
        or the event time is right-censored.

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
    return get_x_y(loadarff(fn, output_type=output_type), attr_labels=["cens", "time"], pos_label="1")


def load_veterans_lung_cancer(*, output_type="pandas"):
    """Load and return data from the Veterans' Administration
    Lung Cancer Trial

    The dataset has 137 samples and 6 features.
    The endpoint is death, which occurred for 128 patients (93.4%).

    See [1]_ for further description.

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *Status*: boolean indicating whether the endpoint has been reached
        or the event time is right-censored.

        *Survival_in_days*: total length of follow-up

    References
    ----------
    .. [1] Kalbfleisch, J.D., Prentice, R.L.:
        "The Statistical Analysis of Failure Time Data." John Wiley & Sons, Inc. (2002)
    """
    fn = _get_data_path("veteran.arff")
    return get_x_y(
        loadarff(fn, output_type=output_type),
        attr_labels=["Status", "Survival_in_days"],
        pos_label="dead",
    )


def load_aids(endpoint="aids", *, output_type="pandas"):
    """Load and return the AIDS Clinical Trial dataset

    The dataset has 1,151 samples and 11 features.
    The dataset has 2 endpoints:

    1. AIDS defining event, which occurred for 96 patients (8.3%)
    2. Death, which occurred for 26 patients (2.3%)

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    endpoint : {'aids', 'death'}, default: 'aids'
        The endpoint.
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *censor*: boolean indicating whether the endpoint has been reached
        or the event time is right-censored.

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
    x, y = get_x_y(loadarff(fn, output_type=output_type), attr_labels=attr_labels, pos_label="1")
    x = nw.from_native(x).drop(drop_columns).to_native()
    return x, y


def load_breast_cancer(*, output_type="pandas"):
    """Load and return the breast cancer dataset

    The dataset has 198 samples and 80 features.
    The endpoint is the presence of distance metastases, which occurred for 51 patients (25.8%).

    See [1]_, [2]_ for further description.

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *e.tdm*: boolean indicating whether the endpoint has been reached
        or the event time is right-censored.

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
    return get_x_y(
        loadarff(fn, output_type=output_type),
        attr_labels=["e.tdm", "t.tdm"],
        pos_label="1",
    )


def load_flchain(*, output_type="pandas"):
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

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *death*: boolean indicating whether the subject died
        or the event time is right-censored.

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
    return get_x_y(
        loadarff(fn, output_type=output_type),
        attr_labels=["death", "futime"],
        pos_label="dead",
    )


def load_bmt(*, output_type="pandas"):
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

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *status*: Integer indicating the endpoint: 0-(survival i.e. right-censored data), 1-(TRM), 2-(relapse)

        *ftime*: total length of follow-up or time of event.

    References
    ----------
    .. [1] https://doi.org/10.1038/sj.bmt.1705727
           Scrucca, L., Santucci, A. & Aversa, F.:
           "Competing risk analysis using R: an easy guide for clinicians. Bone Marrow Transplant 40, 381–387 (2007)"

    .. [2] https://luca-scr.github.io/R/bmt.csv
    """
    full_path = _get_data_path("bmt.arff")
    data = loadarff(full_path, output_type=output_type)
    data = nw.from_native(data).with_columns(nw.col("ftime").cast(nw.Int64)).to_native()
    return get_x_y(data, attr_labels=["status", "ftime"], competing_risks=True)


def load_cgvhd(*, output_type="pandas"):
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

    Parameters
    ----------
    output_type : {"pandas", "polars"}, default="pandas"
        Dataframe library used for the returned features. The derivations
        (horizontal min over the time columns, the boolean-sum encoding of
        the status column) run through narwhals in the requested dataframe library;
        no intermediate pandas conversion is performed when
        ``output_type="polars"``.

    Returns
    -------
    x : pandas.DataFrame or polars.DataFrame
        The measurements for each patient.

    y : structured array with 2 fields
        *status*: Integer indicating the endpoint: 0: right-censored data; 1: CGVHD; 2: relapse; 3: death.

        *ftime*: total length of follow-up or time of event.

    References
    ----------
    .. [1] https://sites.google.com/view/melaniapintiliemscstatistics/home/statistics

    .. [2] Melania Pintilie: "Competing Risks: A Practical Perspective". John Wiley & Sons, 2006
    """
    full_path = _get_data_path("cgvhd.arff")
    data = loadarff(full_path, output_type=output_type)
    nw_data = nw.from_native(data)
    nw_data = nw_data.with_columns(
        nw.min_horizontal("survtime", "reltime", "cgvhtime").alias("ftime"),
    )
    nw_data = nw_data.with_columns(
        (
            ((nw.col("ftime") == nw.col("cgvhtime")) & (nw.col("cgvh") == "1")).cast(nw.Int64)
            + 2 * ((nw.col("ftime") == nw.col("reltime")) & (nw.col("rcens") == "1")).cast(nw.Int64)
            + 3 * ((nw.col("ftime") == nw.col("survtime")) & (nw.col("stat") == "1")).cast(nw.Int64)
        ).alias("status"),
    )
    data = nw_data.select(["ftime", "status", "dx", "tx", "extent", "age"]).to_native()
    return get_x_y(data, attr_labels=["status", "ftime"], competing_risks=True)
