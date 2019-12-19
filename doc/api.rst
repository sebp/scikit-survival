API reference
=============

Non-parametric Estimators
-------------------------
.. currentmodule:: sksurv.nonparametric

.. autosummary::
    :toctree: generated/

    CensoringDistributionEstimator
    SurvivalFunctionEstimator
    kaplan_meier_estimator
    nelson_aalen_estimator
    ipc_weights


Hypothesis testing
------------------
.. currentmodule:: sksurv.compare

.. autosummary::
    :toctree: generated/

    compare_survival


Linear Models
-------------
.. currentmodule:: sksurv.linear_model

.. autosummary::
    :toctree: generated/

    CoxnetSurvivalAnalysis
    CoxPHSurvivalAnalysis
    IPCRidge


Ensemble Models
---------------
.. currentmodule:: sksurv.ensemble

.. autosummary::
    :toctree: generated/

    ComponentwiseGradientBoostingSurvivalAnalysis
    GradientBoostingSurvivalAnalysis
    RandomSurvivalForest


Survival Support Vector Machine
-------------------------------
.. currentmodule:: sksurv.svm

.. autosummary::
    :toctree: generated/

    FastKernelSurvivalSVM
    FastSurvivalSVM
    MinlipSurvivalAnalysis
    HingeLossSurvivalSVM
    NaiveSurvivalSVM


Kernels
-------
.. currentmodule:: sksurv.kernels

.. autosummary::
    :toctree: generated/

    clinical_kernel
    ClinicalKernelTransform


Survival Trees
--------------
.. currentmodule:: sksurv.tree

.. autosummary::
    :toctree: generated/

    SurvivalTree


Meta Models
-----------
.. currentmodule:: sksurv.meta

.. autosummary::
    :toctree: generated/

    EnsembleSelection
    EnsembleSelectionRegressor
    Stacking


Metrics
-------
.. currentmodule:: sksurv.metrics

.. autosummary::
    :toctree: generated/

    concordance_index_censored
    concordance_index_ipcw
    cumulative_dynamic_auc


Pre-Processing
--------------
.. currentmodule:: sksurv.preprocessing

.. autosummary::
    :toctree: generated/

    OneHotEncoder

.. currentmodule:: sksurv.column

.. autosummary::
    :toctree: generated/

    categorical_to_numeric
    encode_categorical
    standardize


I/O Utilities
-------------
.. currentmodule:: sksurv.io

.. autosummary::
    :toctree: generated/

    loadarff
    writearff


Datasets
--------
.. currentmodule:: sksurv.datasets

.. autosummary::
    :toctree: generated/

    get_x_y
    load_arff_files_standardized
    load_aids
    load_breast_cancer
    load_flchain
    load_gbsg2
    load_whas500
    load_veterans_lung_cancer


Utilities
---------
.. currentmodule:: sksurv.util

.. autosummary::
    :toctree: generated/

    Surv
