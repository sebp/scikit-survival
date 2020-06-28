API reference
=============

Datasets
--------
.. currentmodule:: sksurv.datasets

.. autosummary::
    :toctree: generated/

    get_x_y
    load_aids
    load_arff_files_standardized
    load_breast_cancer
    load_flchain
    load_gbsg2
    load_whas500
    load_veterans_lung_cancer


Ensemble Models
---------------
.. currentmodule:: sksurv.ensemble

.. autosummary::
    :toctree: generated/

    ComponentwiseGradientBoostingSurvivalAnalysis
    GradientBoostingSurvivalAnalysis
    RandomSurvivalForest


Functions
---------
.. currentmodule:: sksurv.functions

.. autosummary::
    :toctree: generated/

    StepFunction


Hypothesis testing
------------------
.. currentmodule:: sksurv.compare

.. autosummary::
    :toctree: generated/

    compare_survival


I/O Utilities
-------------
.. currentmodule:: sksurv.io

.. autosummary::
    :toctree: generated/

    loadarff
    writearff


Kernels
-------
.. currentmodule:: sksurv.kernels

.. autosummary::
    :toctree: generated/

    ClinicalKernelTransform
    clinical_kernel


Linear Models
-------------
.. currentmodule:: sksurv.linear_model

.. autosummary::
    :toctree: generated/

    CoxnetSurvivalAnalysis
    CoxPHSurvivalAnalysis
    IPCRidge


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

    brier_score
    concordance_index_censored
    concordance_index_ipcw
    cumulative_dynamic_auc
    integrated_brier_score


Non-parametric Estimators
-------------------------
.. currentmodule:: sksurv.nonparametric

.. autosummary::
    :toctree: generated/

    CensoringDistributionEstimator
    SurvivalFunctionEstimator
    ipc_weights
    kaplan_meier_estimator
    nelson_aalen_estimator


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


Survival Support Vector Machine
-------------------------------
.. currentmodule:: sksurv.svm

.. autosummary::
    :toctree: generated/

    HingeLossSurvivalSVM
    FastKernelSurvivalSVM
    FastSurvivalSVM
    MinlipSurvivalAnalysis
    NaiveSurvivalSVM


Survival Trees
--------------
.. currentmodule:: sksurv.tree

.. autosummary::
    :toctree: generated/

    SurvivalTree


Utilities
---------
.. currentmodule:: sksurv.util

.. autosummary::
    :toctree: generated/

    Surv
