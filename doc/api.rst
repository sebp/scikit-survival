API reference
=============

Linear Models
-------------
.. currentmodule:: survival.linear_model

.. autosummary::
    :toctree: generated/

    CoxPHSurvivalAnalysis
    IPCRidge


Ensemble Models
---------------
.. currentmodule:: survival.ensemble

.. autosummary::
    :toctree: generated/

    ComponentwiseGradientBoostingSurvivalAnalysis
    GradientBoostingSurvivalAnalysis


Survival Support Vector Machine
-------------------------------
.. currentmodule:: survival.svm

.. autosummary::
    :toctree: generated/

    FastKernelSurvivalSVM
    FastSurvivalSVM
    MinlipSurvivalAnalysis
    HingeLossSurvivalSVM
    NaiveSurvivalSVM


Kernels
-------
.. currentmodule:: survival.kernels

.. autosummary::
    :toctree: generated/

    clinical_kernel
    ClinicalKernelTransform


Meta Models
-----------
.. currentmodule:: survival.meta

.. autosummary::
    :toctree: generated/

    EnsembleSelection
    EnsembleSelectionRegressor
    Stacking


Metrics
-------
.. currentmodule:: survival.metrics

.. autosummary::
    :toctree: generated/

    concordance_index_censored


Pre-Processing
--------------
.. currentmodule:: survival.column

.. autosummary::
    :toctree: generated/

    categorical_to_numeric
    encode_categorical
    standardize


I/O Utilities
-------------
.. currentmodule:: survival.io

.. autosummary::
    :toctree: generated/

    loadarff
    writearff
