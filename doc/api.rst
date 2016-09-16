API reference
=============

Survival Support Vector Machine
-------------------------------
.. currentmodule:: survival.svm

.. autosummary::
    :toctree: generated/

    FastKernelSurvivalSVM
    FastSurvivalSVM
    MinlipSurvivalAnalysis
    NaiveSurvivalSVM


Kernels
-------
.. currentmodule:: survival.kernels

.. autosummary::
    :toctree: generated/

    clinical_kernel
    ClinicalKernelTransform


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
