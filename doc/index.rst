.. scikit-survival documentation master file

scikit-survival
===============

scikit-survival is a Python module for `survival analysis <https://en.wikipedia.org/wiki/Survival_analysis>`_
built on top of `scikit-learn <http://scikit-learn.org/>`_. It allows doing survival analysis
while utilizing the power of scikit-learn, e.g., for pre-processing or doing cross-validation.

The objective in survival analysis (also referred to as time-to-event or reliability analysis)
is to establish a connection between covariates and the time of an event.
What makes survival analysis differ from traditional machine learning is the fact that
parts of the training data can only be partially observed – they are *censored*.

For instance, in a clinical study, patients are often monitored for a particular time period,
and events occurring in this particular period are recorded.
If a patient experiences an event, the exact time of the event can
be recorded – the patient’s record is uncensored. In contrast, right censored records
refer to patients that remained event-free during the study period and
it is unknown whether an event has or has not occurred after the study ended.
Consequently, survival analysis demands for models that take
this unique characteristic of such a dataset into account.

Installation
------------

The easiest way to install scikit-survival is to use
`Anaconda <https://www.anaconda.com/distribution/>`_ by running::

  conda install -c sebp scikit-survival

Alternatively, you can install scikit-survival from source
following :doc:`this guide <install>`.


Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   understanding_predictions
   api
   contributing
   release_notes


Notebooks
---------

.. toctree::
   :maxdepth: 1

   examples/00-introduction
   examples/evaluating-survival-models
   examples/survival-svm
   examples/random-survival-forest


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
