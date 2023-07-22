.. scikit-survival documentation master file

scikit-survival
===============

scikit-survival is a Python module for `survival analysis <https://en.wikipedia.org/wiki/Survival_analysis>`_
built on top of `scikit-learn <https://scikit-learn.org/>`_. It allows doing survival analysis
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


.. grid:: 2
    :gutter: 3
    :class-container: overview-grid

    .. grid-item-card:: Install :fas:`download`
        :link: install
        :link-type: doc

        The easiest way to install scikit-survival is to use
        `Anaconda <https://www.anaconda.com/distribution/>`_ by running::

          conda install -c sebp scikit-survival

        Alternatively, you can install scikit-survival from source
        following :ref:`this guide <install-from-source>`.


    .. grid-item-card:: User Guide :fas:`book-open`
        :link: user_guide/index
        :link-type: doc

        The user guide provides in-depth information on the key concepts of scikit-survival, an overview of available survival models, and hands-on examples.


    .. grid-item-card:: API Reference :fas:`cogs`
        :link: api/index
        :link-type: doc

        The reference guide contains a detailed description of the scikit-survival API. It describes which classes and functions are available
        and what their parameters are.


    .. grid-item-card:: Contributing :fas:`code`
        :link: contributing
        :link-type: doc

        Saw a typo in the documentation? Want to add new functionalities? The contributing guidelines will guide you through the process of
        setting up a development environment and submitting your changes to the scikit-survival team.


.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   Install <install>
   user_guide/index
   api/index
   Contribute <contributing>
   release_notes
   Cite <cite>
