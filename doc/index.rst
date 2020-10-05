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


.. raw:: html

    <div class="row">
      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="install.html">
            <h3 class="tile-title">Install
            <i class="fas fa-download tile-icon"></i>
            </h3>
          </a>
          <div class="tile-desc">


The easiest way to install scikit-survival is to use
`Anaconda <https://www.anaconda.com/distribution/>`_ by running::

  conda install -c sebp scikit-survival

Alternatively, you can install scikit-survival from source
following :ref:`this guide <install-from-source>`.


.. raw:: html

          </div>
        </div>
      </div>
      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="user_guide/index.html">
            <h3 class="tile-title">User Guide
            <i class="fas fa-book-open tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>The user guide provides in-depth information on the key concepts of scikit-survival, an overview of available survival models, and hands-on examples.</p>
            </div>
          </a>
        </div>
      </div>
      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="api/index.html">
            <h3 class="tile-title">API reference
            <i class="fas fa-cogs tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>The reference guide contains a detailed description of the scikit-survival API. It describes which classes and functions are available and what their parameters are.</p>
            </div>
          </a>
        </div>
      </div>
      <div class="col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex p-1">
        <div class="flex-fill tile">
          <a class="tile-link" href="contributing.html">
            <h3 class="tile-title">Contributing
            <i class="fas fa-code tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>Saw a typo in the documentation? Want to add new functionalities? The contributing guidelines will guide you through the process of setting up a development environment and submitting your changes to the scikit-survival team.</p>
            </div>
          </a>
        </div>
      </div>
    </div>


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
