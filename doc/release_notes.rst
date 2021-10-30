Release Notes
=============

scikit-survival 0.16.0 (2021-10-30)
-----------------------------------

This release adds support for changing the evaluation metric that
is used in estimators' ``score`` method. This is particular useful
for hyper-parameter optimization using scikit-learn's ``GridSearchCV``.
You can now use :class:`sksurv.metrics.as_concordance_index_ipcw_scorer`,
:class:`sksurv.metrics.as_cumulative_dynamic_auc_scorer`, or
:class:`sksurv.metrics.as_integrated_brier_score_scorer` to adjust the
``score`` method to your needs. A detailed example is available in the
:ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Metrics-in-Hyper-parameter-Search>`.

Moreover, this release adds :class:`sksurv.ensemble.ExtraSurvivalTrees`
to fit an ensemble of randomized survival trees, and improves the speed
of :func:`sksurv.compare.compare_survival` significantly.
The documentation has been extended by a section on
the :ref:`time-dependent Brier score </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`.

Bug fixes
^^^^^^^^^
- Columns are dropped in :func:`sksurv.column.encode_categorical`
  despite ``allow_drop=False`` (:issue:`199`).
- Ensure :func:`sksurv.column.categorical_to_numeric` always
  returns series with int64 dtype.

Enhancements
^^^^^^^^^^^^
- Add :class:`sksurv.ensemble.ExtraSurvivalTrees` ensemble (:issue:`195`).
- Faster speed for :func:`sksurv.compare.compare_survival` (:issue:`215`).
- Add wrapper classes :class:`sksurv.metrics.as_concordance_index_ipcw_scorer`,
  :class:`sksurv.metrics.as_cumulative_dynamic_auc_scorer`, and
  :class:`sksurv.metrics.as_integrated_brier_score_scorer` to override the
  default ``score`` method of estimators (:issue:`192`).
- Remove use of deprecated numpy dtypes.
- Remove use of ``inplace`` in pandas' ``set_categories``.

Documentation
^^^^^^^^^^^^^
- Remove comments and code suggesting log-transforming times prior to training Survival SVM (:issue:`203`).
- Add documentation for ``max_samples`` parameter to :class:`sksurv.ensemble.ExtraSurvivalTrees`
  and :class:`sksurv.ensemble.RandomSurvivalForest` (:issue:`217`).
- Add section on time-dependent Brier score (:issue:`220`).
- Add section on using alternative metrics for hyper-parameter optimization.


scikit-survival 0.15.0 (2021-03-20)
-----------------------------------

This release adds support for scikit-learn 0.24 and Python 3.9.
scikit-survival now requires at least pandas 0.25 and scikit-learn 0.24.
Moreover, if :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`.
or :class:`sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`
are fit with ``loss='coxph'``,   `predict_cumulative_hazard_function` and
`predict_survival_function` are now available.
:func:`sksurv.metrics.cumulative_dynamic_auc` now supports evaluating
time-dependent predictions, for instance for a :class:`sksurv.ensemble.RandomSurvivalForest`
as illustrated in the
:ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Time-dependent-Risk-Scores>`.

Bug fixes
^^^^^^^^^
- Allow passing pandas data frames to all ``fit`` and ``predict`` methods (#148).
- Allow sparse matrices to be passed to
  :meth:`sksurv.ensemble.GradientBoostingSurvivalAnalysis.predict`.
- Fix example in user guide using GridSearchCV to determine alphas for CoxnetSurvivalAnalysis (#186).

Enhancements
^^^^^^^^^^^^
- Add score method to :class:`sksurv.meta.Stacking`,
  :class:`sksurv.meta.EnsembleSelection`, and
  :class:`sksurv.meta.EnsembleSelectionRegressor` (#151).
- Add support for `predict_cumulative_hazard_function` and
  `predict_survival_function` to :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`.
  and :class:`sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`
  if model was fit with ``loss='coxph'``.
- Add support for time-dependent predictions to :func:`sksurv.metrics.cumulative_dynamic_auc`
  See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Using-Time-dependent-Risk-Scores>`
  for an example (#134).

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The score method of :class:`sksurv.linear_model.IPCRidge`,
  :class:`sksurv.svm.FastSurvivalSVM`, and :class:`sksurv.svm.FastKernelSurvivalSVM`
  (if ``rank_ratio`` is smaller than 1) now converts predictions on log(time) scale
  to risk scores prior to computing the concordance index.
- Support for cvxpy and cvxopt solver in :class:`sksurv.svm.MinlipSurvivalAnalysis`
  and :class:`sksurv.svm.HingeLossSurvivalSVM` has been dropped. The default solver
  is now ECOS, which was used by cvxpy (the previous default) internally. Therefore,
  results should be identical.
- Dropped the ``presort`` argument from :class:`sksurv.tree.SurvivalTree`
  and :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`.
- The ``X_idx_sorted`` argument in :meth:`sksurv.tree.SurvivalTree.fit`
  has been deprecated in scikit-learn 0.24 and has no effect now.
- `predict_cumulative_hazard_function` and
  `predict_survival_function` of :class:`sksurv.ensemble.RandomSurvivalForest`
  and :class:`sksurv.tree.SurvivalTree` now return an array of
  :class:`sksurv.functions.StepFunction` objects by default.
  Use ``return_array=True`` to get the old behavior.
- Support for Python 3.6 has been dropped.
- Increase minimum supported versions of dependencies. We now require:

   +--------------+-----------------+
   | Package      | Minimum Version |
   +==============+=================+
   | Pandas       | 0.25.0          |
   +--------------+-----------------+
   | scikit-learn | 0.24.0          |
   +--------------+-----------------+


scikit-survival 0.14.0 (2020-10-07)
-----------------------------------

This release features a complete overhaul of the :doc:`documentation <index>`.
It features a new visual design, and the inclusion of several interactive notebooks
in the :ref:`User Guide`.

In addition, it includes important bug fixes.
It fixes several bugs in :class:`sksurv.linear_model.CoxnetSurvivalAnalysis`
where ``predict``, ``predict_survival_function``, and ``predict_cumulative_hazard_function``
returned wrong values if features of the training data were not centered.
Moreover, the `score` function of :class:`sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`
and :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis` will now
correctly compute the concordance index if ``loss='ipcwls'`` or ``loss='squared'``.

Bug fixes
^^^^^^^^^

- :func:`sksurv.column.standardize` modified data in-place. Data is now always copied.
- :func:`sksurv.column.standardize` works with integer numpy arrays now.
- :func:`sksurv.column.standardize` used biased standard deviation for numpy arrays (``ddof=0``),
  but unbiased standard deviation for pandas objects (``ddof=1``). It always uses ``ddof=1`` now.
  Therefore, the output, if the input is a numpy array, will differ from that of previous versions.
- Fixed :meth:`sksurv.linear_model.CoxnetSurvivalAnalysis.predict_survival_function`
  and :meth:`sksurv.linear_model.CoxnetSurvivalAnalysis.predict_cumulative_hazard_function`,
  which returned wrong values if features of training data were not already centered.
  This adds an ``offset_`` attribute that accounts for non-centered data and is added to the
  predicted risk score. Therefore, the outputs of ``predict``, ``predict_survival_function``,
  and ``predict_cumulative_hazard_function`` will be different to previous versions for
  non-centered data (#139).
- Rescale coefficients of :class:`sksurv.linear_model.CoxnetSurvivalAnalysis` if
  `normalize=True`.
- Fix `score` function of :class:`sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`
  and :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis` if ``loss='ipcwls'`` or ``loss='squared'``
  is used. Previously, it returned ``1.0 - true_cindex``.

Enhancements
^^^^^^^^^^^^

- Add :func:`sksurv.show_versions` that prints the version of all dependencies.
- Add support for pandas 1.1
- Include interactive notebooks in documentation on readthedocs.
- Add user guide on `penalized Cox models <user_guide/coxnet.ipynb>`_.
- Add user guide on `gradient boosted models <user_guide/boosting.ipynb>`_.


scikit-survival 0.13.1 (2020-07-04)
-----------------------------------

This release fixes warnings that were introduced with 0.13.0.

Bug fixes
^^^^^^^^^

- Explicitly pass ``return_array=True`` in :meth:`sksurv.tree.SurvivalTree.predict`
  to avoid FutureWarning.
- Fix error when fitting :class:`sksurv.tree.SurvivalTree` with non-float
  dtype for time (#127).
- Fix RuntimeWarning: invalid value encountered in true_divide
  in :func:`sksurv.nonparametric.kaplan_meier_estimator`.
- Fix PendingDeprecationWarning about use of matrix when fitting
  :class:`sksurv.svm.FastSurvivalSVM` if optimizer is `PRSVM` or `simple`.


scikit-survival 0.13.0 (2020-06-28)
-----------------------------------

The highlights of this release include the addition of
:func:`sksurv.metrics.brier_score` and
:func:`sksurv.metrics.integrated_brier_score`
and compatibility with scikit-learn 0.23.

`predict_survival_function` and `predict_cumulative_hazard_function`
of :class:`sksurv.ensemble.RandomSurvivalForest` and
:class:`sksurv.tree.SurvivalTree` can now return an array of
:class:`sksurv.functions.StepFunction`, similar
to :class:`sksurv.linear_model.CoxPHSurvivalAnalysis`
by specifying ``return_array=False``. This will be the default
behavior starting with 0.14.0.

Note that this release fixes a bug in estimating
inverse probability of censoring weights (IPCW), which will
affect all estimators relying on IPCW.

Enhancements
^^^^^^^^^^^^

- Make build system compatible with PEP-517/518.
- Added :func:`sksurv.metrics.brier_score` and
  :func:`sksurv.metrics.integrated_brier_score` (#101).
- :class:`sksurv.functions.StepFunction` can now be evaluated at multiple points
  in a single call.
- Update documentation on usage of `predict_survival_function` and
  `predict_cumulative_hazard_function` (#118).
- The default value of `alpha_min_ratio` of
  :class:`sksurv.linear_model.CoxnetSurvivalAnalysis` will now depend
  on the `n_samples/n_features` ratio.
  If ``n_samples > n_features``, the default value is 0.0001
  If ``n_samples <= n_features``, the default value is 0.01.
- Add support for scikit-learn 0.23 (#119).

Deprecations
^^^^^^^^^^^^

- `predict_survival_function` and `predict_cumulative_hazard_function`
  of :class:`sksurv.ensemble.RandomSurvivalForest` and
  :class:`sksurv.tree.SurvivalTree` will return an array of
  :class:`sksurv.functions.StepFunction` in the future
  (as :class:`sksurv.linear_model.CoxPHSurvivalAnalysis` does).
  For the old behavior, use `return_array=True`.

Bug fixes
^^^^^^^^^

- Fix deprecation of importing joblib via sklearn.
- Fix estimation of censoring distribution for tied times with events.
  When estimating the censoring distribution,
  by specifying ``reverse=True`` when calling
  :func:`sksurv.nonparametric.kaplan_meier_estimator`,
  we now consider events to occur before censoring.
  For tied time points with an event, those
  with an event are not considered at risk anymore and subtracted from
  the denominator of the Kaplan-Meier estimator.
  The change affects all functions relying on inverse probability
  of censoring weights, namely:

  - :class:`sksurv.nonparametric.CensoringDistributionEstimator`
  - :func:`sksurv.nonparametric.ipc_weights`
  - :class:`sksurv.linear_model.IPCRidge`
  - :func:`sksurv.metrics.cumulative_dynamic_auc`
  - :func:`sksurv.metrics.concordance_index_ipcw`

- Throw an exception when trying to estimate c-index from uncomparable data (#117).
- Estimators in ``sksurv.svm`` will now throw an
  exception when trying to fit a model to data with uncomparable pairs.


scikit-survival 0.12 (2020-04-15)
---------------------------------

This release adds support for scikit-learn 0.22, thereby dropping support for
older versions. Moreover, the regularization strength of the ridge penalty
in :class:`sksurv.linear_model.CoxPHSurvivalAnalysis` can now be set per
feature. If you want one or more features to enter the model unpenalized,
set the corresponding penalty weights to zero.
Finally, :class:`sklearn.pipeline.Pipeline` will now be automatically patched
to add support for `predict_cumulative_hazard_function` and `predict_survival_function`
if the underlying estimator supports it.

Deprecations
^^^^^^^^^^^^

- Add scikit-learn's deprecation of `presort` in :class:`sksurv.tree.SurvivalTree` and
  :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`.
- Add warning that default `alpha_min_ratio` in :class:`sksurv.linear_model.CoxnetSurvivalAnalysis`
  will depend on the ratio of the number of samples to the number of features
  in the future (#41).

Enhancements
^^^^^^^^^^^^

- Add references to API doc of :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis` (#91).
- Add support for pandas 1.0 (#100).
- Add `ccp_alpha` parameter for
  `Minimal Cost-Complexity Pruning <https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning>`_
  to :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`.
- Patch :class:`sklearn.pipeline.Pipeline` to add support for
  `predict_cumulative_hazard_function` and `predict_survival_function`
  if the underlying estimator supports it.
- Allow per-feature regularization for :class:`sksurv.linear_model.CoxPHSurvivalAnalysis` (#102).
- Clarify API docs of :func:`sksurv.metrics.concordance_index_censored` (#96).


scikit-survival 0.11 (2019-12-21)
---------------------------------

This release adds :class:`sksurv.tree.SurvivalTree` and :class:`sksurv.ensemble.RandomSurvivalForest`,
which are based on the log-rank split criterion.
It also adds the OSQP solver as option to :class:`sksurv.svm.MinlipSurvivalAnalysis`
and :class:`sksurv.svm.HingeLossSurvivalSVM`, which will replace the now deprecated
`cvxpy` and `cvxopt` options in a future release.

This release removes support for sklearn 0.20 and requires sklearn 0.21.

Deprecations
^^^^^^^^^^^^

- The `cvxpy` and `cvxopt` options for `solver` in :class:`sksurv.svm.MinlipSurvivalAnalysis`
  and :class:`sksurv.svm.HingeLossSurvivalSVM` are deprecated and will be removed in a future
  version. Choosing `osqp` is the preferred option now.

Enhancements
^^^^^^^^^^^^

- Add support for pandas 0.25.
- Add OSQP solver option to :class:`sksurv.svm.MinlipSurvivalAnalysis` and
  :class:`sksurv.svm.HingeLossSurvivalSVM` which has no additional dependencies.
- Fix issue when using cvxpy 1.0.16 or later.
- Explicitly specify utf-8 encoding when reading README.rst (#89).
- Add :class:`sksurv.tree.SurvivalTree` and :class:`sksurv.ensemble.RandomSurvivalForest` (#90).

Bug fixes
^^^^^^^^^

- Exclude Cython-generated files from source distribution because
  they are not forward compatible.


scikit-survival 0.10 (2019-09-02)
---------------------------------

This release adds the `ties` argument to :class:`sksurv.linear_model.CoxPHSurvivalAnalysis`
to choose between Breslow's and Efron's likelihood in the presence of tied event times.
Moreover, :func:`sksurv.compare.compare_survival` has been added, which implements
the log-rank hypothesis test for comparing the survival function of 2 or more groups.

Enhancements
^^^^^^^^^^^^

- Update API doc of predict function of boosting estimators (#75).
- Clarify documentation for GradientBoostingSurvivalAnalysis (#78).
- Implement Efron's likelihood for handling tied event times.
- Implement log-rank test for comparing survival curves.
- Add support for scipy 1.3.1 (#66).

Bug fixes
^^^^^^^^^

- Re-add `baseline_survival_` and `cum_baseline_hazard_` attributes
  to :class:`sksurv.linear_model.CoxPHSurvivalAnalysis` (#76).


scikit-survival 0.9 (2019-07-26)
--------------------------------

This release adds support for sklearn 0.21 and pandas 0.24.

Enhancements
^^^^^^^^^^^^

- Add reference to IPCRidge (#65).
- Use scipy.special.comb instead of deprecated scipy.misc.comb.
- Add support for pandas 0.24 and drop support for 0.20.
- Add support for scikit-learn 0.21 and drop support for 0.20 (#71).
- Explain use of intercept in ComponentwiseGradientBoostingSurvivalAnalysis (#68)
- Bump Eigen to 3.3.7.

Bug fixes
^^^^^^^^^
- Disallow scipy 1.3.0 due to scipy regression (#66).


scikit-survival 0.8 (2019-05-01)
--------------------------------

Enhancements
^^^^^^^^^^^^

- Add :meth:`sksurv.linear_model.CoxnetSurvivalAnalysis.predict_survival_function`
  and :meth:`sksurv.linear_model.CoxnetSurvivalAnalysis.predict_cumulative_hazard_function`
  (#46).
- Add :class:`sksurv.nonparametric.SurvivalFunctionEstimator`
  and :class:`sksurv.nonparametric.CensoringDistributionEstimator` that
  wrap :func:`sksurv.nonparametric.kaplan_meier_estimator` and provide
  a `predict_proba` method for evaluating the estimated function on
  test data.
- Implement censoring-adjusted C-statistic proposed by Uno et al. (2011)
  in :func:`sksurv.metrics.concordance_index_ipcw`.
- Add estimator of cumulative/dynamic AUC of Uno et al. (2007)
  in :func:`sksurv.metrics.cumulative_dynamic_auc`.
- Add flchain dataset (see :func:`sksurv.datasets.load_flchain`).

Bug fixes
^^^^^^^^^

- The `tied_time` return value of :func:`sksurv.metrics.concordance_index_censored`
  now correctly reflects the number of comparable pairs that share the same time
  and that are used in computing the concordance index.
- Fix a bug in :func:`sksurv.metrics.concordance_index_censored` where a
  pair with risk estimates within tolerance was counted both as
  concordant and tied.


scikit-survival 0.7 (2019-02-27)
--------------------------------

This release adds support for Python 3.7 and sklearn 0.20.

**Changes:**

- Add support for sklearn 0.20 (#48).
- Migrate to py.test (#50).
- Explicitly request ECOS solver for :class:`sksurv.svm.MinlipSurvivalAnalysis`
  and :class:`sksurv.svm.HingeLossSurvivalSVM`.
- Add support for Python 3.7 (#49).
- Add support for cvxpy >=1.0.
- Add support for numpy 1.15.


scikit-survival 0.6 (2018-10-07)
--------------------------------

This release adds support for numpy 1.14 and pandas up to 0.23.
In addition, the new class :class:`sksurv.util.Surv` makes it easier
to construct a structured array from numpy arrays, lists, or a pandas data frame.

**Changes:**

- Support numpy 1.14 and pandas 0.22, 0.23 (#36).
- Enable support for cvxopt with Python 3.5+ on Windows (requires cvxopt >=1.1.9).
- Add `max_iter` parameter to :class:`sksurv.svm.MinlipSurvivalAnalysis`
  and :class:`sksurv.svm.HingeLossSurvivalSVM`.
- Fix score function of :class:`sksurv.svm.NaiveSurvivalSVM` to use concordance index.
- :class:`sksurv.linear_model.CoxnetSurvivalAnalysis` now throws an exception if coefficients get too large (#47).
- Add :class:`sksurv.util.Surv` class to ease constructing a structured array (#26).


scikit-survival 0.5 (2017-12-09)
--------------------------------

This release adds support for scikit-learn 0.19 and pandas 0.21. In turn,
support for older versions is dropped, namely Python 3.4, scikit-learn 0.18,
and pandas 0.18.


scikit-survival 0.4 (2017-10-28)
--------------------------------

This release adds :class:`sksurv.linear_model.CoxnetSurvivalAnalysis`, which implements
an efficient algorithm to fit Cox's proportional hazards model with LASSO, ridge, and
elastic net penalty.
Moreover, it includes support for Windows with Python 3.5 and later by making the cvxopt
package optional.


scikit-survival 0.3 (2017-08-01)
--------------------------------

This release adds :meth:`sksurv.linear_model.CoxPHSurvivalAnalysis.predict_survival_function`
and :meth:`sksurv.linear_model.CoxPHSurvivalAnalysis.predict_cumulative_hazard_function`,
which return the survival function and cumulative hazard function using Breslow's
estimator.
Moreover, it fixes a build error on Windows (`gh #3 <https://github.com/sebp/scikit-survival/issues/3>`_)
and adds the :class:`sksurv.preprocessing.OneHotEncoder` class, which can be used in
a `scikit-learn pipeline <http://scikit-learn.org/dev/modules/generated/sklearn.pipeline.Pipeline.html>`_.


scikit-survival 0.2 (2017-05-29)
--------------------------------

This release adds support for Python 3.6, and pandas 0.19 and 0.20.


scikit-survival 0.1 (2016-12-29)
--------------------------------

This is the initial release of scikit-survival.
It combines the `implementation of survival support vector machines <https://github.com/tum-camp/survival-support-vector-machine>`_
with the code used in the `Prostate Cancer DREAM challenge <https://f1000research.com/articles/5-2676/>`_.
