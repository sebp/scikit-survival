Release Notes
=============

scikit-survival 0.22.2 (2023-12-30)
-----------------------------------

This release adds support for Python 3.12.

Bug fixes
^^^^^^^^^
- Fix invalid escape sequence in :ref:`Introduction </user_guide/00-introduction.ipynb>` of user guide.

Enhancements
^^^^^^^^^^^^
- Mark Cython functions as noexcept (:issue:`418`).
- Add support for Python 3.12 (:issue:`422`).
- Do not use deprecated ``is_categorical_dtype()`` of Pandas API.

Documentation
^^^^^^^^^^^^^
- Add section :ref:`building-cython-code` to contributing guidelines (:issue:`379`).
- Improve the description of the ``estimate`` parameter in :func:`sksurv.metrics.brier_score`
  and :func:`sksurv.metrics.integrated_brier_score` (:issue:`424`).


scikit-survival 0.22.1 (2023-10-08)
-----------------------------------

Bug fixes
^^^^^^^^^
- Fix error in :meth:`sksurv.tree.SurvivalTree.fit` if ``X`` has missing values and dtype other than float32 (:issue:`412`).


scikit-survival 0.22.0 (2023-10-01)
-----------------------------------

This release adds support for scikit-learn 1.3,
which includes :ref:`missing value support <tree_missing_value_support>` for
:class:`sksurv.tree.SurvivalTree`.
Support for previous versions of scikit-learn has been dropped.

Moreover, a ``low_memory`` option has been added to :class:`sksurv.ensemble.RandomSurvivalForest`,
:class:`sksurv.ensemble.ExtraSurvivalTrees`, and :class:`sksurv.tree.SurvivalTree`
which reduces the memory footprint of calling ``predict``, but disables the use
of ``predict_cumulative_hazard_function`` and ``predict_survival_function``.

Bug fixes
^^^^^^^^^
- Fix issue where an estimator could be fit to data containing
  negative event times (:issue:`410`).

Enhancements
^^^^^^^^^^^^
- Expand test_stacking.py coverage w.r.t. ``predict_log_proba`` (:issue:`380`).
- Add ``low_memory`` option to :class:`sksurv.ensemble.RandomSurvivalForest`,
  :class:`sksurv.ensemble.ExtraSurvivalTrees`, and
  :class:`sksurv.tree.SurvivalTree`. If set, ``predict`` computations use
  less memory, but ``predict_cumulative_hazard_function``
  and ``predict_survival_function`` are not implemented (:issue:`369`).
- Allow calling :meth:`sksurv.meta.Stacking.predict_cumulative_hazard_function`
  and :meth:`sksurv.meta.Stacking.predict_survival_function`
  if the meta estimator supports it (:issue:`388`).
- Add support for missing values in :class:`sksurv.tree.SurvivalTree` based
  on missing value support in scikit-learn 1.3 (:issue:`405`).
- Update bundled Eigen to 3.4.0.

Documentation
^^^^^^^^^^^^^
- Add :attr:`sksurv.meta.Stacking.unique_times_` to API docs.
- Upgrade to Sphinx 6.2.1 and pydata_sphinx_theme 0.13.3 (:issue:`390`).

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The ``loss_`` attribute of :class:`sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`
  and :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis` has been removed (:issue:`402`).
- Support for ``max_features='auto'`` in :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`
  and :class:`sksurv.tree.SurvivalTree` has been removed (:issue:`402`).


scikit-survival 0.21.0 (2023-06-11)
-----------------------------------

This is a major release bringing new features and performance improvements.

- :func:`sksurv.nonparametric.kaplan_meier_estimator` can estimate
  pointwise confidence intervals by specifying the `conf_type` parameter.
- :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis` supports
  early-stopping via the `monitor` parameter of
  :meth:`sksurv.ensemble.GradientBoostingSurvivalAnalysis.fit`.
- :func:`sksurv.metrics.concordance_index_censored` has a significantly
  reduced memory footprint. Memory usage now scales linear, instead of quadratic,
  in the number of samples.
- Fitting of :class:`sksurv.tree.SurvivalTree`,
  :class:`sksurv.ensemble.RandomSurvivalForest`, or :class:`sksurv.ensemble.ExtraSurvivalTrees`
  is about 3x faster.
- Finally, the release adds support for Python 3.11 and pandas 2.0.

Bug fixes
^^^^^^^^^
- Fix bug where `times` passed to :func:`sksurv.metrics.brier_score`
  was downcast, resulting in a loss of precision that may lead
  to duplicate time points (:issue:`349`).
- Fix inconsistent behavior of evaluating functions returned by
  `predict_cumulative_hazard_function` or `predict_survival_function`
  (:issue:`375`).

Enhancements
^^^^^^^^^^^^
- :func:`sksurv.nonparametric.kaplan_meier_estimator`
  and :class:`sksurv.nonparametric.CensoringDistributionEstimator`
  support returning confidence intervals by specifying the `conf_type`
  parameter (:issue:`348`).
- Configure package via pyproject.toml (:issue:`347`).
- Add support for Python 3.11 (:issue:`350`).
- Add support for early-stopping to
  :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`
  (:issue:`354`).
- Do not use deprecated `pkg_resources` API (:issue:`353`).
- Significantly reduce memory usage of :func:`sksurv.metrics.concordance_index_censored`
  (:issue:`362`).
- Set `criterion` attribute in :class:`sksurv.tree.SurvivalTree`
  such that :func:`sklearn.tree.plot_tree` can be used (:issue:`366`).
- Significantly improve speed to fit a :class:`sksurv.tree.SurvivalTree`,
  :class:`sksurv.ensemble.RandomSurvivalForest`, or :class:`sksurv.ensemble.ExtraSurvivalTrees`
  (:issue:`371`).
- Expose ``_predict_risk_score`` attribute in :class:`sklearn.pipeline.Pipeline`
  if the final estimator of the pipeline has such property (:issue:`374`).
- Add support for pandas 2.0 (:issue:`373`).

Documentation
^^^^^^^^^^^^^
- Fix wrong number of selected features in the guide
  :ref:`Introduction to Survival Analysis </user_guide/00-introduction.ipynb>`
  (:issue:`345`).
- Fix broken links with nbsphinx 0.9.2 (:issue:`367`).

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The attribute ``event_times_`` of estimators has been replaced by ``unique_times_``
  to clarify that these are all the unique times points, not just the once where
  an event occurred (:issue:`371`).
- Functions returned by `predict_cumulative_hazard_function` and `predict_survival_function`
  of :class:`sksurv.tree.SurvivalTree`, :class:`sksurv.ensemble.RandomSurvivalForest`,
  and :class:`sksurv.ensemble.ExtraSurvivalTrees` are over all unique time points
  passed as training data, instead of all unique time points where events occurred
  (:issue:`371`).
- Evaluating a function returned by `predict_cumulative_hazard_function`
  or `predict_survival_function` will no longer raise an exception if the
  specified time point is smaller than the smallest time point observed
  during training. Instead, the value at ``StepFunction.x[0]`` will be returned
  (:issue:`375`).


scikit-survival 0.20.0 (2023-03-05)
-----------------------------------

This release adds support for scikit-learn 1.2 and drops support for previous versions.

Enhancements
^^^^^^^^^^^^
- Raise more informative error messages when a parameter does
  not have a valid type/value (see
  `sklearn#23462 <https://github.com/scikit-learn/scikit-learn/issues/23462>`_).
- Add ``positive`` and ``random_state`` parameters to :class:`sksurv.linear_model.IPCRidge`.

Documentation
^^^^^^^^^^^^^
- Update API docs based on scikit-learn 1.2 (where applicable).

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- To align with the scikit-learn API, many parameters of estimators must be
  provided with their names, as keyword arguments, instead of positional arguments.
- Remove deprecated ``normalize`` parameter from :class:`sksurv.linear_model.IPCRidge`.
- Remove deprecated ``X_idx_sorted`` argument from :meth:`sksurv.tree.SurvivalTree.fit`.
- Setting ``kernel="polynomial"`` in :class:`sksurv.svm.FastKernelSurvivalSVM`,
  :class:`sksurv.svm.HingeLossSurvivalSVM`, and :class:`sksurv.svm.MinlipSurvivalAnalysis`
  has been replaced with ``kernel="poly"``.


scikit-survival 0.19.0 (2022-10-23)
-----------------------------------

This release adds :meth:`sksurv.tree.SurvivalTree.apply` and
:meth:`sksurv.tree.SurvivalTree.decision_path`, and support
for sparse matrices to :class:`sksurv.tree.SurvivalTree`.
Moreover, it fixes build issues with scikit-learn 1.1.2
and on macOS with ARM64 CPU.

Bug fixes
^^^^^^^^^
- Fix build issue with scikit-learn 1.1.2, which is binary-incompatible with
  previous releases from the 1.1 series.
- Fix build from source on macOS with ARM64 by specifying numpy 1.21.0 as install
  requirement for that platform (:issue:`313`).

Enhancements
^^^^^^^^^^^^
- :class:`sksurv.tree.SurvivalTree`: Add :meth:`sksurv.tree.SurvivalTree.apply` and
  :meth:`sksurv.tree.SurvivalTree.decision_path` (:issue:`290`).
- :class:`sksurv.tree.SurvivalTree`: Add support for sparse matrices (:issue:`290`).


scikit-survival 0.18.0 (2022-08-15)
-----------------------------------

This release adds support for scikit-learn 1.1, which
includes more informative error messages.
Support for Python 3.7 has been dropped, and
the minimum supported versions of dependencies are updated to

   +--------------+-----------------+
   | Package      | Minimum Version |
   +==============+=================+
   | numpy        | 1.17.3          |
   +--------------+-----------------+
   | Pandas       | 1.0.5           |
   +--------------+-----------------+
   | scikit-learn | 1.1.0           |
   +--------------+-----------------+
   | scipy        | 1.3.2           |
   +--------------+-----------------+

Enhancements
^^^^^^^^^^^^
- Add ``n_iter_`` attribute to all estimators in :ref:`sksurv.svm <mod-svm>` (:issue:`277`).
- Add ``return_array`` argument to all models providing
  ``predict_survival_function`` and ``predict_cumulative_hazard_function``
  (:issue:`268`).

Deprecations
^^^^^^^^^^^^
- The ``loss_`` attribute of :class:`sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`
  and :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`
  has been deprecated.
- The default for the ``max_features`` argument has been changed
  from ``'auto'`` to ``'sqrt'`` for :class:`sksurv.ensemble.RandomSurvivalForest`
  and :class:`sksurv.ensemble.ExtraSurvivalTrees`. ``'auto'`` and ``'sqrt'``
  have the same effect.


scikit-survival 0.17.2 (2022-04-24)
-----------------------------------

This release fixes several issues with packaging scikit-survival.

Bug fixes
^^^^^^^^^
- Added backward support for gcc-c++ (:issue:`255`).
- Do not install C/C++ and Cython source files.
- Add ``packaging`` to build requirements in ``pyproject.toml``.
- Exclude generated API docs from source distribution.
- Add Python 3.10 to classifiers.

Documentation
^^^^^^^^^^^^^
- Use `permutation_importance <https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance>`_
  from sklearn instead of eli5.
- Build documentation with Sphinx 4.4.0.
- Fix missing documentation for classes in ``sksurv.meta``.


scikit-survival 0.17.1 (2022-03-05)
-----------------------------------

This release adds support for Python 3.10.


scikit-survival 0.17.0 (2022-01-09)
-----------------------------------

This release adds support for scikit-learn 1.0, which includes
support for feature names.
If you pass a pandas dataframe to ``fit``, the estimator will
set a `feature_names_in_` attribute containing the feature names.
When a dataframe is passed to ``predict``, it is checked that the
column names are consistent with those passed to ``fit``. See the
`scikit-learn release highlights <https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_0_0.html#feature-names-support>`_
for details.

Bug fixes
^^^^^^^^^
- Fix a variety of build problems with LLVM (:issue:`243`).

Enhancements
^^^^^^^^^^^^
- Add support for ``feature_names_in_`` and ``n_features_in_``
  to all estimators and transforms.
- Add :meth:`sksurv.preprocessing.OneHotEncoder.get_feature_names_out`.
- Update bundled version of Eigen to 3.3.9.

Backwards incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Drop ``min_impurity_split`` parameter from
  :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis`.
- ``base_estimators`` and ``meta_estimator`` attributes of
  :class:`sksurv.meta.Stacking` do not contain fitted models anymore,
  use ``estimators_`` and ``final_estimator_``, respectively.

Deprecations
^^^^^^^^^^^^
- The ``normalize`` parameter of :class:`sksurv.linear_model.IPCRidge`
  is deprecated and will be removed in a future version. Instead, use
  a scikit-learn pipeline:
  ``make_pipeline(StandardScaler(with_mean=False), IPCRidge())``.


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

- Throw an exception when trying to estimate c-index from incomparable data (#117).
- Estimators in ``sksurv.svm`` will now throw an
  exception when trying to fit a model to data with incomparable pairs.


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
Moreover, it fixes a build error on Windows (:issue:`3`)
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
