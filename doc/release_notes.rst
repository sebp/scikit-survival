Release Notes
=============

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
