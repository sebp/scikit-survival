.. release notes

Release Notes
=============

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
