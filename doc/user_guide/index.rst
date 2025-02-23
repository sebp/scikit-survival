.. _User Guide:

User Guide
==========

The User Guide covers the most important aspects of doing to survival analysis with scikit-survival.

It is assumed that users have a basic understanding of survival analysis. If you are brand-new to survival
analysis, consider studying the basics first, e.g. by reading an introductory book, such as

* David G. Kleinbaum and Mitchel Klein (2012), Survival Analysis: A Self-Learning Text, Springer.
* John P. Klein and Melvin L. Moeschberger (2003), Survival Analysis: Techniques for Censored and Truncated Data, Springer.

Users new to scikit-survival should read :ref:`understanding_predictions` to get familiar with the basic concepts.
The interactive guide :ref:`/user_guide/00-introduction.ipynb` gives a brief overview of how to use scikit-survival for survival analysis.
Once you are familiar with the basics, it is highly recommended reading the guide :ref:`/user_guide/evaluating-survival-models.ipynb`,
which discusses common pitfalls when evaluating the predictive performance of survival models.
Finally, there are several model-specific guides that discuss details about particular models, with many examples throughout.

Background
----------

.. toctree::
   :maxdepth: 1

   understanding_predictions
   00-introduction
   evaluating-survival-models
   competing-risks

Models
------

.. toctree::
   :maxdepth: 1

   coxnet
   random-survival-forest
   boosting
   survival-svm
