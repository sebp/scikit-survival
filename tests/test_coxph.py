from os.path import join, dirname
import warnings

import numpy
from numpy.testing import TestCase, assert_array_almost_equal, run_module_suite
import pandas

from sksurv.column import standardize

from sksurv.linear_model.coxph import CoxPHSurvivalAnalysis, CoxPHOptimizer

ROSSI_FILE = join(dirname(__file__), 'data', 'rossi.csv')


class TestCoxPH(TestCase):
    def setUp(self):
        data = pandas.read_csv(ROSSI_FILE)
        self.y = numpy.fromiter(zip(data["arrest"] == 1, data["week"]),
                                dtype=[('arrest', numpy.bool), ('week', numpy.float64)])

        self.x = data.drop(["arrest", "week"], axis=1)

    def test_likelihood(self):
        cph = CoxPHOptimizer(self.x.values, self.y['arrest'], self.y['week'], alpha=0.)

        w = pandas.Series({"fin": -0.37902189,
                           "age": -0.05724593,
                           "race": 0.31412977,
                           "wexp": -0.15111460,
                           "mar": -0.43278257,
                           "paro": -0.08498284,
                           "prio": 0.09111154})

        actual_loss = cph.nlog_likelihood(w.loc[self.x.columns].values)

        self.assertAlmostEqual(659.1206, self.x.shape[0] * actual_loss, 4)

    def test_fit(self):
        cph = CoxPHSurvivalAnalysis()
        cph.fit(self.x.values, self.y)

        expected = pandas.Series({"fin": -0.37902189,
                                  "age": -0.05724593,
                                  "race": 0.31412977,
                                  "wexp": -0.15111460,
                                  "mar": -0.43278257,
                                  "paro": -0.08498284,
                                  "prio": 0.09111154})

        actual = pandas.Series(cph.coef_, index=self.x.columns)
        assert_array_almost_equal(expected.values,
                                  actual.loc[expected.index].values)

    def test_predict(self):
        cph = CoxPHSurvivalAnalysis()
        xc = standardize(self.x, with_std=False)
        cph.fit(xc.values, self.y)

        expected = numpy.array([-0.136002823953217, -1.13104636905577, 0.741965816026403, -0.98072115186145,
                                -0.600098931134794, -0.997407014712788, -0.0993800739865776, -0.266761246895696,
                                -0.665145743277517, -0.418747210463951, -0.0770761787926419, 0.411385264707043,
                                -0.0770761787926419, 0.563114305747799, -1.07096133044073])

        idx = numpy.array([15, 77, 79, 90, 113, 122, 134, 172, 213, 219, 257, 313, 364, 395, 409])

        pred = cph.predict(xc.iloc[idx, :].values)

        assert_array_almost_equal(expected, pred)

    def test_fit_ridge_1(self):
        # coxph(Surv(week, arrest) ~ ridge(fin, age, race, wexp, mar, paro, prio,
        #     theta=1, scale=FALSE), data=rossi, ties="breslow")
        cph = CoxPHSurvivalAnalysis(alpha=1.0)
        cph.fit(self.x.values, self.y)

        expected = pandas.Series({'fin': -0.36366779384675196,
                                  'age': -0.057788417088377418,
                                  'race': 0.28960521422300672,
                                  'wexp': -0.15082851149160476,
                                  'mar': -0.3829568076550468,
                                  'paro': -0.08230383874483703,
                                  'prio': 0.090951189830228568})

        actual = pandas.Series(cph.coef_, index=self.x.columns)
        assert_array_almost_equal(expected.values,
                                  actual.loc[expected.index].values)

    def test_fit_ridge_2(self):
        # coxph(Surv(week, arrest) ~ ridge(fin, age, race, wexp, mar, paro, prio,
        #     theta=19.67, scale=FALSE), data=rossi, ties="breslow")
        cph = CoxPHSurvivalAnalysis(alpha=19.67)
        cph.fit(self.x.values, self.y)

        expected = pandas.Series({'fin': -0.21145000,
                                  'age': -0.06223214,
                                  'race': 0.11957591,
                                  'wexp': -0.10694088,
                                  'mar': -0.13696844,
                                  'paro': -0.04929119,
                                  'prio': 0.09029133})

        actual = pandas.Series(cph.coef_, index=self.x.columns)
        assert_array_almost_equal(expected.values,
                                  actual.loc[expected.index].values)

    def test_alpha(self):
        cph = CoxPHSurvivalAnalysis(alpha=-0.0001)

        self.assertRaisesRegex(ValueError, "alpha must be positive, but was -0\.0001",
                               cph.fit, self.x.values, self.y)

        cph.set_params(alpha=-1.25)
        self.assertRaisesRegex(ValueError, "alpha must be positive, but was -1\.25",
                               cph.fit, self.x.values, self.y)

    def test_convergence(self):
        cph = CoxPHSurvivalAnalysis(n_iter=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            cph.fit(self.x.values, self.y)

            self.assertEqual(1, len(w))
            self.assertEqual("Optimization did not converge: Maximum number of iterations has been exceeded.",
                             str(w[0].message))

if __name__ == '__main__':
    run_module_suite()
