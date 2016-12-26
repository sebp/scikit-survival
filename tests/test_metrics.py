import os.path

import numpy
from numpy.testing import TestCase, run_module_suite

from sksurv.metrics import concordance_index_censored

WHAS500_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'whas500_predictions.csv')


class TestConcordanceIndex(TestCase):
    def test_concordance_index_no_censoring_all_correct(self):
        time = [1, 5, 6, 11, 34, 45, 46, 50]
        event = numpy.repeat(True, len(time))
        estimate = numpy.arange(len(time))[::-1]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)
        self.assertEqual(28, con)
        self.assertEqual(0, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(0, tie_t)
        self.assertEqual(1.0, c)

    def test_concordance_index_no_censoring_all_wrong(self):
        time = [1, 5, 6, 11, 34, 45, 46, 50]
        event = numpy.repeat(True, len(time))
        # order is exactly reversed
        estimate = numpy.arange(len(time))

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)
        self.assertEqual(0, con)
        self.assertEqual(28, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(0, tie_t)
        self.assertEqual(0.0, c)

    def test_concordance_index_no_ties(self):
        event = [False, True, True, False, False, True, False, False]
        time = [1, 5, 6, 11, 34, 45, 46, 50]
        estimate = [5, 8, 11, 34, 12, 3, 9, 12]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

        self.assertEqual(3, con)
        self.assertEqual(10, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(0, tie_t)
        self.assertAlmostEqual(0.2307692, c, 6)

    def test_concordance_index_with_tied_time(self):
        event = [False, True, True, False, True, False, True, False, False]
        time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
        estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

        self.assertEqual(8, con)
        self.assertEqual(12, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(0, tie_t)
        self.assertAlmostEqual(0.4, c, 6)

    def test_concordance_index_with_tied_time2(self):
        event = [False, True, True, False, False, False, True, False, False]
        time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
        estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

        self.assertEqual(3, con)
        self.assertEqual(12, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(0, tie_t)
        self.assertAlmostEqual(0.2, c, 6)

    def test_concordance_index_with_tied_event(self):
        event = [False, True, False, True, True, False, True, False, False]
        time = [1, 5, 6, 11, 11, 34, 45, 45, 50]
        estimate = [5, 8, 11, 19, 34, 12, 3, 9, 12]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event[::-1], time[::-1], estimate[::-1])

        self.assertEqual(9, con)
        self.assertEqual(8, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(1, tie_t)
        self.assertAlmostEqual(0.5294118, c, 6)

    def test_concordance_index_with_tied_event_and_time(self):
        event = [True, False, False, False, True, False, True, True, False, False, False, True]
        time = [34, 11, 11, 5, 1, 89, 13, 45, 7, 13, 9, 13]
        estimate = [1, 19, 13, 13, 15, 14, 19, 23, 11, 10, 11, 1]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, estimate)

        self.assertEqual(12, con)
        self.assertEqual(9, dis)
        self.assertEqual(1, tie_r)
        self.assertEqual(1, tie_t)
        self.assertAlmostEqual(0.5681818, c, 6)

    def test_concordance_index(self):
        dat = numpy.loadtxt(WHAS500_DATA_FILE, delimiter=",")
        event = dat[:, 0] == 1
        time = dat[:, 1]
        risk = dat[:, 2]

        c, con, dis, tie_r, tie_t = concordance_index_censored(event, time, risk)
        self.assertEqual(57849, con)
        self.assertEqual(17300, dis)
        self.assertEqual(0, tie_r)
        self.assertEqual(119, tie_t)
        self.assertAlmostEqual(0.7697907, c, 6)

    def test_different_length(self):
        event = numpy.array([True, False, False, True, True, False])
        time = numpy.array([1, 5, 10, 12, 7, 65])
        estimate = numpy.array([12, 8, 1, 89, 56, 13])

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: .+",
                               concordance_index_censored, event, time[:3], estimate)

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: .+",
                               concordance_index_censored, event, time, estimate[:3])

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: .+",
                               concordance_index_censored, event[:3], time, estimate, )

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: .+",
                               concordance_index_censored, event, time[:3], estimate[:3])

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: .+",
                               concordance_index_censored, event[:3], time, estimate[:3])

        self.assertRaisesRegex(ValueError, "Found input variables with inconsistent numbers of samples: .+",
                               concordance_index_censored, event[:3], time[:3], estimate)

    def test_boolean_event(self):
        event = numpy.array([1, 0, 0, 1, 1, 0])
        time = numpy.array([1, 5, 10, 12, 7, 65])
        estimate = numpy.array([12, 8, 1, 89, 56, 13])

        self.assertRaisesRegex(ValueError, "only boolean arrays are supported as class labels for survival analysis.+",
                               concordance_index_censored, event, time, estimate)

    def test_min_samples(self):
        event = numpy.array([False])
        time = numpy.array([10])
        estimate = numpy.array([12])

        self.assertRaisesRegex(ValueError, "Need a minimum of two samples",
                               concordance_index_censored, event, time, estimate)

    def test_all_censored(self):
        event = numpy.array([False, False])
        time = numpy.array([10, 12])
        estimate = numpy.array([12, 13])

        self.assertRaisesRegex(ValueError, "All samples are censored",
                               concordance_index_censored, event, time, estimate)

    def test_all_finite(self):
        event = numpy.array([True, False, None, True, True, False])
        time = numpy.array([1, 5, 10, 12, 7, 65], dtype=float)
        estimate = numpy.array([12, 8, 1, 89, 56, 13], dtype=float)

        self.assertRaisesRegex(ValueError, "Input contains NaN, infinity or a value too large for .+",
                               concordance_index_censored, event, time, estimate)

        event[2] = False
        time[3] = numpy.nan
        self.assertRaisesRegex(ValueError, "Input contains NaN, infinity or a value too large for .+",
                               concordance_index_censored, event, time, estimate)

        time[3] = numpy.nan
        estimate[5] = numpy.inf
        self.assertRaisesRegex(ValueError, "Input contains NaN, infinity or a value too large for .+",
                               concordance_index_censored, event, time, estimate)


if __name__ == '__main__':
    run_module_suite()
