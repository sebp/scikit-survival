# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport calloc, free, malloc, realloc
from libc.stdlib cimport qsort
from libc.string cimport memset
from libc.math cimport fabs, sqrt, INFINITY

import numpy as np
cimport numpy as cnp
cnp.import_array()

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport SIZE_t
from sklearn.tree._tree cimport DOUBLE_t


ctypedef struct Timepoint:

    SIZE_t index
    double time
    double event


ctypedef struct Riskset:

    double n_events
    SIZE_t n_at_risk


cdef int compare_timepoint_desc(const void* a, const void* b) nogil:
    """Comparison function for sort by time (descending order)."""
    cdef double ta = (<Timepoint *> a).time
    cdef double tb = (<Timepoint *> b).time

    if ta > tb:
        return -1
    if ta < tb:
        return 1
    return 0


cdef inline Riskset* set_risk_set(Riskset * riskset, SIZE_t n_events, SIZE_t n_at_risk) nogil:
    riskset.n_events = <double> n_events
    riskset.n_at_risk = n_at_risk
    return riskset


cdef inline Timepoint* set_time_point(Timepoint * tp_ptr, SIZE_t i, const DOUBLE_t[:, ::1] y) nogil:
    tp_ptr.index = i
    tp_ptr.time = y[i, 0]
    tp_ptr.event = y[i, 1]
    return tp_ptr


cdef void compute_risksets(const Timepoint * time_arr, SIZE_t n_samples,
                           const DOUBLE_t[::1] event_times,
                           Riskset * riskset) nogil:
    cdef DOUBLE_t time_idx
    cdef DOUBLE_t tp
    cdef SIZE_t total_events
    cdef SIZE_t n_event_times = event_times.shape[0]
    cdef SIZE_t idx = 0  # index over time_arr
    cdef SIZE_t idx_tp = n_event_times - 1  # index over event_times and riskset

    while idx_tp >= 0 and idx < n_samples:
        time_idx = time_arr[idx].time
        tp = event_times[idx_tp]
        if time_idx < tp:
            set_risk_set(&riskset[idx_tp], 0, idx)
            idx_tp -= 1  # move to next smaller event time point
            continue
        if time_idx > tp:
            idx += 1  # move to to next smaller sample time point
            continue

        total_events = 0
        while idx < n_samples and tp == time_arr[idx].time:
            if time_arr[idx].event != 0.0:
                total_events += 1
            idx += 1

        set_risk_set(&riskset[idx_tp], total_events, idx)
        idx_tp -= 1  # move to next smaller event time point

    # for remaining smaller time points, everyone is at risk
    while idx_tp >= 0:
        set_risk_set(&riskset[idx_tp], 0, n_samples)
        idx_tp -= 1


cdef class LogrankCriterion(Criterion):

    # unique time points sorted in ascending order
    cdef const DOUBLE_t[::1] event_times
    cdef SIZE_t n_event_times
    cdef Riskset * riskset_total
    cdef Riskset * riskset_left

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, const DOUBLE_t[::1] event_times):
        # Default values
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.event_times = event_times
        self.n_event_times = event_times.shape[0]
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL  # not used
        self.sum_left = NULL  # not used
        self.sum_right = NULL  # not used

        self.riskset_total = NULL
        self.riskset_left = NULL

    def __dealloc__(self):
        """Destructor."""
        cdef SIZE_t p
        if self.riskset_total is not NULL:
            free(self.riskset_total)
            free(self.riskset_left)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples, self.event_times), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef Timepoint * time_arr

        time_arr = < Timepoint * > malloc(self.n_node_samples * sizeof(Timepoint))
        if time_arr is NULL:
            raise MemoryError()

        for k, p in enumerate(range(start, end)):
            i = samples[p]
            set_time_point(&time_arr[k], i, self.y)

            if sample_weight is not NULL:
                w = sample_weight[i]

            self.weighted_n_node_samples += w

        qsort(time_arr, self.n_node_samples, sizeof(Timepoint), compare_timepoint_desc)

        if self.riskset_total is not NULL:
            free(self.riskset_total)
            free(self.riskset_left)

        self.riskset_total = < Riskset * > malloc(self.n_event_times * sizeof(Riskset))
        if self.riskset_total is NULL:
            raise MemoryError()

        compute_risksets(time_arr, self.n_node_samples, self.event_times, self.riskset_total)

        free(time_arr)

        self.riskset_left = < Riskset * > malloc(self.n_event_times * sizeof(Riskset))
        if self.riskset_left is NULL:
            raise MemoryError()

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        # cdef SIZE_t n_bytes = self.n_samples * sizeof(double)
        # memset(self.time_points, 0, n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        # cdef SIZE_t n_bytes = self.n_samples * sizeof(double)
        # memset(self.time_points, 0, n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef const double* sample_weight = self.sample_weight
        cdef const SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.start  # always start from the beginning
        cdef SIZE_t n_samples_left = new_pos - pos
        cdef SIZE_t i
        cdef SIZE_t idx_left
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef Timepoint * time_arr

        time_arr = < Timepoint * > malloc(n_samples_left * sizeof(Timepoint))
        if time_arr is NULL:
            raise MemoryError()

        # Update statistics up to new_pos
        self.weighted_n_left = 0.0
        for k, idx_left in enumerate(range(pos, new_pos)):
            i = samples[idx_left]
            set_time_point(&time_arr[k], i, self.y)

            if sample_weight is not NULL:
                w = sample_weight[i]

            self.weighted_n_left += w

        qsort(time_arr, n_samples_left, sizeof(Timepoint), compare_timepoint_desc)

        memset(self.riskset_left, 0, self.n_event_times * sizeof(Riskset))

        # use same time points as in riskset_total
        compute_risksets(time_arr, n_samples_left, self.event_times, self.riskset_left)

        free(time_arr)

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)

        self.pos = new_pos
        return 0

    cdef double impurity_improvement(self, double impurity) nogil:
        """Compute the improvement in impurity"""
        return self.proxy_impurity_improvement()

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction"""

        # cdef double impurity_left
        # cdef double impurity_right
        #
        # self.children_impurity(&impurity_left, &impurity_right)

        cdef SIZE_t i
        cdef double at_risk
        cdef double total_at_risk
        cdef double v
        cdef Riskset * rs_total
        cdef Riskset * rs_left
        cdef double denom = 0.0
        cdef double numer = 0.0

        for i in range(self.n_event_times):
            rs_total = &self.riskset_total[i]
            rs_left = &self.riskset_left[i]
            total_at_risk = <double> rs_total.n_at_risk

            if total_at_risk == 0:
                break  # we reached the end
            at_risk = rs_left.n_at_risk / total_at_risk
            numer += rs_left.n_events - rs_total.n_events * at_risk
            if total_at_risk > 1.0:
                v = (total_at_risk - rs_total.n_events) / (total_at_risk - 1.0) * rs_total.n_events
                denom += at_risk * (1.0 - at_risk) * v

        if denom != 0.0:
            # absolute value is the measure of node separation
            v = fabs(numer / sqrt(denom))
        else:  # all samples are censored
            v = -INFINITY  # indicates that this node cannot be split

        return v

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        return INFINITY

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        impurity_left[0] = INFINITY
        impurity_right[0] = INFINITY

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        # Estimate cumulative hazard function
        cdef SIZE_t k
        cdef SIZE_t j
        cdef double ratio
        cdef Riskset * rs

        rs = &self.riskset_total[0]
        ratio = rs.n_events / (<double> rs.n_at_risk)
        dest[0] = ratio  # Nelson-Aalen estimator
        dest[1] = 1.0 - ratio  # Kaplan-Meier estimator

        j = 2
        for k in range(1, self.n_event_times):
            rs = &self.riskset_total[k]
            dest[j] = dest[j - 2]
            dest[j + 1] = dest[j - 1]
            if rs.n_at_risk != 0:
                ratio = rs.n_events / (<double> rs.n_at_risk)
                dest[j] += ratio
                dest[j + 1] *= 1.0 - ratio
            j += 2
