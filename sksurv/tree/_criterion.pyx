# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport INFINITY, NAN, fabs, sqrt
from libc.stdlib cimport free, malloc
from libc.string cimport memset

import numpy as np

cimport numpy as cnp

cnp.import_array()

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport DOUBLE_t, SIZE_t


cpdef get_unique_times(cnp.ndarray[DOUBLE_t, ndim=1] time, cnp.ndarray[cnp.npy_bool, ndim=1] event):
    cdef:
        SIZE_t[:] order = cnp.PyArray_ArgSort(time, 0, cnp.NPY_MERGESORT)
        DOUBLE_t value
        DOUBLE_t last_value = NAN
        SIZE_t i
        SIZE_t idx
        list unique_values = []
        list has_event = []

    for i in range(time.shape[0]):
        idx = order[i]
        value = time[idx]
        if value != last_value:
            unique_values.append(value)
            has_event.append(event[idx])
            last_value = value
        if event[idx]:
            has_event[len(has_event) - 1] = True

    return np.asarray(unique_values), np.asarray(has_event, dtype=np.bool_)


cdef class RisksetCounter:
    cdef:
        const DOUBLE_t[:] unique_times
        cnp.npy_int64 * n_events
        cnp.npy_int64 * n_at_risk
        const DOUBLE_t[:, ::1] data
        size_t nbytes

    def __cinit__(self, const DOUBLE_t[:] unique_times):
        cdef SIZE_t n_unique_times = unique_times.shape[0]
        self.nbytes = n_unique_times * sizeof(cnp.npy_int64)
        self.n_events = <cnp.npy_int64 *> malloc(self.nbytes)
        self.n_at_risk = <cnp.npy_int64 *> malloc(self.nbytes)
        self.unique_times = unique_times

    def __dealloc__(self):
        """Destructor."""
        free(self.n_events)
        free(self.n_at_risk)

    cdef void reset(self) noexcept nogil:
        memset(self.n_events, 0, self.nbytes)
        memset(self.n_at_risk, 0, self.nbytes)

    cdef void set_data(self, const DOUBLE_t[:, ::1] data) noexcept nogil:
        self.data = data

    cdef void update(self, const SIZE_t[:] samples, SIZE_t start, SIZE_t end) noexcept nogil:
        cdef:
            SIZE_t i
            SIZE_t idx
            SIZE_t ti
            DOUBLE_t time
            DOUBLE_t event
            const DOUBLE_t[:] unique_times = self.unique_times
            SIZE_t n_times = unique_times.shape[0]
            const DOUBLE_t[:, ::1] y = self.data

        self.reset()

        for i in range(start, end):
            idx = samples[i]
            time, event = y[idx, 0], y[idx, 1]

            # i-th sample is in all risk sets with time <= i-th time
            ti = 0
            while ti < n_times and unique_times[ti] < time:
                self.n_at_risk[ti] += 1
                ti += 1

            if ti < n_times:  # unique_times[ti] == time
                self.n_at_risk[ti] += 1
                if event != 0.0:
                    self.n_events[ti] += 1

    cdef inline void at(self, SIZE_t index, DOUBLE_t * at_risk, DOUBLE_t * events) noexcept nogil:
        if at_risk != NULL:
            at_risk[0] = <DOUBLE_t> self.n_at_risk[index]
        if events != NULL:
            events[0] = <DOUBLE_t> self.n_events[index]


cdef int argbinsearch(const DOUBLE_t[:] arr, DOUBLE_t key_val, SIZE_t * ret) except -1 nogil:
    cdef:
        SIZE_t arr_len = arr.shape[0]
        SIZE_t min_idx = 0
        SIZE_t max_idx = arr_len
        SIZE_t mid_idx
        DOUBLE_t mid_val

    while min_idx < max_idx:
        mid_idx = min_idx + ((max_idx - min_idx) >> 1)

        if mid_idx < 0 or mid_idx >= arr_len:
            return -1

        mid_val = arr[mid_idx]
        if mid_val < key_val:
            min_idx = mid_idx + 1
        else:
            max_idx = mid_idx

    ret[0] = min_idx

    return 0


cdef class LogrankCriterion(Criterion):

    cdef:
        # unique time points sorted in ascending order
        const DOUBLE_t[::1] unique_times
        const cnp.npy_bool[::1] is_event_time
        SIZE_t n_unique_times
        size_t nbytes
        RisksetCounter riskset_total
        cnp.npy_int64 * delta_n_at_risk_left
        cnp.npy_int64 * n_events_left
        SIZE_t * samples_time_idx
        SIZE_t n_samples_left

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples, const DOUBLE_t[::1] unique_times, const cnp.npy_bool[::1] is_event_time):
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.unique_times = unique_times
        self.is_event_time = is_event_time
        self.n_unique_times = unique_times.shape[0]
        self.nbytes = self.n_unique_times * sizeof(cnp.npy_int64)
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.riskset_total = RisksetCounter(unique_times)
        self.delta_n_at_risk_left = <cnp.npy_int64 *> malloc(self.nbytes)
        self.n_events_left = <cnp.npy_int64 *> malloc(self.nbytes)
        self.samples_time_idx = <SIZE_t *> malloc(n_samples * sizeof(SIZE_t))

    def __dealloc__(self):
        """Destructor."""
        free(self.delta_n_at_risk_left)
        free(self.n_events_left)
        free(self.samples_time_idx)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples, self.unique_times, self.is_event_time), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, const DOUBLE_t[:] sample_weight,
                  double weighted_n_samples, const SIZE_t[:] sample_indices, SIZE_t start,
                  SIZE_t end) except -1 nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef:
            SIZE_t i
            SIZE_t idx
            DOUBLE_t time
            DOUBLE_t w = 1.0
            const DOUBLE_t[::1] unique_times = self.unique_times

        self.riskset_total.set_data(y)
        self.riskset_total.update(sample_indices, start, end)

        for i in range(start, end):
            idx = sample_indices[i]
            time = y[idx, 0]
            argbinsearch(unique_times, time, &self.samples_time_idx[idx])

            if sample_weight is not None:
                w = sample_weight[idx]

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef:
            const DOUBLE_t[:] sample_weight = self.sample_weight
            const SIZE_t[:] samples = self.sample_indices
            const DOUBLE_t[:, ::1] y = self.y

            SIZE_t pos = self.start  # always start from the beginning
            SIZE_t i
            SIZE_t idx
            DOUBLE_t event
            SIZE_t time_idx
            DOUBLE_t w = 1.0

        self.n_samples_left = new_pos - pos
        memset(self.delta_n_at_risk_left, 0, self.nbytes)
        memset(self.n_events_left, 0, self.nbytes)

        # Update statistics up to new_pos
        self.weighted_n_left = 0.0
        for i in range(pos, new_pos):
            idx = samples[i]
            event = y[idx, 1]
            time_idx = self.samples_time_idx[idx]

            self.delta_n_at_risk_left[time_idx] += 1
            if event != 0.0:
                self.n_events_left[time_idx] += 1

            if sample_weight is not None:
                w = sample_weight[idx]

            self.weighted_n_left += w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)

        self.pos = new_pos
        return 0

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) noexcept nogil:
        """Compute the improvement in impurity"""
        return self.proxy_impurity_improvement()

    cdef double proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction"""

        cdef:
            SIZE_t i
            DOUBLE_t at_risk = <DOUBLE_t> self.n_samples_left
            DOUBLE_t events
            DOUBLE_t total_at_risk
            DOUBLE_t total_events
            DOUBLE_t ratio
            DOUBLE_t v
            DOUBLE_t denom = 0.0
            DOUBLE_t numer = 0.0

        for i in range(self.n_unique_times):
            events = <DOUBLE_t> self.n_events_left[i]
            self.riskset_total.at(i, &total_at_risk, &total_events)

            if total_at_risk == 0:
                break  # we reached the end
            ratio = at_risk / total_at_risk
            numer += events - total_events * ratio
            if total_at_risk > 1.0:
                v = (total_at_risk - total_events) / (total_at_risk - 1.0) * total_events
                denom += ratio * (1.0 - ratio) * v

            # Update number of samples at risk for next bigger timepoint
            at_risk -= <DOUBLE_t> self.delta_n_at_risk_left[i]

        if denom != 0.0:
            # absolute value is the measure of node separation
            v = fabs(numer / sqrt(denom))
        else:  # all samples are censored
            v = -INFINITY  # indicates that this node cannot be split

        return v

    cdef double node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        return INFINITY

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        impurity_left[0] = INFINITY
        impurity_right[0] = INFINITY

    cdef void node_value(self, double* dest) noexcept nogil:
        """Compute the node value of samples[start:end] into dest."""
        # Estimate cumulative hazard function
        cdef:
            const cnp.npy_bool[::1] is_event_time = self.is_event_time
            SIZE_t i
            SIZE_t j
            DOUBLE_t ratio
            DOUBLE_t n_events
            DOUBLE_t n_at_risk
            DOUBLE_t dest_j0

        # low memory mode
        if  self.n_outputs == 1:
            dest[0] = dest_j0 = 0
            for i in range(0, self.n_unique_times):
                self.riskset_total.at(i, &n_at_risk, &n_events)
                if n_at_risk != 0:
                    ratio = n_events / n_at_risk
                    dest_j0 += ratio
                if is_event_time[i]:
                    dest[0] += dest_j0
        else:
            self.riskset_total.at(0, &n_at_risk, &n_events)
            ratio = n_events / n_at_risk
            dest[0] = ratio  # Nelson-Aalen estimator
            dest[1] = 1.0 - ratio  # Kaplan-Meier estimator

            j = 2
            for i in range(1, self.n_unique_times):
                self.riskset_total.at(i, &n_at_risk, &n_events)
                dest[j] = dest[j - 2]
                dest[j + 1] = dest[j - 1]
                if n_at_risk != 0:
                    ratio = n_events / n_at_risk
                    dest[j] += ratio
                    dest[j + 1] *= 1.0 - ratio
                j += 2
