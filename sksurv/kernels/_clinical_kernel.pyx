# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
cimport cython
from libc cimport math

cimport numpy as cnp

cnp.import_array()


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef void _get_min_and_max(cnp.npy_double[:] x, cnp.npy_double * min_out, cnp.npy_double * max_out) nogil:
    cdef cnp.npy_double amin = x[0]
    cdef cnp.npy_double amax = x[0]
    cdef int i

    for i in range(x.shape[0]):
        if x[i] < amin:
            amin = x[i]
        if x[i] > amax:
            amax = x[i]

    min_out[0] = amin
    max_out[0] = amax


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def continuous_ordinal_kernel_with_ranges(cnp.npy_double[:, :] x,
                                          cnp.npy_double[:, :] y,
                                          cnp.npy_double[:] ranges,
                                          cnp.npy_double[:, :] out):
    cdef cnp.npy_intp n_samples_x = x.shape[0]
    cdef cnp.npy_intp n_samples_y = y.shape[0]
    cdef cnp.npy_intp n_features = x.shape[1]
    cdef int i, j, k

    if out.shape[0] != n_samples_x or out.shape[1] != n_samples_y:
        raise ValueError("out matrix must be of shape (%d, %d)" % out.shape)

    with nogil:
        for i in range(n_samples_x):
            for j in range(n_samples_y):
                for k in range(n_features):
                    out[i, j] += (ranges[k] - math.fabs(x[i, k] - y[j, k])) / ranges[k]

    return out


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def continuous_ordinal_kernel(cnp.npy_double[:, :] x,
                              cnp.npy_double[:, :] y,
                              cnp.npy_double[:, :] out):
    cdef cnp.npy_intp n_samples_x = x.shape[0]
    cdef cnp.npy_intp n_features = x.shape[1]
    cdef cnp.npy_double min_x, max_x, min_y, max_y

    cdef cnp.npy_double[:] ranges = cnp.PyArray_EMPTY(1, &n_samples_x, cnp.NPY_DOUBLE, 0)
    with nogil:
        for i in range(n_features):
            _get_min_and_max(x[:, i], &min_x, &max_x)
            _get_min_and_max(y[:, i], &min_y, &max_y)
            ranges[i] = math.fmax(max_x, max_y) - math.fmin(min_x, min_y)

    return continuous_ordinal_kernel_with_ranges(x, y, ranges, out)


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def pairwise_continuous_ordinal_kernel(cnp.npy_double[:] x,
                                       cnp.npy_double[:] y,
                                       cnp.npy_double[:] ranges):
    cdef cnp.npy_double out = 0
    cdef int k

    with nogil:
        for k in range(x.shape[0]):
            out += (ranges[k] - math.fabs(x[k] - y[k])) / ranges[k]

    return out


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def pairwise_nominal_kernel(cnp.npy_int8[:] x,
                            cnp.npy_int8[:] y):
    cdef cnp.npy_double out = 0
    cdef int k

    with nogil:
        for k in range(x.shape[0]):
            if x[k] == y[k]:
                out += 1.

    return out
