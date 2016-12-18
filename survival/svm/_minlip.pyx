# This file is part of the submission of the Chair for Computer Aided
# Medical Procedures, Technische Universitaet Muenchen, Germany to the
# Prostate Cancer DREAM Challenge 2015.
#
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

import numpy as np
cimport numpy as cnp
from scipy.sparse import csr_matrix

cnp.import_array()


@cython.wraparound(False)
@cython.boundscheck(False)
def create_difference_matrix(cnp.npy_uint8[:] event,
                             cnp.ndarray[cnp.npy_double, ndim=1] time,
                             object kind):
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] order
    cdef cnp.ndarray[cnp.npy_int8, ndim=1] values
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] columns
    cdef cnp.npy_intp max_size
    cdef cnp.npy_intp k
    cdef cnp.npy_intp n_samples = event.shape[0]
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] indptr
    cdef object ret

    order = cnp.PyArray_ArgSort(time, 0, cnp.NPY_MERGESORT)

    if kind == "all":
        max_size = n_samples * n_samples * 2
        values = cnp.PyArray_EMPTY(1, &max_size, cnp.NPY_INT8, 0)
        columns = cnp.PyArray_EMPTY(1, &max_size, cnp.NPY_INTP, 0)

        k = create_difference_matrix_full(event, order, values, columns)
    elif kind == "nearest":
        max_size = n_samples * 2
        values = cnp.PyArray_EMPTY(1, &max_size, cnp.NPY_INT8, 0)
        columns = cnp.PyArray_EMPTY(1, &max_size, cnp.NPY_INTP, 0)

        k = create_difference_matrix_nearest_neighbor(event, order, values, columns)
    elif kind == "next":
        max_size = n_samples * 2
        values = cnp.PyArray_EMPTY(1, &max_size, cnp.NPY_INT8, 0)
        columns = cnp.PyArray_EMPTY(1, &max_size, cnp.NPY_INTP, 0)

        k = create_difference_matrix_direct_neighbor(event, order, values, columns)
    else:
        raise ValueError("pairs must be one of (all|nearest|next)")

    cdef cnp.PyArray_Dims new_dim
    new_dim.ptr = &k
    new_dim.len = 1

    ret = cnp.PyArray_Resize(values, &new_dim, 1, cnp.NPY_CORDER)
    if ret is not None:  # returns NULL on error
        return

    ret = cnp.PyArray_Resize(columns, &new_dim, 1, cnp.NPY_CORDER)
    if ret is not None:  # returns NULL on error
        return

    indptr = cnp.PyArray_Arange(0, k + 1, 2, cnp.NPY_INTP)
    D = csr_matrix((values, columns, indptr), shape=(k // 2, n_samples), copy=False, dtype=np.int8)

    return D


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void set_entries(cnp.npy_intp[:] columns,
                             cnp.npy_int8[:] values,
                             cnp.npy_intp k,
                             cnp.npy_intp i, cnp.npy_intp j) nogil:
    """Create sparse matrix with sorted indices"""
    if i < j:
        columns[k] = i
        values[k] = 1
        columns[k + 1] = j
        values[k + 1] = -1
    else:
        columns[k] = j
        values[k] = -1
        columns[k + 1] = i
        values[k + 1] = 1


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef cnp.npy_intp create_difference_matrix_direct_neighbor(cnp.npy_uint8[:] event,
                                              cnp.npy_intp[:] o,
                                              cnp.npy_int8[:] values,
                                              cnp.npy_intp[:] columns) nogil:
    """Only compare against direct nearest neighbor according to time"""
    cdef cnp.npy_intp n_samples = event.shape[0]
    cdef cnp.npy_intp i
    cdef cnp.npy_intp j = 0
    cdef cnp.npy_intp k = 0
    cdef cnp.npy_intp k1
    cdef cnp.npy_intp k2

    for i in range(1, n_samples):
        while j < i:
            if event[o[j]]:
                set_entries(columns, values, k, o[i], o[j])
                k += 2
            j += 1

    return k


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef cnp.npy_intp create_difference_matrix_nearest_neighbor(cnp.npy_uint8[:] event,
                                               cnp.npy_intp[:] o,
                                               cnp.npy_int8[:] values,
                                               cnp.npy_intp[:] columns) nogil:
    """Only considers comparable pairs (i, j) where j is uncensored sample
    with highest survival time smaller than y_i"""
    cdef cnp.npy_intp n_samples = event.shape[0]
    cdef cnp.npy_intp k = 0
    cdef cnp.npy_intp i, j

    for i in range(1, n_samples):
        j = i - 1
        while j >= 0:
            if event[o[j]]:
                set_entries(columns, values, k, o[i], o[j])
                k += 2
                break
            j -= 1

    return k


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef cnp.npy_intp create_difference_matrix_full(cnp.npy_uint8[:] event,
                                   cnp.npy_intp[:] o,
                                   cnp.npy_int8[:] values,
                                   cnp.npy_intp[:] columns) nogil:
    """Considers all possible comparable pairs"""
    cdef cnp.npy_intp n_samples = event.shape[0]
    cdef cnp.npy_intp i, j
    cdef cnp.npy_intp k = 0

    for i in range(1, n_samples):
        for j in range(i):
            if event[o[j]]:
                set_entries(columns, values, k, o[i], o[j])
                k += 2

    return k
