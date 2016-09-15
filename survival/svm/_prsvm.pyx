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
from cython.operator import preincrement

import numpy as np
cimport numpy as cnp
from scipy.sparse import csr_matrix

cnp.import_array()


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def survival_constraints_simple(cnp.npy_uint8[:] y):
    cdef int i
    cdef int j
    cdef int k = 0
    cdef cnp.npy_intp n_samples = y.shape[0]
    cdef cnp.npy_intp n = n_samples * (n_samples - 1)

    cdef cnp.ndarray[cnp.npy_int8, ndim=1] data = cnp.PyArray_EMPTY(1, &n, cnp.NPY_INT8, 0)
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] indices = cnp.PyArray_EMPTY(1, &n, cnp.NPY_INTP, 0)

    with nogil:
        for i in range(n_samples - 1):
            if y[i] == 0:
                continue

            for j in range(i + 1, n_samples):
                data[k] = -1
                data[k + 1] = 1
                indices[k] = i
                indices[k + 1] = j
                k += 2

    data.resize(k, refcheck=False)
    indices.resize(k, refcheck=False)

    cdef object indptr = cnp.PyArray_Arange(0, k + 1, 2, cnp.NPY_INTP)
    A = csr_matrix((data, indices, indptr), shape=(k // 2, n_samples), dtype=np.int8)

    return A


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def survival_constraints_with_support_vectors(cnp.npy_uint8[:] y,
                                              cnp.npy_double[:] xw):
    cdef int i
    cdef int j
    cdef cnp.npy_double vi
    cdef int k = 0
    cdef cnp.npy_intp n_samples = y.shape[0]
    cdef cnp.npy_intp n = n_samples * (n_samples - 1)

    cdef cnp.ndarray[cnp.npy_int8, ndim=1] data = cnp.PyArray_EMPTY(1, &n, cnp.NPY_INT8, 0)
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] indices = cnp.PyArray_EMPTY(1, &n, cnp.NPY_INTP, 0)

    with nogil:
        for i in range(n_samples - 1):
            if y[i] == 0:
                continue
            vi = xw[i] + 1.

            for j in range(i + 1, n_samples):
                if vi > xw[j]:
                    data[k] = -1
                    data[k + 1] = 1
                    indices[k] = i
                    indices[k + 1] = j
                    k += 2

    data.resize(k, refcheck=False)
    indices.resize(k, refcheck=False)

    cdef object indptr = cnp.PyArray_Arange(0, k + 1, 2, cnp.NPY_INTP)
    A = csr_matrix((data, indices, indptr), shape=(k // 2, n_samples), dtype=np.int8)

    return A
