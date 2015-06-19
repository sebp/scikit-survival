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


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def survival_constraints(cnp.npy_double[:, ::1] x,
                         cnp.npy_uint8[:] y,
                         cnp.npy_double[:] w):
        cdef cnp.npy_intp n_samples = x.shape[0]
        cdef cnp.npy_intp n_features = x.shape[1]
        cdef int k = 0
        cdef int i
        cdef int j
        cdef int m
        cdef cnp.npy_double vi

        cdef cnp.npy_double[:] xw = np.dot(x, w)
        cdef cnp.npy_intp shape[2]
        shape[0] = n_samples * (n_samples - 1) / 2
        shape[1] = n_features
        cdef cnp.npy_double[:, ::1] A = cnp.PyArray_EMPTY(2, shape, cnp.NPY_DOUBLE, 0)

        with nogil:
            for i in range(n_samples - 1):
                if not y[i]:
                    continue
                vi = xw[i] + 1.

                for j in range(i + 1, n_samples):
                    if vi > xw[j]:
                        for m in range(n_features):
                            A[k, m] = x[j, m] - x[i, m]
                        preincrement(k)

        assert 0 < k <= A.shape[0]

        return np.asarray(A[:k, :])
