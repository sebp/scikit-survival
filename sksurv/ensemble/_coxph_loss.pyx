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

import numpy as np
cimport numpy as cnp

cnp.import_array()


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def coxph_negative_gradient(cnp.npy_uint8[:] event,
                            cnp.npy_double[:] time,
                            cnp.npy_double[:] y_pred):
    cdef cnp.npy_double s
    cdef int i
    cdef int j
    cdef cnp.npy_intp n_samples = event.shape[0]

    cdef cnp.ndarray[cnp.npy_double, ndim=1] gradient = cnp.PyArray_EMPTY(1, &n_samples, cnp.NPY_DOUBLE, 0)
    cdef cnp.npy_double[:] exp_tsj = cnp.PyArray_ZEROS(1, &n_samples, cnp.NPY_DOUBLE, 0)

    cdef cnp.npy_double[:] exp_pred = np.exp(y_pred)
    with nogil:
        for i in range(n_samples):
            for j in range(n_samples):
                if time[j] >= time[i]:
                    exp_tsj[i] += exp_pred[j]

        for i in range(n_samples):
            s = 0
            for j in range(n_samples):
                if event[j] and time[i] >= time[j]:
                    s += exp_pred[i] / exp_tsj[j]
            gradient[i] = event[i] - s

    return gradient


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def coxph_loss(cnp.npy_uint8[:] event,
               cnp.npy_double[:] time,
               cnp.npy_double[:] y_pred):
    cdef cnp.npy_intp n_samples = event.shape[0]
    cdef cnp.npy_double at_risk
    cdef cnp.npy_double loss = 0

    with nogil:
        for i in range(n_samples):
            at_risk = 0
            for j in range(n_samples):
                if time[j] >= time[i]:
                    at_risk += math.exp(y_pred[j])
            loss += event[i] * (y_pred[i] - math.log(at_risk))

    return - loss
