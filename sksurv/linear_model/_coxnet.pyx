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
cimport numpy as cnp
from libcpp cimport bool

cnp.import_array()


cdef extern from "coxnet_wrapper.h":
    cdef object fit_coxnet[T, S, U] (cnp.ndarray, cnp.ndarray, cnp.ndarray, cnp.ndarray,
        cnp.ndarray, bool, cnp.npy_float64, cnp.npy_float64, int, double, bool) except +


cdef extern from "coxnet_wrapper.h" namespace "Eigen":

    cdef cppclass Dynamic:
        pass

    cdef cppclass RowMajor:
        pass

    cdef cppclass ColMajor:
        pass

    cdef cppclass Aligned:
        pass

    cdef cppclass Unaligned:
        pass

    cdef cppclass PlainObjectBase:
        pass

    cdef cppclass Matrix(PlainObjectBase):
        pass

    cdef cppclass MatrixXd(PlainObjectBase):
        pass

    cdef cppclass VectorXd(PlainObjectBase):
        pass

    cdef cppclass VectorXuint8(PlainObjectBase):
        pass


def call_fit_coxnet(cnp.ndarray[cnp.npy_float64, ndim=2, mode='fortran'] X,
                    cnp.ndarray[cnp.npy_float64, ndim=1] time,
                    cnp.ndarray[cnp.npy_uint8, ndim=1] event,
                    cnp.ndarray[cnp.npy_float64, ndim=1] penalty_factor,
                    cnp.ndarray[cnp.npy_float64, ndim=1] alphas,
                    bool create_path,
                    cnp.npy_float64 alpha_min_ratio,
                    cnp.npy_float64 l1_ratio,
                    int max_iter,
                    cnp.npy_float64 eps,
                    bool verbose):
    cdef object result = fit_coxnet[MatrixXd, VectorXd, VectorXuint8] (
        X, time, event, penalty_factor, alphas, create_path,
        alpha_min_ratio, l1_ratio, max_iter, eps, verbose)
    return result
