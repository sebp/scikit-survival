/**
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef GLMNET_EIGEN_WRAPPER_H
#define GLMNET_EIGEN_WRAPPER_H

#include <Python.h>
#include "numpy/arrayobject.h"

#include "coxnet/coxnet.h"


namespace Eigen {
    using VectorXuint8 = Matrix<std::uint8_t, Eigen::Dynamic, 1>;
}


template <typename MatrixType>
Eigen::Map<MatrixType> create_map(PyArrayObject *object) {
    using MapType = Eigen::Map<MatrixType>;
    using PointerType = typename MapType::PointerType;

    // ROW: If array is in row-major order, transpose (see README)
    if ((PyObject*)object == Py_None) {
        throw std::invalid_argument("expected numpy array but got None");
    }
    if (!PyArray_ISONESEGMENT(object)) {
        throw std::invalid_argument("Numpy array must be one contiguous segment "
                                    "to be able to be transferred to a Eigen Map.");
    }

    const npy_intp *dimensions = PyArray_DIMS(object);
    npy_intp n_rows = (!PyArray_IS_F_CONTIGUOUS(object)
                       ? ((PyArray_NDIM(object) == 1)
                          ? 1  // ROW: If 1D row-major numpy array, set to 1 (row vector)
                          : dimensions[1])
                       : dimensions[0]);

    // COLUMN: If array is in row-major order: transpose (see README)
    npy_intp n_cols = (!PyArray_IS_F_CONTIGUOUS(object)
                      ? dimensions[0]
                      : ((PyArray_NDIM(object) == 1)
                         ? 1  // COLUMN: If 1D col-major numpy array, set to length (column vector)
                         : dimensions[1]));

    MapType v((PointerType) PyArray_DATA(object), n_rows, n_cols);
    return v;
}


template <typename T, typename S, typename U>
PyObject* fit_coxnet(
        PyArrayObject * X,
        PyArrayObject * time,
        PyArrayObject * event,
        PyArrayObject * penalty_factor,
        PyArrayObject * alphas,
        bool create_path,
        typename T::Scalar alpha_min_ratio,
        typename T::Scalar l1_ratio,
        std::size_t max_iter,
        double eps,
        bool verbose)
{
    using MatrixType = Eigen::Map<T>;
    using VectorType = Eigen::Map<S>;
    using IntVectorType = Eigen::Map<U>;
    using CoxnetType = coxnet::Coxnet<MatrixType, VectorType, IntVectorType>;
    using DataType = typename CoxnetType::DataType;
    using ResultType = coxnet::FitResult<MatrixType, VectorType>;

    MatrixType x_map(create_map<T> (X));
    VectorType time_map(create_map<S> (time));
    IntVectorType event_map(create_map<U> (event));
    VectorType pen_map(create_map<S> (penalty_factor));

    const DataType _data(x_map, time_map, event_map, pen_map);

    auto *final_alphas = (PyArrayObject*)PyArray_EMPTY(1, PyArray_SHAPE(alphas), NPY_FLOAT64, NPY_FORTRANORDER);
    auto *final_dev_ratio = (PyArrayObject*)PyArray_EMPTY(1, PyArray_SHAPE(alphas), NPY_FLOAT64, NPY_FORTRANORDER);
    npy_intp coef_shape[2] = { PyArray_DIM(X, 1), PyArray_DIM(alphas, 0) };
    auto *coef_path = (PyArrayObject*)PyArray_EMPTY(2, coef_shape, NPY_FLOAT64, NPY_FORTRANORDER);

    MatrixType coef_path_map(create_map<T> (coef_path));
    VectorType final_alphas_map(create_map<S> (final_alphas));
    VectorType final_dev_ratio_map(create_map<S> (final_dev_ratio));
    ResultType result(coef_path_map, final_alphas_map, final_dev_ratio_map);

    const coxnet::Parameters _params(alpha_min_ratio, l1_ratio, max_iter, eps, verbose);
    CoxnetType object(_data, _params);
    const VectorType alphas_map(create_map<S> (alphas));
    object.fit(alphas_map, create_path, result);

    switch (result.getError()) {
        case WEIGHT_TOO_LARGE: {
            throw std::range_error(
                "Numerical error, because weights are too large. Consider increasing alpha.");
            return NULL;
        }
        case NONE:
            break;
    }

    if (result.getNumberOfAlphas() != PyArray_DIM(alphas, 0)) {
        npy_intp vec_shape[1] = { result.getNumberOfAlphas()  };
        PyArray_Dims vec_dim;
        vec_dim.ptr = vec_shape;
        vec_dim.len = 1;

        PyArray_Resize (final_alphas, &vec_dim, 1, NPY_FORTRANORDER);
        PyArray_Resize (final_dev_ratio, &vec_dim, 1, NPY_FORTRANORDER);

        npy_intp mat_shape[2] = { coef_shape[0], result.getNumberOfAlphas() };
        PyArray_Dims mat_dim;
        mat_dim.ptr = mat_shape;
        mat_dim.len = 2;
        PyArray_Resize (coef_path, &mat_dim, 1, NPY_FORTRANORDER);
    }

    PyObject *py_result = PyTuple_New(4);
    PyTuple_SET_ITEM (py_result, 0, (PyObject*)coef_path);
    PyTuple_SET_ITEM (py_result, 1, (PyObject*)final_alphas);
    PyTuple_SET_ITEM (py_result, 2, (PyObject*)final_dev_ratio);
    PyTuple_SET_ITEM (py_result, 3, PyLong_FromSize_t(result.getNumberOfIterations()));

    return py_result;
}

#endif //GLMNET_EIGEN_WRAPPER_H
