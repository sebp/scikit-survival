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
#ifndef GLMNET_FIT_PARAMS_H
#define GLMNET_FIT_PARAMS_H

#include <cstdint>
#include <Eigen/Core>

#include "error.h"
#include "ordered_dict.h"


namespace coxnet {

template<typename VectorType>
struct FitParams {
    typedef typename VectorType::Index Index;
    typedef typename VectorType::Scalar Scalar;

    FitParams(Index n_samples,
              Index n_features,
              double _eps) : coef_x(n_features),
                                            residuals(n_samples),
                                            weights(n_samples),
                                            risk_set(n_samples),
                                            xw(n_samples),
                                            eps(_eps),
                                            maybe_active_set(n_features),
                                            n_iterations(0),
                                            error_type(NONE)
    {
    }

    void init();
    bool has_error() const { return error_type != NONE; }

    VectorType coef_x;
    VectorType residuals;
    VectorType weights;
    VectorType risk_set;
    VectorType xw;
    double eps;

    Eigen::Array<bool, Eigen::Dynamic, 1> maybe_active_set;
    ordered_dict<Index> active_set;
    std::size_t n_iterations;
    ErrorType error_type;
};


template <typename VectorType>
void FitParams<VectorType>::init() {
    maybe_active_set.setZero();
    coef_x.setZero();
    xw.setZero();
}

};

#endif //GLMNET_FIT_PARAMS_H
