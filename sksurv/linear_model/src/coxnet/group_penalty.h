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
#ifndef GLMNET_GROUP_PENALTY_H
#define GLMNET_GROUP_PENALTY_H

#include <Eigen/Core>

#include "data.h"
#include "fit_params.h"
#include "fit_result.h"
#include "parameters.h"


namespace coxnet {

template<typename T, typename S, typename U>
class GroupPenalty {
public:
    using Scalar = typename T::Scalar;
    using Index = typename T::Index;
    typedef Eigen::Matrix <Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
    typedef DataWithGroups <T, S, U> DataType;
    typedef typename DataType::SizeVector SizeVectorType;

    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorType);

    GroupPenalty(const DataType &data, const Parameters &params) : m_data(data), m_parameters(params),
       m_params(data.n_samples(), data.n_features(), params.get_tolerance())
    {
    }

    template<typename VectorDerived, typename ResultType>
    void fit(const Eigen::MatrixBase<VectorDerived> &alphas,
             bool create_path,
             ResultType& result);

private:
    void fit_alpha(Scalar l1penalty, Scalar l2penalty);

    template<typename __iter>
    Scalar update_groups(Scalar l1penalty,
                         Scalar l2penalty,
                         VectorType &coef_x,
                         const __iter &iter_begin,
                         const __iter &iter_end);

    const DataType &m_data;
    const Parameters &m_parameters;

    FitParams<VectorType> m_params;
};

};

#include "group_penalty.hxx"

#endif //GLMNET_GROUP_PENALTY_H
