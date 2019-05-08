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
#include "soft_threshold.h"
#include "irange.h"

namespace coxnet {

template<typename T, typename S, typename U>
template<typename VectorDerived, typename ResultType>
void GroupPenalty<T, S, U>::fit(const Eigen::MatrixBase<VectorDerived> &alphas,
                                bool create_path, ResultType& result)
{
    Scalar alpha = 0;
    Scalar alpha_old;
    Scalar l1penalty;
    Scalar l2penalty;

    const Index n_alphas = alphas.size();
    const double factor = std::pow(m_parameters.get_alpha_min_ratio(), 1.0 / (n_alphas - 1.0));
    const double l1_ratio = m_parameters.get_l1_ratio();

    typename ResultType::VectorType &final_alphas = result.getAlphas();
    typename ResultType::MatrixType &coef_path = result.getCoefficientPath();
    typename ResultType::VectorType &dev_ratio_path = result.getDevianceRatio();
    const T &m_X = m_data.x();

    m_params.init();

    Index i;
    for (i = 0; i <= n_alphas; ++i) {
        alpha_old = alpha;
        if (create_path) {
            if (i == 0)
                alpha = Constants<Scalar>::BIG();
            else if (i == 1) {
                const SizeVectorType &groups = m_data.groups();

                VectorType Xr = VectorType::Zero(m_X.cols());  // TODO
                Scalar max_coef = 0.0;
                for (Index k = 0; k < groups.size() - 1; ++k) {
                    Index group_size = groups[k + 1] - groups[k];
                    max_coef = std::max(
                        max_coef,
                        Xr.segment(groups[k], group_size).norm() / std::sqrt(group_size));
                }
                alpha_old = max_coef / l1_ratio;
                final_alphas[0] = alpha_old;
                alpha = alpha_old * factor;
            } else
                alpha *= factor;
        } else {
            if (i == 0)
                continue;
            alpha = alphas[i - 1];
        }

        eigen_assert (!is_zero(alpha) && alpha > 0);

        l1penalty = alpha * l1_ratio;
        l2penalty = alpha * (1.0 - l1_ratio);

        fit_alpha(l1penalty, l2penalty);

        if (m_params.has_error()) {
            result.setError(m_params.error_type);
            break;
        }

        final_alphas[i] = alpha;
        coef_path.col(i) = m_params.coef_x;
    }

    result.setNumberOfAlphas(i);
    result.setNumberOfIterations(m_params.n_iterations);
};


template <typename T, typename S, typename U>
void GroupPenalty<T, S, U>::fit_alpha(Scalar l1penalty, Scalar l2penalty) {
//    const S &m_penalty_factor = m_data.penalty_factor();
    const SizeVectorType &groups = m_data.groups();
    const Index n_groups = groups.size() - 1;
    double max_iter = m_parameters.get_max_iter();
    Scalar max_update;
    Scalar max_update_old = std::numeric_limits<Scalar>::infinity();

    while (m_params.n_iterations < max_iter) {
        VectorType coef_x(m_params.coef_x);

        while (m_params.n_iterations < max_iter) {
            m_params.n_iterations++;

            // complete cycle
            // update groups
            auto ri = irange<Index> (0, n_groups);
            max_update = update_groups(l1penalty, l2penalty, coef_x,
                    ri.begin(), ri.end());

            while (m_params.n_iterations < max_iter) {
                m_params.n_iterations++;

                // iterate over active set
                // update coefficients
                max_update = update_groups(l1penalty, l2penalty, coef_x,
                              m_params.active_set.cbegin_ordered(),
                              m_params.active_set.cend_ordered());

                if (m_parameters.is_verbose() && max_update - max_update_old > 1e-4) {
                    std::cerr << "max update after " << m_params.n_iterations
                              << " iterations increased from "
                              << max_update_old
                              << " to " << max_update << std::endl;
                }

                if (max_update < m_params.eps)
                    break;
                max_update_old = max_update;
            }
        }

        Scalar delta_coef;
        bool converged = true;
        // check active coefficients
        for (auto it = m_params.active_set.cbegin_ordered();
             it != m_params.active_set.cend_ordered(); ++it) {
            Index j = *it;
            delta_coef = coef_x[j] - m_params.coef_x[j];
            if (delta_coef * delta_coef >= m_params.eps) {
                converged = false;
                break;
            }
        }

        m_params.coef_x = coef_x;

        if (converged)
            break;
    }
}


template <typename T, typename S, typename U>
template <typename __iter>
typename GroupPenalty<T, S, U>::Scalar
GroupPenalty<T, S, U>::update_groups(
    Scalar l1penalty,
    Scalar l2penalty,
    VectorType &coef_x,
    const __iter &iter_begin,
    const __iter &iter_end)
{
    const T &m_X = m_data.x();
    const SizeVectorType &groups = m_data.groups();
    Scalar coef_j;
    Scalar delta_coef;
    Scalar lam1;
    Index group_start;
    Index group_end;
    Index group_size;
    Scalar group_multiplier;
    Scalar max_update = 0;
    Scalar nu = 1.0;

    for (__iter it = iter_begin; it != iter_end; ++it) {
        Index g = *it;
        group_start = groups[g];
        group_end = groups[g + 1];
        group_size = group_end - group_start;

        group_multiplier = std::sqrt(group_size);
        lam1 = l1penalty * group_multiplier;

        // 1/n * nu * X_j^T * r + nu * beta_j
        VectorType z(group_size);
        for (Index j = group_start; j < group_end; ++j) {
            z[j - group_start] = 0.0; // TODO
        }
        Scalar z_norm = z.norm();

        Scalar st = soft_threshold_norm(z_norm, lam1);

        if (!is_zero(st) || !is_zero(coef_x[group_start])) {
            for (Index j = group_start; j < group_end; ++j) {
                coef_j = coef_x[j];
                coef_x[j] = st * z[j - group_start];
                eigen_assert (coef_x[j] == coef_x[j]);
                delta_coef = coef_x[j] - coef_j;

                max_update = std::max(max_update, std::fabs(delta_coef));
            }
            // update active set
            m_params.active_set.insert_ordered(g);
        }
    }

    return max_update;
};

}
