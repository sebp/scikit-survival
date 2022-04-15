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
#ifndef GLMNET_COXNET_H
#define GLMNET_COXNET_H

#include <Eigen/Core>
#include <cstdint>
#include <cmath>
#include <iostream>

#include "constants.h"
#include "data.h"
#include "fit_result.h"
#include "fit_params.h"
#include "parameters.h"
#include "soft_threshold.h"


namespace coxnet {

template<typename T, typename S, typename U>
class Coxnet
{
public:
    typedef typename T::Scalar Scalar;
    typedef typename T::Index Index;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorType;
    typedef Data<T, S, U> DataType;

    EIGEN_STATIC_ASSERT_VECTOR_ONLY(VectorType);

    Coxnet(const DataType &data, const Parameters &params) : m_data(data), m_parameters(params),
        m_params(data.n_samples(), data.n_features(), params.get_tolerance())
    {
    }

    template<typename VectorDerived, typename ResultType>
    void fit(const Eigen::MatrixBase<VectorDerived> &alphas, bool create_path, ResultType& result);


private:
    void fit_alpha(Scalar l1penalty, Scalar l2penalty);

    void update_maybe_active_set(const Scalar &alpha, const Scalar &alpha_old);

    void update();
    Scalar log_likelihood();
    Scalar null_deviance (Scalar *out_loglik_saturated);

    const DataType &m_data;
    const Parameters &m_parameters;

    FitParams<VectorType> m_params;
};

template <typename T, typename S, typename U>
typename Coxnet<T, S, U>::Scalar
Coxnet<T, S, U>::null_deviance (
    Scalar *out_loglik_saturated)
{
    const U &event = m_data.event();
    Scalar norm_factor = 1.0 / m_data.n_samples();
    Index n_events = 0;
    Scalar out_loglik_null = 0;

    for (Index i = 0; i < m_data.n_samples(); ++i) {
        if (event[i] != 0) {
            out_loglik_null -= norm_factor * std::log((i + 1) * norm_factor);
            n_events++;
        }
    }

    (*out_loglik_saturated) = - n_events * norm_factor * std::log(norm_factor);

    return (*out_loglik_saturated) - out_loglik_null;
}

template <typename T, typename S, typename U>
typename Coxnet<T, S, U>::Scalar
Coxnet<T, S, U>::log_likelihood ()
{
    const U &event = m_data.event();
    const VectorType &xw = m_params.xw;
    const VectorType &risk_set = m_params.risk_set;

    Scalar norm_factor = 1.0 / m_data.n_samples();
    Scalar loglik = 0.;

    /* iterate time in descending order */
    for (Index i = 0; i < m_data.n_samples(); ++i) {
        if (event[i] != 0) {
            loglik += (xw[i] - std::log(risk_set[i])) * norm_factor;
        }
    }

    return loglik;
}

template <typename T, typename S, typename U>
void
Coxnet<T, S, U>::update ()
{
    VectorType &out_risk_set = m_params.risk_set;
    VectorType &out_weights = m_params.weights;
    VectorType &out_residuals = m_params.residuals;
    const S &time = m_data.time();
    const U &event = m_data.event();
    const VectorType &xw = m_params.xw;
    Scalar ti;
    Scalar v;

    Index n_samples = m_data.n_samples();
    Scalar norm_factor = 1.0 / n_samples;
    VectorType exp_xw = xw.array().exp();

    /* iterate time in descending order */
    out_risk_set[0] = exp_xw[0] * norm_factor;
    Index i = 1;
    Index k = 1;
    while (i < n_samples) {
        ti = time[i];
        v = out_risk_set[i - 1];

        while (k < n_samples && ti == time[k]) {
            /* abort if values are too large */
            if (!std::isfinite(exp_xw[k])) {
                m_params.error_type = WEIGHT_TOO_LARGE;
            }
            v += exp_xw[k] * norm_factor;
            k++;
        }

        while (i < k) {
            out_risk_set[i] = v;
            i++;
        }
    }

    Scalar denominator_risk_set = 0.0;
    Scalar denominator_risk_set_sq = 0.0;

    /* iterate time in ascending order */
    i = n_samples - 1;
    while (i >= 0) {
        ti = time[i];
        std::uint64_t c = 0;
        Index k = i;
        while (k >= 0 && ti == time[k]) {
            if (event[k] == 1) {
                c++;
            }
            k--;
        }

        v = out_risk_set[i];
        denominator_risk_set += c * norm_factor / v;
        denominator_risk_set_sq += c * norm_factor / (v * v);

        while (i > k) {
            v = exp_xw[i] * norm_factor;
            out_weights[i] = v * (denominator_risk_set - v * denominator_risk_set_sq);
            out_residuals[i] = (event[i] * norm_factor - v * denominator_risk_set);

            i--;
        }
    }
}


template <typename T, typename S, typename U>
template<typename VectorDerived, typename ResultType>
void Coxnet<T, S, U>::fit(
        const Eigen::MatrixBase<VectorDerived> &alphas,
        bool create_path,
        ResultType &result)
{
    Scalar alpha = 0;
    Scalar alpha_old;
    Scalar l1penalty;
    Scalar l2penalty;

    Scalar loglik_saturated = 0;
    Scalar deviance_null;

    const Index n_alphas = alphas.size();
    const double factor = std::pow(m_parameters.get_alpha_min_ratio(), 1.0 / (n_alphas - 1.0));
    const double l1_ratio = m_parameters.get_l1_ratio();

    typename ResultType::VectorType &final_alphas = result.getAlphas();
    typename ResultType::MatrixType &coef_path = result.getCoefficientPath();
    typename ResultType::VectorType &dev_ratio_path = result.getDevianceRatio();
    const T &m_X = m_data.x();

    m_params.init();
    update();
    deviance_null = null_deviance(&loglik_saturated);
    m_params.eps *= deviance_null;

    VectorType inverse_penalty = m_data.penalty_factor().unaryExpr(
            [] (Scalar x) {
                return (is_zero(x)) ? 0. : 1. / x;
            });

    Index i;
    for (i = 0; i < n_alphas; ++i) {
        alpha_old = alpha;
        if (create_path) {
            if (i == 0)
                alpha = Constants<Scalar>::BIG();
            else if (i == 1) {
                Scalar max_coef = ((m_X.transpose() * m_params.residuals).cwiseAbs().cwiseProduct(inverse_penalty)).maxCoeff();
                alpha_old = max_coef / l1_ratio;
                final_alphas[0] = alpha_old;
                alpha = alpha_old * factor;
            } else
                alpha *= factor;
        } else {
            alpha = alphas[i];
        }

        eigen_assert (!is_zero(alpha) && alpha > 0);

        l1penalty = alpha * l1_ratio;
        l2penalty = alpha * (1.0 - l1_ratio);

        update();
        update_maybe_active_set(alpha, alpha_old);
        fit_alpha(l1penalty, l2penalty);

        if (m_params.has_error()) {
            result.setError(m_params.error_type);
            break;
        }

        final_alphas[i] = alpha;
        coef_path.col(i) = m_params.coef_x;
        dev_ratio_path[i] = 1. - (loglik_saturated - log_likelihood()) / deviance_null;

        if (i >= Constants<Scalar>::MIN_ALPHAS()) {
            /* abort if change in deviance ratio was small, compared to previous (larger) alpha */
            if (dev_ratio_path[i - Constants<Scalar>::MIN_ALPHAS() + 1] / dev_ratio_path[i] > 0.999) {
                ++i;
                break;
            }
        }
    }

    result.setNumberOfAlphas(i);
    result.setNumberOfIterations(m_params.n_iterations);
}


template <typename T, typename S, typename U>
void Coxnet<T, S, U>::update_maybe_active_set(const Scalar &alpha, const Scalar &alpha_old) {
    Scalar strong_rule_threshold = m_parameters.get_l1_ratio() * (2.0 * alpha - alpha_old);

    // np.abs(np.dot(X.T, residuals)) > strong_rule_threshold * penalty_factor
    auto term = (m_data.x().transpose() * m_params.residuals).cwiseAbs().array() > (strong_rule_threshold * m_data.penalty_factor()).array();
    m_params.maybe_active_set = m_params.maybe_active_set.max(term);
};


template <typename T, typename S, typename U>
void Coxnet<T, S, U>::fit_alpha(Scalar l1penalty, Scalar l2penalty) {
    VectorType wx2(m_data.n_features());
    Scalar coef_j;
    Scalar temp;
    Scalar delta_coef;
    const T &m_X = m_data.x();
    const S &m_penalty_factor = m_data.penalty_factor();
    std::size_t max_iter = m_parameters.get_max_iter();

    while (true) {
        VectorType coef_x(m_params.coef_x);

        while (true) {
            /* complete cycle */
            m_params.n_iterations++;
            Scalar max_update = 0.;

            for (Index j = 0; j < m_data.n_features(); ++j) {
                if (m_params.maybe_active_set[j])
                    wx2[j] = m_X.col(j).array().square().matrix().dot(m_params.weights);
            }

            /* update coefficients */
            for (Index j = 0; j < m_data.n_features(); ++j) {
                if (!m_params.maybe_active_set[j])
                    continue;

                coef_j = coef_x[j];
                temp = m_X.col(j).dot(m_params.residuals) + coef_j * wx2[j];
                coef_x[j] = soft_threshold(temp,
                                           l1penalty * m_penalty_factor[j]) / (wx2[j] + l2penalty * m_penalty_factor[j]);
                eigen_assert (coef_x[j] == coef_x[j]);

                delta_coef = coef_x[j] - coef_j;
                if (!is_zero(delta_coef)) {
                    m_params.residuals -= delta_coef * m_X.col(j).cwiseProduct(m_params.weights);
                    m_params.xw += delta_coef * m_X.col(j);
                    max_update = std::max(max_update, wx2[j] * delta_coef * delta_coef);
                    /* update active set */
                    m_params.active_set.insert_ordered(j);
                }
            }

            if (max_update < m_params.eps)
                break;

            /* iterate over active set */
            while (true) {
                m_params.n_iterations++;
                Scalar max_update_old(max_update);
                max_update = 0.;

                /* update coefficients */
                for (auto it = m_params.active_set.cbegin_ordered();
                     it != m_params.active_set.cend_ordered(); ++it) {
                    Index j = *it;
                    coef_j = coef_x[j];
                    temp = m_X.col(j).dot(m_params.residuals) + coef_j * wx2[j];
                    coef_x[j] = soft_threshold(temp,
                                               l1penalty * m_penalty_factor[j]) / (wx2[j] + l2penalty * m_penalty_factor[j]);
                    eigen_assert (coef_x[j] == coef_x[j]);

                    delta_coef = coef_x[j] - coef_j;
                    if (!is_zero(delta_coef)) {
                        m_params.residuals -= delta_coef * m_X.col(j).cwiseProduct(m_params.weights);
                        m_params.xw += delta_coef * m_X.col(j);
                        max_update = std::max(max_update, wx2[j] * delta_coef * delta_coef);
                    }
                }

                if (m_parameters.is_verbose() && max_update - max_update_old > 1e-4) {
                    std::cerr << "max update after " << m_params.n_iterations
                              << " iterations increased from "
                              << max_update_old
                              << " to "<< max_update << std::endl;
                }

                if (max_update < m_params.eps || m_params.n_iterations > max_iter)
                    break;
            }

            if (m_params.n_iterations > max_iter)
                break;
        }

        /* check convergence */
//        if (m_params.weights_sum <= WEIGHTS_SUM_MIN)
//            return;
        bool converged = m_params.n_iterations > max_iter;
        if (!converged) {
            /* update residuals and weights */
            update();
            if (m_params.has_error())
                break;

            converged = true;
            /* check active coefficients */
            for (auto it = m_params.active_set.cbegin_ordered();
                 it != m_params.active_set.cend_ordered(); ++it) {
                Index j = *it;
                delta_coef = coef_x[j] - m_params.coef_x[j];
                if (wx2[j] * delta_coef * delta_coef >= m_params.eps) {
                    converged = false;
                    break;
                }
            }

            if (converged) {
                /* check if set of maybe active coefficients changes */
                for (Index j = 0; j < m_data.n_features(); ++j) {
                    if (m_params.maybe_active_set[j])
                        continue;
                    temp = m_X.col(j).dot(m_params.residuals);
                    if (std::fabs(temp) > l1penalty * m_penalty_factor[j]) {
                        m_params.maybe_active_set[j] = true;
                        converged = false;
                    }
                }
            }
        }

        m_params.coef_x = coef_x;

        if (converged)
            break;
    }
}

};

#endif //GLMNET_COXNET_H
