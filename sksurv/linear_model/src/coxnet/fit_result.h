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
#ifndef GLMNET_FIT_RESULT_H
#define GLMNET_FIT_RESULT_H

#include <cstddef>
#include "error.h"


namespace coxnet {

template <typename T, typename S>
class FitResult {
public:
    typedef T MatrixType;
    typedef S VectorType;

    FitResult(MatrixType &coef,
              VectorType &alphas,
              VectorType &deviance_ratio) : m_coef_path(coef),
                                            m_alphas(alphas),
                                            m_deviance_ratio(deviance_ratio),
                                            m_iterations(0),
                                            m_n_alphas(0),
                                            m_error(NONE)
    {}

    const MatrixType& getCoefficientPath() const {
        return m_coef_path;
    }
    MatrixType& getCoefficientPath() {
        return m_coef_path;
    }

    const VectorType& getAlphas() const {
        return m_alphas;
    }
    VectorType& getAlphas() {
        return m_alphas;
    }

    const VectorType& getDevianceRatio() const {
        return m_deviance_ratio;
    }
    VectorType& getDevianceRatio() {
        return m_deviance_ratio;
    }

    std::size_t getNumberOfIterations() const {
        return m_iterations;
    }
    void setNumberOfIterations(const std::size_t value) {
        m_iterations = value;
    }

    typename VectorType::Index getNumberOfAlphas() const {
        return m_n_alphas;
    }
    void setNumberOfAlphas(const typename VectorType::Index value) {
        m_n_alphas = value;
    }

    ErrorType getError() const {
        return m_error;
    }
    void setError(const ErrorType error_type) {
        m_error = error_type;
    }

private:
    MatrixType &m_coef_path;
    VectorType &m_alphas;
    VectorType &m_deviance_ratio;
    std::size_t m_iterations;
    typename VectorType::Index m_n_alphas;
    ErrorType m_error;

    // intentionally not implemented
    FitResult (const FitResult&);
    FitResult& operator=(const FitResult&);
};

}

#endif //GLMNET_FIT_RESULT_H
