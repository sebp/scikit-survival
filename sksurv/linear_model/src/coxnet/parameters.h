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
#ifndef GLMNET_PARAMETERS_H
#define GLMNET_PARAMETERS_H

#include <cstddef>


namespace coxnet {

class Parameters {
public:
    Parameters() : m_alpha_min_ratio(0.01), m_l1_ratio(0.5), m_max_iter(10000), m_eps(1e-7), m_verbose(false) {}
    Parameters(const double alpha_min_ratio,
               const double l1_ratio,
               const std::size_t max_iter,
               const double eps,
               const bool verbose) : m_alpha_min_ratio(alpha_min_ratio),
                                     m_l1_ratio(l1_ratio),
                                     m_max_iter(max_iter),
                                     m_eps(eps),
                                     m_verbose(verbose) {}

    void set_alpha_min_ratio(const double value) {
        m_alpha_min_ratio = value;
    }
    double get_alpha_min_ratio() const {
        return m_alpha_min_ratio;
    }

    void set_l1_ratio(const double value) {
        m_l1_ratio = value;
    }
    double get_l1_ratio() const {
        return m_l1_ratio;
    }

    void set_max_iter(const std::size_t value) {
        m_max_iter = value;
    }
    std::size_t get_max_iter() const {
        return m_max_iter;
    }

    void set_tolerance(const double value) {
        m_eps = value;
    }
    double get_tolerance() const {
        return m_eps;
    }

    void set_verbose(const bool value) {
        m_verbose = value;
    }
    bool is_verbose() const {
        return m_verbose;
    }

private:
    double m_alpha_min_ratio;
    double m_l1_ratio;
    std::size_t m_max_iter;
    double m_eps;
    bool m_verbose;
};

}

#endif // GLMNET_PARAMETERS_H
