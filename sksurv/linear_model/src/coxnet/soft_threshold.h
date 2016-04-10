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
#ifndef GLMNET_SOFT_THRESHOLD_H
#define GLMNET_SOFT_THRESHOLD_H

#include <cmath>
#include <Eigen/Core>


template <typename _T>
bool is_zero (const _T value,
              const _T &prec = Eigen::NumTraits<_T>::dummy_precision())
{
    return (std::fabs(value) <= prec);
}

inline
double fsign(double f) {
    double val;
    if (is_zero(f))
        val = 0.;
    else if (f > 0)
        val = 1.;
    else //if (f < 0)
        val = -1.;
    return val;
}

inline
float fsign(float f) {
    float val;
    if (is_zero(f))
        val = 0.f;
    else if (f > 0)
        val = 1.f;
    else //if (f < 0)
        val = -1.f;
    return val;
}

inline
double soft_threshold(double z, double t) {
    double v = std::fabs(z) - t;
    if (!is_zero(v) && v > 0)
        return fsign(z) * v;
    return 0.0;
}

inline
float soft_threshold(float z, float t) {
    float v = std::fabs(z) - t;
    if (!is_zero(v) && v > 0)
        return fsign(z) * v;
    return 0.0f;
}

#endif //GLMNET_SOFT_THRESHOLD_H
