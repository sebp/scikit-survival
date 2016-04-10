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
#ifndef GLMNET_CONSTANTS_H
#define GLMNET_CONSTANTS_H

#include <cmath>

#define LOG_99999 11.512915464920228086754123920088321036576923514461324197617865972730311703692501039693619281678632872501544746697

#if defined(_MSC_VER)
#define COXNET_CONSTEXPR
#else
#define COXNET_CONSTEXPR constexpr
#endif


namespace coxnet {

template<typename Scalar>
struct Constants {
    static COXNET_CONSTEXPR Scalar BIG() { return Scalar(1e35); }
    static COXNET_CONSTEXPR Scalar WEIGHTS_SUM_MIN() { return Scalar((1.0+1.0E-5)*1.0E-5*(1.0-1.0E-5)); }
    static COXNET_CONSTEXPR Scalar PMAX() { return Scalar(LOG_99999); }
    static COXNET_CONSTEXPR Scalar PMIN() { return Scalar(-LOG_99999); }
    static COXNET_CONSTEXPR int MIN_ALPHAS() { return 5; }
};

};

#endif //GLMNET_CONSTANTS_H
