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
#ifndef GLMNET_DATA_H
#define GLMNET_DATA_H

#include <Eigen/Core>
#include <ostream>

namespace coxnet {

enum {
    FLOATING_POINT_ARGUMENT_PASSED__INTEGER_WAS_EXPECTED=1
};

template <
    typename DerivedMatrix,
    typename DerivedFloatVector,
    typename DerivedIntVector >
class Data
{
public:
    typedef typename DerivedMatrix::Index Index;
    typedef Eigen::MatrixBase<DerivedMatrix> Matrix;
    typedef Eigen::MatrixBase<DerivedFloatVector> FloatVector;
    typedef Eigen::MatrixBase<DerivedIntVector> IntVector;

    EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedFloatVector);
    EIGEN_STATIC_ASSERT_VECTOR_ONLY(DerivedIntVector);
    EIGEN_STATIC_ASSERT(Eigen::NumTraits<typename DerivedIntVector::Scalar>::IsInteger,
                        FLOATING_POINT_ARGUMENT_PASSED__INTEGER_WAS_EXPECTED);

    Data(const Matrix &x,
         const FloatVector &time,
         const IntVector &event,
         const FloatVector &penalty_factor) : m_x(x), m_time(time), m_event(event),
                                               m_penalty_factor(penalty_factor),
                                               m_samples(x.rows()),
                                               m_features(x.cols())
    {
        eigen_assert (time.size() == x.rows());
        eigen_assert (event.size() == x.rows());
        eigen_assert (penalty_factor.size() == x.cols());
        eigen_assert ((event.array() >= 0).all());
        eigen_assert ((event.array() <= 1).all());
    }

    const DerivedMatrix& x() const { return m_x.derived(); }
    const DerivedFloatVector& time() const { return m_time.derived(); }
    const DerivedIntVector& event() const { return m_event.derived(); }
    const DerivedFloatVector& penalty_factor() const { return m_penalty_factor.derived(); }
    const Index& n_samples() const { return m_samples; }
    const Index& n_features() const { return m_features; }

    template<typename _M, typename _V, typename _I>
    friend std::ostream& operator<< (std::ostream& os, const Data<_M, _V, _I> &obj);

private:
    const Matrix &m_x;
    const FloatVector &m_time;
    const IntVector &m_event;
    const FloatVector &m_penalty_factor;
    const Index m_samples;
    const Index m_features;
};

template<typename _M, typename _V, typename _I>
std::ostream& operator<< (std::ostream& os, const Data<_M, _V, _I> &obj) {
    os << "Data(x=" << obj.m_x.size() << ", "
       << "time=" << obj.m_time.size() << ", "
       << "event=" << obj.m_event.size() << ", "
       << "penalty_factor=" << obj.m_penalty_factor.size()
       << ")";
    return os;
}

}

#endif
