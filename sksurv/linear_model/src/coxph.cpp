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
#ifndef SK_SURV_COXPH_H
#define SK_SURV_COXPH_H

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <iostream>

template<typename _Scalar, typename _Int>
class CoxPHSolver
{
public:
  typedef _Scalar Scalar;
  typedef _Int Integer;
  typedef Eigen::Matrix<_Int, Eigen::Dynamic, 1> IntVector;
  typedef Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> FloatVector;
  typedef Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic> FloatMatrix;
  typedef typename FloatMatrix::Index Index;

  EIGEN_STATIC_ASSERT_VECTOR_ONLY(FloatVector);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(IntVector);
  EIGEN_STATIC_ASSERT(Eigen::NumTraits<Integer>::IsInteger,
                      FLOATING_POINT_ARGUMENT_PASSED__INTEGER_WAS_EXPECTED);

  template<typename MatrixDerived,
           typename VectorfDerived,
           typename VectoriDerived>
  explicit CoxPHSolver(const Eigen::MatrixBase<MatrixDerived>& x,
                       const Eigen::MatrixBase<VectoriDerived>& event,
                       const Eigen::MatrixBase<VectorfDerived>& time,
                       const Eigen::MatrixBase<VectorfDerived>& alpha,
                       bool breslow)
    : m_x(x)
    , m_event(event)
    , m_time(time)
    , m_alpha(alpha)
    , m_breslow(breslow)
  {
    eigen_assert(time.size() == x.rows());
    eigen_assert(event.size() == x.rows());
    eigen_assert((event.array() >= 0).all());
    eigen_assert((event.array() <= 1).all());
  }

  // Function to compute negative partial log-likelihood
  template<typename VectorDerived>
  Scalar nlog_likelihood(const Eigen::MatrixBase<VectorDerived>& w) const;

  // Function to update gradient and hessian
  template<typename DerivedA, typename DerivedB, typename DerivedC>
  void update(const Eigen::MatrixBase<DerivedA>& w,
              Eigen::MatrixBase<DerivedB>& gradient,
              Eigen::MatrixBase<DerivedC>& hessian,
              Scalar offset = 0);

private:
  const FloatMatrix m_x;
  const FloatVector m_time;
  const IntVector m_event;
  const FloatVector m_alpha;
  bool m_breslow;
};

// Function to compute negative partial log-likelihood
template<typename _S, typename _I>
template<typename VectorDerived>
typename CoxPHSolver<_S, _I>::Scalar
CoxPHSolver<_S, _I>::nlog_likelihood(
  const Eigen::MatrixBase<VectorDerived>& w) const
{
  FloatVector xw = m_x * w;
  const Index n_samples = m_x.rows();
  Scalar loss = 0.0;
  Scalar risk_set = 0.0;
  Index k = 0;

  while (k < n_samples) {
    Scalar ti = m_time[k];
    Scalar numerator = 0.0;
    Index n_events = 0;
    Scalar risk_set2 = 0.0;

    while (k < n_samples && ti == m_time[k]) {
      if (m_event(k) == 1) {
        numerator += xw[k];
        risk_set2 += exp(xw[k]);
        n_events += 1;
      } else {
        risk_set += exp(xw[k]);
      }
      k += 1;
    }

    if (n_events > 0) {
      if (m_breslow) {
        risk_set += risk_set2;
        loss -= (numerator - n_events * log(risk_set)) / n_samples;
      } else {
        numerator /= n_events;
        for (Index i = 0; i < n_events; ++i) {
          risk_set += risk_set2 / n_events;
          loss -= (numerator - log(risk_set)) / n_samples;
        }
      }
    }
  }
  // add regularization term to log-likelihood
  loss += (m_alpha.array() * w.array().square()).sum() / (2.0 * n_samples);
  return loss;
}

// Function to update gradient and hessian
template<typename _S, typename _I>
template<typename DerivedA, typename DerivedB, typename DerivedC>
void
CoxPHSolver<_S, _I>::update(const Eigen::MatrixBase<DerivedA>& w,
                            Eigen::MatrixBase<DerivedB>& gradient,
                            Eigen::MatrixBase<DerivedC>& hessian,
                            Scalar offset)
{
  typedef typename DerivedB::PlainObject Vector;
  typedef typename DerivedC::PlainObject Matrix;

  Vector exp_xw = (m_x * w /*+ offset*/).array().exp().matrix();
  const Index n_samples = m_x.rows();
  const Index n_features = m_x.cols();
  Scalar inv_n_samples = 1.0 / n_samples;
  Scalar risk_set = 0.0;
  Vector risk_set_x = Vector::Zero(n_features);
  Matrix risk_set_xx = Matrix::Zero(n_features, n_features);
  Index k = 0;

  while (k < n_samples) {
    Scalar ti = m_time(k);
    Index n_events = 0;
    Vector numerator = Vector::Zero(n_features);
    Scalar risk_set2 = 0.0;
    Vector risk_set_x2 = Vector::Zero(n_features);
    Matrix risk_set_xx2 = Matrix::Zero(n_features, n_features);

    while (k < n_samples && ti == m_time[k]) {
      Vector xk = m_x.row(k);
      Matrix xx = xk * xk.transpose();

      if (m_event[k] == 1) {
        numerator += xk;
        risk_set2 += exp_xw[k];
        risk_set_x2 += exp_xw[k] * xk;
        risk_set_xx2 += exp_xw[k] * xx;
        n_events += 1;
      } else {
        risk_set += exp_xw[k];
        risk_set_x += exp_xw[k] * xk;
        risk_set_xx += exp_xw[k] * xx;
      }
      k += 1;
    }

    if (n_events > 0) {
      if (m_breslow) {
        risk_set += risk_set2;
        risk_set_x += risk_set_x2;
        risk_set_xx += risk_set_xx2;

        Vector z = risk_set_x / risk_set;
        gradient -= (numerator - n_events * z) * inv_n_samples;

        Matrix a = risk_set_xx / risk_set;
        Matrix b = z * z.transpose();

        hessian += n_events * (a - b) * inv_n_samples;
      } else {
        numerator /= n_events;
        for (Index i = 0; i < n_events; ++i) {
          risk_set += risk_set2 / n_events;
          risk_set_x += risk_set_x2 / n_events;
          risk_set_xx += risk_set_xx2 / n_events;

          Vector z = risk_set_x / risk_set;
          gradient -= (numerator - z) * inv_n_samples;

          Matrix a = risk_set_xx / risk_set;
          Matrix b = z * z.transpose();

          hessian += (a - b) * inv_n_samples;
        }
      }
    }
  }
  gradient += inv_n_samples * m_alpha.cwiseProduct(w);
  hessian.diagonal() += m_alpha * inv_n_samples;
}

template<typename FloatType>
std::uint64_t
coxph_fit(FloatType* X_ptr,
          std::uint8_t* event_ptr,
          FloatType* time_ptr,
          FloatType* w_ptr,
          FloatType* alpha_ptr,
          std::uint64_t n_samples,
          std::uint64_t n_features,
          FloatType tol,
          std::uint64_t n_iter,
          bool breslow)
{
  typedef CoxPHSolver<FloatType, std::uint8_t> SolverType;
  typedef typename SolverType::IntVector VectorXuint8;
  typedef typename SolverType::FloatVector VectorXd;
  typedef typename SolverType::FloatMatrix MatrixXd;

  // Map raw pointers to Eigen objects
  Eigen::Map<MatrixXd> X(X_ptr, n_samples, n_features);
  Eigen::Map<VectorXuint8> event(event_ptr, n_samples);
  Eigen::Map<VectorXd> time(time_ptr, n_samples);
  Eigen::Map<VectorXd> w(w_ptr, n_features);
  Eigen::Map<VectorXd> alpha(alpha_ptr, n_features);

  VectorXd w_prev = w;
  FloatType loss = std::numeric_limits<double>::infinity();
  FloatType loss_new;
  VectorXd gradient = VectorXd::Zero(n_features);
  MatrixXd hessian = MatrixXd::Zero(n_features, n_features);

  SolverType solver(X, event, time, alpha, breslow);
  std::uint64_t i;
  for (i = 0; i < n_iter; ++i) {
    gradient.setZero();
    hessian.setZero();
    solver.update(w, gradient, hessian);
    VectorXd delta = hessian.llt().solve(gradient);

    if (!delta.array().isFinite().all()) {
      throw std::runtime_error(
        "search direction contains NaN or infinite values");
    }

    VectorXd w_new = w - delta;
    loss_new = solver.nlog_likelihood(w_new);
    std::cout << "ITER " << i << " " << w << "\n==\n" << gradient << std::endl;

    if (loss_new > loss) {
      // perform step-halving if negative log-likelihood does not decrease
      w = (w_prev + w) / 2;
      loss = solver.nlog_likelihood(w);
      std::cout << "HALF" << " " << loss << " " << loss_new << std::endl;
      continue;
    }

    w_prev = w;
    w = w_new;

    FloatType res = std::abs(1 - (loss_new / loss));
    std::cout << "\t==> " << res << std::endl;
    if (res < tol) {
      break;
    }

    loss = loss_new;
  }

  return i;
}

#endif