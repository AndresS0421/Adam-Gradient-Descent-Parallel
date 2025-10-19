#include <vector>
#include <cmath>
#include "optimizer.hpp"

AdamOptimizer::AdamOptimizer(int d, double alpha_, double beta1_,
                             double beta2_, double eps_)
    : alpha(alpha_), beta1(beta1_), beta2(beta2_), eps(eps_),
      m(d, 0.0), v(d, 0.0) {}

void AdamOptimizer::step(std::vector<double> &w,
                         const std::vector<double> &grad, int t) {
    const int d = w.size();
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < d; ++j) {
        m[j] = beta1 * m[j] + (1 - beta1) * grad[j];
        v[j] = beta2 * v[j] + (1 - beta2) * grad[j] * grad[j];
        double m_hat = m[j] / (1 - std::pow(beta1, t));
        double v_hat = v[j] / (1 - std::pow(beta2, t));
        w[j] -= alpha * m_hat / (std::sqrt(v_hat) + eps);
    }
}