#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <cmath>
#include <omp.h>

class AdamOptimizer {
public:
    AdamOptimizer(int d, double alpha = 0.01, double beta1 = 0.9,
                  double beta2 = 0.999, double eps = 1e-8);
    void step(std::vector<double> &w, const std::vector<double> &grad, int t);
private:
    double alpha, beta1, beta2, eps;
    std::vector<double> m, v;
};

#endif