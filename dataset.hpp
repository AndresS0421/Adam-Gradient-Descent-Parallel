#ifndef DATASET_HPP
#define DATASET_HPP

#include <cmath>
#include <vector>

// High-dimensional objective function (Rosenbrock-like)
inline double high_dim_objective(const std::vector<double>& params) {
    double sum = 0.0;
    for (size_t i = 0; i < params.size() - 1; ++i) {
        double term1 = 100.0 * std::pow(params[i+1] - params[i]*params[i], 2);
        double term2 = std::pow(1.0 - params[i], 2);
        sum += term1 + term2;
    }
    return sum;
}

// Gradient of high-dimensional function
inline std::vector<double> high_dim_grad(const std::vector<double>& params) {
    std::vector<double> grad(params.size(), 0.0);
    
    for (size_t i = 0; i < params.size(); ++i) {
        if (i == 0) {
            // First parameter
            grad[i] = -400.0 * params[i] * (params[i+1] - params[i]*params[i]) - 2.0 * (1.0 - params[i]);
        } else if (i == params.size() - 1) {
            // Last parameter
            grad[i] = 200.0 * (params[i] - params[i-1]*params[i-1]);
        } else {
            // Middle parameters
            grad[i] = 200.0 * (params[i] - params[i-1]*params[i-1]) - 400.0 * params[i] * (params[i+1] - params[i]*params[i]) - 2.0 * (1.0 - params[i]);
        }
    }
    
    return grad;
}

#endif