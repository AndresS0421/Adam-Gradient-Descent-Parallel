#ifndef DATASET_HPP
#define DATASET_HPP

#include <cmath>
#include <utility>

// Himmelblau test function (for visualization)
inline double himmelblau(double x, double y) {
    return std::pow(x*x + y - 11, 2) + std::pow(x + y*y - 7, 2);
}

// Gradient of Himmelblau
struct Gradient {
    double dx, dy;
};

inline Gradient himmelblau_grad(double x, double y) {
    double dx = 4*x*(x*x + y - 11) + 2*(x + y*y - 7);
    double dy = 2*(x*x + y - 11) + 4*y*(x + y*y - 7);
    return {dx, dy};
}

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