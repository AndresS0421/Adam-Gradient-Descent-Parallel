#ifndef ADAM_PARALLEL_HPP
#define ADAM_PARALLEL_HPP

#include <vector>
#include <utility>
#include <omp.h>
#include "dataset.hpp"
#include "optimizer.hpp"

// Parallel Adam optimizer that processes multiple points simultaneously
class ParallelAdamOptimizer {
public:
    ParallelAdamOptimizer(int d, double alpha = 0.01, double beta1 = 0.9,
                         double beta2 = 0.999, double eps = 1e-8);
    
    // Process multiple points in parallel
    void parallel_step(std::vector<std::vector<double> >& points,
                      const std::vector<std::vector<double> >& gradients,
                      int t);
    
    // Get the best point (lowest loss) from the parallel batch
    struct Point2D {
        double x, y;
    };
    Point2D get_best_point(const std::vector<std::vector<double> >& points,
                          double (*func)(double, double)) const;

private:
    double alpha, beta1, beta2, eps;
    std::vector<std::vector<double> > m_batch, v_batch;  // Momentum and velocity for each point
    int d;  // Dimension
};

// Constructor
ParallelAdamOptimizer::ParallelAdamOptimizer(int d_, double alpha_, double beta1_,
                                           double beta2_, double eps_)
    : alpha(alpha_), beta1(beta1_), beta2(beta2_), eps(eps_), d(d_) {
    // Initialize momentum and velocity for batch processing
    // We'll resize these when we know the batch size
}

// Parallel step for multiple points
void ParallelAdamOptimizer::parallel_step(std::vector<std::vector<double> >& points,
                                         const std::vector<std::vector<double> >& gradients,
                                         int t) {
    int batch_size = points.size();
    
    // Resize momentum and velocity vectors if needed
    if (m_batch.size() != batch_size) {
        m_batch.resize(batch_size, std::vector<double>(d, 0.0));
        v_batch.resize(batch_size, std::vector<double>(d, 0.0));
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < d; ++j) {
            // Update momentum and velocity for point i
            m_batch[i][j] = beta1 * m_batch[i][j] + (1 - beta1) * gradients[i][j];
            v_batch[i][j] = beta2 * v_batch[i][j] + (1 - beta2) * gradients[i][j] * gradients[i][j];
            
            // Bias correction
            double m_hat = m_batch[i][j] / (1 - std::pow(beta1, t));
            double v_hat = v_batch[i][j] / (1 - std::pow(beta2, t));
            
            // Update parameters
            points[i][j] -= alpha * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

// Get the best point from the batch
ParallelAdamOptimizer::Point2D ParallelAdamOptimizer::get_best_point(
    const std::vector<std::vector<double> >& points,
    double (*func)(double, double)) const {
    
    double best_loss = std::numeric_limits<double>::max();
    Point2D best_point = {0.0, 0.0};
    
    for (const auto& point : points) {
        double loss = func(point[0], point[1]);
        if (loss < best_loss) {
            best_loss = loss;
            best_point = {point[0], point[1]};
        }
    }
    
    return best_point;
}

#endif
