#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sys/stat.h>
#include <chrono>
#include "dataset.hpp"
#include "optimizer.hpp"
#include "adam_parallel.hpp"

int main() {
    mkdir("results", 0777);
    std::ofstream log("results/experiments.csv");
    std::ofstream timing_log("results/timing.csv");
    log << "method,lr,step,x,y,loss\n";
    timing_log << "method,lr,execution_time_ms\n";

    std::vector<double> learning_rates;
    learning_rates.push_back(0.1);
    learning_rates.push_back(0.01);
    learning_rates.push_back(0.001);
    const int steps = 100;
    const int n_points = 50;   // Reduced points for higher dimensions
    const int n_params = 20;   // Increased from 2 to 20 parameters
    unsigned seed = 181763002;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (size_t i = 0; i < learning_rates.size(); ++i) {
        double lr = learning_rates[i];
        // ---------- Adam Optimizer (Sequential) ----------
        auto start_adam = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < n_points; ++k) {
            // Initialize random parameters
            std::vector<double> w(n_params);
            for (int i = 0; i < n_params; ++i) {
                w[i] = dist(rng);
            }
            
            AdamOptimizer opt(n_params, lr);
            
            for (int t = 1; t <= steps; ++t) {
                // Compute gradient
                std::vector<double> grad_vec = high_dim_grad(w);
                
                // Use sequential Adam step
                opt.step(w, grad_vec, t);
                
                if (k == 0) {  // Only log first point to avoid too much data
                    log << "Adam_Sequential," << lr << "," << t << "," << w[0] << "," << w[1] << "," << high_dim_objective(w) << "\n";
                }
            }
        }
        auto end_adam = std::chrono::high_resolution_clock::now();
        auto duration_adam = std::chrono::duration_cast<std::chrono::milliseconds>(end_adam - start_adam);
        timing_log << "Adam_Sequential," << lr << "," << duration_adam.count() << "\n";

        // ---------- Parallel Adam Optimizer ----------
        auto start_adam_par = std::chrono::high_resolution_clock::now();
        
        // Process ALL points in parallel (batch processing)
        std::vector<std::vector<double> > all_points(n_points);
        std::vector<AdamOptimizer> all_optimizers(n_points, AdamOptimizer(n_params, lr));
        
        // Initialize all points
        for (int k = 0; k < n_points; ++k) {
            all_points[k].resize(n_params);
            for (int i = 0; i < n_params; ++i) {
                all_points[k][i] = dist(rng);
            }
        }
        
        for (int t = 1; t <= steps; ++t) {
            // Process all points in parallel
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n_points; ++k) {
                // Compute gradient for this point
                std::vector<double> grad_vec = high_dim_grad(all_points[k]);
                
                // Use Adam step
                all_optimizers[k].step(all_points[k], grad_vec, t);
            }
            
            // Log first point only
            if (t % 10 == 0) {  // Log every 10 steps to avoid too much data
                log << "Adam_Parallel," << lr << "," << t << "," << all_points[0][0] << "," << all_points[0][1] << "," << high_dim_objective(all_points[0]) << "\n";
            }
        }
        auto end_adam_par = std::chrono::high_resolution_clock::now();
        auto duration_adam_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_adam_par - start_adam_par);
        timing_log << "Adam_Parallel," << lr << "," << duration_adam_par.count() << "\n";
    }

    log.close();
    timing_log.close();
    std::cout << "✅ Experiment complete. Results saved to results/experiments.csv\n";
    std::cout << "✅ Timing data saved to results/timing.csv\n";
}