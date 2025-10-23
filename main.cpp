#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <sys/stat.h>
#include <chrono>
#include <omp.h>
#include "dataset.hpp"
#include "optimizer.hpp"

int main() {
    mkdir("results", 0777);
    std::ofstream log("results/experiments.csv");
    std::ofstream timing_log("results/timing.csv");
    log << "method,lr,step,x,y,loss\n";
    timing_log << "method,lr,execution_time_ms\n";
    
    // Print OpenMP information
    std::cout << "🔧 OpenMP Configuration:" << std::endl;
    std::cout << "   Max threads: " << omp_get_max_threads() << std::endl;
    std::cout << "   Number of processors: " << omp_get_num_procs() << std::endl;
    std::cout << "   Thread limit: " << omp_get_thread_limit() << std::endl;
    std::cout << std::endl;

    std::vector<double> learning_rates;
    learning_rates.push_back(0.1);
    learning_rates.push_back(0.01);
    learning_rates.push_back(0.001);
    const int steps = 100;
    const int n_points = 1000;   // Increased points to make parallelization worthwhile
    const int n_params = 20;     // 20 parameters
    unsigned seed = 181763002;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (size_t i = 0; i < learning_rates.size(); ++i) {
        double lr = learning_rates[i];
        // ---------- Adam Optimizer (Sequential) ----------
        double start_adam = omp_get_wtime();
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
                
                // Use truly sequential Adam step (no OpenMP)
                opt.step(w, grad_vec, t);
                
                if (k == 0) {  // Only log first point to avoid too much data
                    log << "Adam_Sequential," << lr << "," << t << "," << w[0] << "," << w[1] << "," << high_dim_objective(w) << "\n";
                }
            }
        }
        double end_adam = omp_get_wtime();
        double duration_adam = (end_adam - start_adam) * 1000.0; // Convert to milliseconds
        timing_log << "Adam_Sequential," << lr << "," << duration_adam << "\n";
        std::cout << "Sequential (lr=" << lr << "): " << duration_adam << "ms" << std::endl;

        // ---------- Parallel Adam Optimizer ----------
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
        
        double start_adam_par = omp_get_wtime();
        for (int t = 1; t <= steps; ++t) {
            // Process all points in parallel - MEASURE ONLY THIS PART
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n_points; ++k) {
                // Compute gradient for this point
                std::vector<double> grad_vec = high_dim_grad(all_points[k]);
                
                // Use sequential Adam step (parallelization is in the outer loop)
                all_optimizers[k].step(all_points[k], grad_vec, t);
            }
            
            // Log first point only
            if (t % 10 == 0) {  // Log every 10 steps to avoid too much data
                log << "Adam_Parallel," << lr << "," << t << "," << all_points[0][0] << "," << all_points[0][1] << "," << high_dim_objective(all_points[0]) << "\n";
            }
        }
        double end_adam_par = omp_get_wtime();
        double duration_adam_par = (end_adam_par - start_adam_par) * 1000.0; // Convert to milliseconds
        timing_log << "Adam_Parallel," << lr << "," << duration_adam_par << "\n";
        std::cout << "Parallel (lr=" << lr << "): " << duration_adam_par << "ms" << std::endl;
        std::cout << "Speedup: " << duration_adam / duration_adam_par << "x" << std::endl;
        std::cout << std::endl;
    }

    log.close();
    timing_log.close();
    std::cout << "✅ Experiment complete. Results saved to results/experiments.csv\n";
    std::cout << "✅ Timing data saved to results/timing.csv\n";
}