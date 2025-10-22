# Adam Gradient Descent Parallel

A C++ implementation comparing sequential and parallel Adam optimization algorithms using OpenMP for high-dimensional optimization problems.

## 📋 Overview

This project implements and compares:
- **Adam Sequential**: Traditional Adam optimizer processing points one by one
- **Adam Parallel**: Parallel Adam optimizer using OpenMP to process multiple points simultaneously

The implementation uses a high-dimensional Rosenbrock-like objective function to demonstrate the performance benefits of parallelization.

## 🚀 Features

- **OpenMP Parallelization**: Multi-threaded optimization using OpenMP
- **High-Dimensional Optimization**: Supports optimization problems with 20+ parameters
- **Performance Comparison**: Detailed timing analysis between sequential and parallel implementations
- **Data Visualization**: Python scripts for plotting results and performance metrics
- **Cross-Platform**: Compatible with macOS, Linux, and Windows

## 📁 Project Structure

```
├── main.cpp              # Main program with timing experiments
├── optimizer.hpp         # Adam optimizer class definition
├── optimizer.cpp         # Adam optimizer implementation with OpenMP
├── adam_parallel.hpp     # Parallel Adam optimizer for batch processing
├── dataset.hpp           # Objective functions and gradients
├── plot_results.py       # Python visualization script
├── results/              # Generated results directory
│   ├── experiments.csv   # Optimization trajectory data
│   ├── timing.csv        # Execution time measurements
│   └── plots/            # Generated visualization plots
└── README.md             # This file
```

## 🛠️ Requirements

### C++ Compilation
- **GCC 9+** or **Clang 10+** with OpenMP support
- **OpenMP** library
- **C++17** standard support

### Python Visualization
- Python 3.7+
- pandas
- matplotlib
- seaborn

## 🔧 Installation & Compilation

### macOS (using Homebrew)
```bash
# Install OpenMP
brew install libomp

# Compile with Clang
clang++ -Xpreprocessor -fopenmp -lomp -O2 -std=c++17 -o main main.cpp optimizer.cpp

# Or compile with GCC
g++-15 -fopenmp -O2 -std=c++17 -o main main.cpp optimizer.cpp
```

### Linux
```bash
# Install OpenMP (Ubuntu/Debian)
sudo apt-get install libomp-dev

# Compile
g++ -fopenmp -O2 -std=c++17 -o main main.cpp optimizer.cpp
```

## 🏃‍♂️ Usage

### Run the optimization experiments
```bash
./main
```

This will generate:
- `results/experiments.csv`: Optimization trajectories
- `results/timing.csv`: Execution time measurements

### Generate visualizations
```bash
python plot_results.py
```

This creates plots in `results/plots/`:
- Loss curves comparison
- Execution time analysis
- Performance metrics

## 📊 Results

The program compares performance across different learning rates:
- **0.1**: High learning rate
- **0.01**: Medium learning rate  
- **0.001**: Low learning rate

### Expected Performance Gains
- **Parallel processing** typically shows 2-4x speedup on multi-core systems
- **Higher dimensional problems** (20+ parameters) benefit more from parallelization
- **Batch processing** of multiple optimization points simultaneously

## 🔬 Technical Details

### Adam Optimizer
- **Adaptive learning rates** for each parameter
- **Momentum** (β₁ = 0.9) for gradient smoothing
- **RMSprop** (β₂ = 0.999) for learning rate adaptation
- **Bias correction** for unbiased estimates

### Parallelization Strategy
- **OpenMP parallel for loops** for parameter updates
- **Static scheduling** for even work distribution
- **Thread-safe** operations with private variables
- **Batch processing** for multiple optimization points

### Objective Function
High-dimensional Rosenbrock function:
```
f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²]
```
- **Non-convex** optimization landscape
- **Multiple local minima**
- **Challenging** for gradient-based methods

## 📈 Performance Analysis

The visualization script provides:
- **Loss curves** showing convergence behavior
- **Execution time** comparison between methods
- **Speedup analysis** for parallel implementation
- **Learning rate** sensitivity analysis

## 🎯 Educational Value

This project demonstrates:
- **Parallel programming** concepts with OpenMP
- **Optimization algorithms** implementation
- **Performance measurement** and analysis
- **Scientific computing** best practices
- **Data visualization** for research results

## 📝 License

This project is created for educational purposes as part of a Parallel Programming course.

## 👨‍💻 Authors

**Andrés Solís**  
**Alonso Flores**  
**Joel García**

Universidad de Colima - 7° Semestre\
Programación Paralela - 2° Parcial

---

*For questions or issues, please refer to the course materials or contact the instructor.*