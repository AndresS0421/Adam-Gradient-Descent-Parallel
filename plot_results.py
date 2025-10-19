import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create plots directory
os.makedirs("results/plots", exist_ok=True)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load experiment data
df = pd.read_csv("results/experiments.csv")
timing_df = pd.read_csv("results/timing.csv")

# --- 1️⃣ LOSS CURVES ---
plt.figure(figsize=(12, 5))
for i, method_group in enumerate(["Adam_Sequential", "Adam_Parallel"], 1):
    plt.subplot(1, 2, i)
    for lr, group in df[df.method == method_group].groupby("lr"):
        mean_loss = group.groupby("step")["loss"].mean()
        plt.plot(mean_loss.index, mean_loss.values, label=f"{method_group} α={lr}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{method_group} - Convergence")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/loss_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 2️⃣ CONTOUR + TRAJECTORIES ---
x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

plt.figure(figsize=(12, 8))
plt.contourf(X, Y, Z, levels=50, cmap="gray", alpha=0.8)
plt.colorbar(label="Himmelblau Function Value")

colors = {
    "Adam_Sequential": "cyan",
    "Adam_Parallel": "red"
}

# Plot trajectories for each method and learning rate
for method in ["Adam_Sequential", "Adam_Parallel"]:
    method_data = df[df.method == method]
    for lr in method_data.lr.unique():
        group = method_data[method_data.lr == lr]
        # Plot trajectory
        plt.plot(group["x"], group["y"], lw=1.5, color=colors[method], alpha=0.7, 
                label=f"{method} α={lr}" if lr == 0.01 else "")
        # Mark final position
        plt.scatter(group["x"].iloc[-1], group["y"].iloc[-1], s=50, color=colors[method], 
                   marker='*', edgecolor='black', linewidth=0.5)

plt.title("Optimizer Trajectories on Himmelblau Function", fontsize=16, fontweight='bold')
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/trajectories.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 3️⃣ EXECUTION TIME BY LEARNING RATE ---
plt.figure(figsize=(12, 8))
timing_pivot = timing_df.pivot(index='method', columns='lr', values='execution_time_ms')
timing_pivot.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title("Execution Time by Method and Learning Rate")
plt.xlabel("Method")
plt.ylabel("Execution Time (ms)")
plt.legend(title="Learning Rate")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/execution_time_by_lr.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 4️⃣ AVERAGE EXECUTION TIME COMPARISON ---
plt.figure(figsize=(10, 6))
methods = ['Adam_Sequential', 'Adam_Parallel']
colors = ['cyan', 'red']
x_pos = range(len(methods))

# Calculate average times across all learning rates
avg_times = []
for method in methods:
    avg_time = timing_df[timing_df['method'] == method]['execution_time_ms'].mean()
    avg_times.append(avg_time)

bars = plt.bar(x_pos, avg_times, color=colors, alpha=0.7)
plt.title("Average Execution Time Comparison")
plt.xlabel("Method")
plt.ylabel("Average Time (ms)")
plt.xticks(x_pos, methods, rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, time in zip(bars, avg_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{time:.1f}ms', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("results/plots/average_execution_time.png", dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("EXECUTION TIME ANALYSIS")
print("=" * 60)

print("\n📊 Execution Times (ms):")
print(timing_df.pivot(index='method', columns='lr', values='execution_time_ms').round(2))

print("\n⚡ Speedup Analysis (Sequential vs Parallel):")
for lr in timing_df['lr'].unique():
    adam_seq_time = timing_df[(timing_df['method'] == 'Adam_Sequential') & (timing_df['lr'] == lr)]['execution_time_ms'].iloc[0]
    adam_par_time = timing_df[(timing_df['method'] == 'Adam_Parallel') & (timing_df['lr'] == lr)]['execution_time_ms'].iloc[0]
    adam_speedup = adam_seq_time / adam_par_time
    print(f"  LR {lr}: {adam_speedup:.2f}x speedup")

print("\n📈 Average Times:")
avg_times = timing_df.groupby('method')['execution_time_ms'].mean()
for method, time in avg_times.items():
    print(f"  {method}: {time:.2f}ms")

print("\n🎯 Performance Summary:")
seq_avg = timing_df[timing_df['method'] == 'Adam_Sequential']['execution_time_ms'].mean()
par_avg = timing_df[timing_df['method'] == 'Adam_Parallel']['execution_time_ms'].mean()
overall_speedup = seq_avg / par_avg
print(f"  Overall Speedup: {overall_speedup:.2f}x")
print(f"  Time Reduction: {((seq_avg - par_avg) / seq_avg * 100):.1f}%")

print("\n✅ All plots saved to results/plots/ directory!")