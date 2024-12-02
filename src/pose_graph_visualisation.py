import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
ground_truth = pd.read_csv("src/ground_truth_poses.csv")
initial_estimates = pd.read_csv("src/initial_estimates.csv")
optimized = pd.read_csv("src/optimized_poses.csv")

# Plot the trajectories
plt.figure(figsize=(10, 6))
plt.plot(ground_truth['x'], ground_truth['y'], label='Ground Truth', linestyle='-', marker='o')
plt.plot(initial_estimates['x'], initial_estimates['y'], label='Initial Estimates', linestyle='--', marker='x')
plt.plot(optimized['x'], optimized['y'], label='Optimized', linestyle='-.', marker='s')

# Add labels and legend
plt.title('Trajectory Comparison')
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.legend()
plt.grid(True)
plt.show()
