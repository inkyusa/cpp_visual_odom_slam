# plot_trajectory.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the trajectory file
positions = []
with open('src/trajectory.txt', 'r') as f:
    for line in f:
        x, y, z = map(float, line.strip().split())
        positions.append((x, y, z))

# Unpack positions into x, y, z lists
x_vals, y_vals, z_vals = zip(*positions)

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label='Estimated Trajectory', color='blue')
ax.scatter(x_vals[0], y_vals[0], z_vals[0], color='green', label='Start')
ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color='red', label='End')
ax.legend()

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('VO Estimated Trajectory')

plt.show()
