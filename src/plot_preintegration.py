import numpy as np
import matplotlib.pyplot as plt

# Load data
gt = np.loadtxt('src/ground_truth.csv')
ord_int = np.loadtxt('src/ordinary_integration.csv')
preint = np.loadtxt('src/preintegration.csv')

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], label='Ground Truth', linewidth=2)
ax.plot(ord_int[:, 0], ord_int[:, 1], ord_int[:, 2], label='Ordinary Integration', linestyle='--')
ax.plot(preint[:, 0], preint[:, 1], preint[:, 2], color = 'red', label='Preintegration', linestyle=':', linewidth=2)

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.legend()
ax.set_title('Trajectory Comparison')

plt.show()
