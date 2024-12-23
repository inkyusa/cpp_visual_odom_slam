import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.visualization import plotting

def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def load_trajectory(file_path):
    """
    Loads trajectory data from a file.

    Parameters
    ----------
    file_path : str
        Path to the trajectory file.

    Returns
    -------
    positions : list of tuples
        List of 3D positions (x, y, z).
    """
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            positions.append((x, y, z))
    return positions



# load_poses(os.path.join(data_dir,"poses.txt"))
def main():
    # Load trajectories
    estimated_trajectory_path = "src/trajectory.txt"
    ground_truth_path = "dataset/KITTI_sequence_2/poses.txt"  # Replace with the correct ground truth file path if available.
    ground_truth_positions = load_poses(ground_truth_path)
    estimated_positions = load_trajectory(estimated_trajectory_path)
    gt_path = []
    estimated_path = []
    for gt_pose, esti in zip(ground_truth_positions, estimated_positions):
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((esti[0], esti[2]))
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out= "src/traj.html")

if __name__ == "__main__":
    main()
