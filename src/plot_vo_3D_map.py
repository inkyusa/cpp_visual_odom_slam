import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def load_data(file_path):
    frames = []
    with open(file_path, 'r') as file:
        current_frame = {"trajectory": [], "points": []}
        for line in file:
            if line.startswith("#"):
                if current_frame["trajectory"] or current_frame["points"]:
                    frames.append(current_frame)
                current_frame = {"trajectory": [], "points": []}
            elif line.startswith("TRAJ"):
                _, x, y, z = line.strip().split()
                current_frame["trajectory"].append([float(x), float(y), float(z)])
            elif line.startswith("MAP"):
                _, x, y, z = line.strip().split()
                current_frame["points"].append([float(x), float(y), float(z)])
        if current_frame["trajectory"] or current_frame["points"]:
            frames.append(current_frame)
    return frames

import open3d as o3d
import numpy as np

def visualize_with_open3d(frames):
    traj_points = []
    map_points = []

    for frame in frames:
        traj_points.extend(frame['trajectory'])
        map_points.extend(frame['points'])

    # Create point clouds
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(np.array(traj_points))
    traj_pcd.paint_uniform_color([0, 0, 1])  # Blue for trajectory

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(np.array(map_points))
    map_pcd.paint_uniform_color([1, 0, 0])  # Red for map points

    # Visualize
    o3d.visualization.draw_geometries([traj_pcd, map_pcd])

if __name__ == "__main__":
    file_path = "src/trajectory_and_points.txt"
    frames = load_data(file_path)
    visualize_with_open3d(frames)
