import os
import numpy as np
import cv2
import pykitti
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions
# -----------------------------

def load_stereo_images(dataset, idx):
    """Load the left and right stereo images for a given index."""
    img_left_gray = np.array(dataset.get_cam0(idx))
    img_right_gray = np.array(dataset.get_cam1(idx))

    # Convert to grayscale
    # img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    # img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    return img_left_gray, img_right_gray

def feature_detection(img, max_feats=1000):
    """Detect corners/features in the image using GFTT (Good Features to Track)."""
    # Parameters can be tuned as needed
    corners = cv2.goodFeaturesToTrack(img, max_feats, 0.01, 10)
    return corners

def feature_tracking(img1, img2, points1):
    """Track features from img1 to img2 using optical flow (LK)."""
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    points2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **lk_params)
    # st = st.reshape(st.shape[0])
    
    # # Filter only good points
    # points1_good = points1[st == 1]
    # points2_good = points2[st == 1]

    trackable = st.astype(bool)
    points1_good = points1[trackable]
    points2_good = points2[trackable]
    points1_good = points1_good.reshape(-1, 2)
    points2_good = points2_good.reshape(-1, 2)


    return points1_good, points2_good

def compute_disparity(img_left, img_right):
    """Compute disparity using StereoSGBM."""
    # Parameters can be tuned
    block_size = 5
    num_disp = 128  # must be divisible by 16
    min_disp = 0
    matcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                    numDisparities=num_disp,
                                    blockSize=block_size,
                                    P1=8 * block_size**2,
                                    P2=32 * block_size**2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=2)
    disparity = matcher.compute(img_left, img_right).astype(np.float32) / 16.0
    return disparity

def triangulate_points(q_l, q_r, P_l, P_r):
    """Triangulate 3D points from matched left-right keypoints.
       q_l, q_r are Nx2 arrays of corresponding points.
    """
    # Convert to homogeneous 2xN
    q_l_h = q_l.T
    q_r_h = q_r.T
    # Triangulate
    Q = cv2.triangulatePoints(P_l, P_r, q_l_h, q_r_h)
    Q = (Q / Q[3])[:3].T  # N x 3
    return Q

def filter_points_by_disparity(q_l, disparity, min_disp=1.0, max_disp=256.0):
    """Filter matched points by valid disparity range."""
    q_idx = q_l.astype(int)
    h, w = disparity.shape
    valid_mask = (q_idx[:,0] >= 0) & (q_idx[:,0] < w) & (q_idx[:,1] >= 0) & (q_idx[:,1] < h)
    q_idx = q_idx[valid_mask]
    q_l = q_l[valid_mask]

    disp_values = disparity[q_idx[:,1], q_idx[:,0]]
    valid_disp = (disp_values > min_disp) & (disp_values < max_disp)

    return q_l[valid_disp], disp_values[valid_disp]

# -----------------------------
# Main VO pipeline
# -----------------------------

def main():
    # Set dataset path and sequence
    basedir = 'dataset/kitti'
    date = '2011_09_26'
    drive = '0001'
    dataset = pykitti.raw(basedir, date, drive)

    # Projection matrices from KITTI raw data (rectified)
    # P_rect_00 and P_rect_10 are available from dataset.calib
    P_l = dataset.calib.P_rect_00
    P_r = dataset.calib.P_rect_10

    # Ground truth poses (T_w_imu)
    gt_poses = [oxts[1] for oxts in dataset.oxts]

    # We'll store trajectory
    estimated_path = []
    gt_path = []

    # Initial pose
    current_pose = np.eye(4)

    # Load first frame
    img_l_prev, img_r_prev = load_stereo_images(dataset, 0)
    disparity_prev = compute_disparity(img_l_prev, img_r_prev)
    kp_prev = feature_detection(img_l_prev)
    
    # Convert keypoints to Nx2 float32
    kp_prev = kp_prev.reshape(-1, 2)

    for i in tqdm(range(1, len(dataset)), unit="frame"):
        img_l, img_r = load_stereo_images(dataset, i)
        disparity = compute_disparity(img_l, img_r)

        # Track keypoints from previous frame to current frame (left image only)
        kp_prev_tracked, kp_curr = feature_tracking(img_l_prev, img_l, kp_prev.reshape(-1,1,2))
        
        # Filter out points with no good matches
        if len(kp_curr) < 8:
            # If not enough points to estimate pose, redetect features
            kp_curr = feature_detection(img_l)
            if kp_curr is None or len(kp_curr) < 8:
                # Move on if we cannot track
                img_l_prev, img_r_prev = img_l, img_r
                disparity_prev = disparity
                kp_prev = kp_curr if kp_curr is not None else np.empty((0,2))
                current_gt_pose = gt_poses[i]
                gt_path.append((current_gt_pose[0,3], current_gt_pose[2,3]))
                estimated_path.append((current_pose[0,3], current_pose[2,3]))
                continue
            kp_curr = kp_curr.reshape(-1,2)

        # For each matched kp_curr, find the corresponding disparity and get its 3D point in the previous frame
        q_l_prev, disp_prev = filter_points_by_disparity(kp_prev_tracked, disparity_prev)
        q_l_curr, disp_curr = filter_points_by_disparity(kp_curr, disparity)
        
        # Match the number of points: we need corresponding sets
        # We'll use the order they appear; in a robust system you'd match them by index
        # For simplicity, only use points that exist in both sets after filtering.
        # A real system would track indexes more carefully.
        # Here we find correspondences by nearest matches:
        # This might be simplistic. Ideally, you'd keep track of indices from optical flow.
        
        # Find intersection via a kd-tree or naive method
        # We'll just do a simple nearest-neighbor since we know they came from optical flow:
        # In practice, q_l_prev_tracked and q_l_curr correspond to each other by optical flow.
        
        # We must ensure q_l_prev_tracked are the same as we used in q_l_prev...
        # Wait, we must ensure consistency:
        #   kp_prev_tracked, kp_curr are matched pairs (frame-to-frame)
        #   q_l_prev and q_l_curr come from filtering disparity. Let's intersect by coordinates.
        # For simplicity, let's assume all kp_prev_tracked are in q_l_prev, and all kp_curr are in q_l_curr.
        # We'll intersect by rounding coordinates to nearest pixel.
        
        def round_coords(q):
            return np.round(q).astype(int)
        
        prev_set = {tuple(pt) for pt in round_coords(q_l_prev)}
        curr_set = {tuple(pt) for pt in round_coords(q_l_curr)}
        common = np.array(list(prev_set.intersection(curr_set)))
        if len(common) < 8:
            # Not enough common points; skip pose estimation for this frame
            img_l_prev, img_r_prev = img_l, img_r
            disparity_prev = disparity
            kp_prev = kp_curr
            current_gt_pose = gt_poses[i]
            gt_path.append((current_gt_pose[0,3], current_gt_pose[2,3]))
            estimated_path.append((current_pose[0,3], current_pose[2,3]))
            continue
        
        # Get indices of these common points in prev and curr arrays
        # def get_indices(q, pts):
        #     # q: Nx2 array
        #     # pts: set of tuples
        #     idx_list = []
        #     q_rounded = round_coords(q)
        #     pt_map = {tuple(k): idx for idx, k in enumerate(q_rounded)}
        #     for p in pts:
        #         if p in pt_map:
        #             idx_list.append(pt_map[p])
        #     return np.array(idx_list)
        
        # Helper function to round coordinates
        def round_coords(q):
            # Round to a desired precision (e.g., integers)
            return np.round(q).astype(int)

        # Fixed get_indices function
        def get_indices(q, pts):
            """
            Get indices of common points in q and pts.
            
            Parameters:
            q: Nx2 numpy array of points
            pts: Set of tuples representing common points
            
            Returns:
            idx_list: Indices of q points that match pts
            """
            idx_list = []
            
            # Round coordinates and convert to tuples for hashing
            q_rounded = round_coords(q)
            pt_map = {tuple(k): idx for idx, k in enumerate(q_rounded)}

            # Ensure all points in pts are tuples
            pts_tuples = {tuple(p) for p in pts}

            # Match points
            for p in pts_tuples:
                if p in pt_map:
                    idx_list.append(pt_map[p])
            
            return np.array(idx_list)

        idx_prev = get_indices(q_l_prev, common)
        idx_curr = get_indices(q_l_curr, common)
        
        q_l_prev_common = q_l_prev[idx_prev]
        q_l_curr_common = q_l_curr[idx_curr]
        
        # Now we have matching points in consecutive frames.
        # Triangulate 3D points from the previous frame.
        q_r_prev = np.copy(q_l_prev_common)
        q_r_prev[:,0] = q_r_prev[:,0] - disp_prev[idx_prev]  # x - disparity
        Q_prev = triangulate_points(q_l_prev_common, q_r_prev, P_l, P_r)

        # Triangulate 3D points from the current frame
        q_r_curr = np.copy(q_l_curr_common)
        q_r_curr[:,0] = q_r_curr[:,0] - disp_curr[idx_curr]
        Q_curr = triangulate_points(q_l_curr_common, q_r_curr, P_l, P_r)

        # Estimate the pose from Q_prev (3D points in prev frame coords) and q_l_curr_common (2D points in current frame)
        # But we have Q_prev as coordinates in prev frame camera system. We want R,t that brings Q_prev to current frame image coords.
        # Actually, we can use Q_prev directly with solvePnPRansac if we assume previous pose as reference.
        # The new pose transforms Q_prev to Q_curr. Let's just assume Q_prev is in the previous camera frame.
        # The 2D points for solvePnP must correspond to the current frame image coordinates q_l_curr_common.
        
        # Use the left camera intrinsics from P_l:
        K = P_l[:3,:3]
        
        # We'll use solvePnPRansac to find R,t that aligns Q_prev to q_l_curr_common in the current frame.
        # Q_prev are objectPoints (3D) in previous camera frame, q_l_curr_common are imagePoints (2D) in current frame.
        dist_coeffs = np.zeros(4)
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(Q_prev, q_l_curr_common, K, dist_coeffs,
                                                         flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not retval:
            # If pose could not be estimated, just skip
            img_l_prev, img_r_prev = img_l, img_r
            disparity_prev = disparity
            kp_prev = kp_curr
            current_gt_pose = gt_poses[i]
            gt_path.append((current_gt_pose[0,3], current_gt_pose[2,3]))
            estimated_path.append((current_pose[0,3], current_pose[2,3]))
            continue
        
        # Convert rvec,tvec into a transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = tvec.squeeze()
        
        # Update current pose (poses are cumulative)
        # current_pose is in world coordinates. The first frame is identity.
        # The new pose: current_pose = current_pose * T
        # T transforms points from previous frame to current frame.
        # If we keep camera fixed at origin and transform world, we do inverse.
        # But we want camera pose in a fixed world frame. The first pose is identity.
        # The found T is camera_{prev} to camera_{curr}. So:
        # camera_{curr} = camera_{prev} * T
        # If camera_{prev} is current_pose, then:
        current_pose = current_pose @ np.linalg.inv(T)

        # Store trajectory
        current_gt_pose = gt_poses[i]
        gt_path.append((current_gt_pose[0,3], current_gt_pose[2,3]))
        estimated_path.append((current_pose[0,3], current_pose[2,3]))
        
        # Prepare next iteration
        img_l_prev, img_r_prev = img_l, img_r
        disparity_prev = disparity
        kp_prev = kp_curr

    # Plot trajectories
    gt_path = np.array(gt_path)
    estimated_path = np.array(estimated_path)

    plt.figure(figsize=(10,5))
    plt.plot(gt_path[:,0], gt_path[:,1], label="Ground Truth")
    plt.plot(estimated_path[:,0], estimated_path[:,1], label="Estimated")
    plt.title("Stereo Visual Odometry Trajectory")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
