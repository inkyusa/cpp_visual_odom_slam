/*
g++ -std=c++17 -o ceres_11_stereo_calib_gpt ./ceres_11_stereo_calib_gpt.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

//input, fx, fy, cx, cy
//2d points (u, v) and their correspondecnes 3D points (X,Y,Z)

//output, refined intrinsics, pose, 3D points
#include <iostream>
#include <vector>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Observation struct to hold the data
struct Observation {
    int cam_idx;
    int pt_idx;
    double u;
    double v;
};

// Reprojection error functor for stereo calibration
struct ReprojectionErrorStereo {
    ReprojectionErrorStereo(double observed_u, double observed_v)
        : observed_u(observed_u), observed_v(observed_v) {}

    template <typename T>
    bool operator()(const T* const camera_intrinsics,
                    const T* const camera_rotation,
                    const T* const camera_translation,
                    const T* const point3D,
                    T* residuals) const {
        // Camera rotation (AngleAxis)
        T p[3];
        ceres::AngleAxisRotatePoint(camera_rotation, point3D, p);

        // Camera translation
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        // Perspective division
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Apply intrinsics
        T predicted_u = camera_intrinsics[0] * xp + camera_intrinsics[2];
        T predicted_v = camera_intrinsics[1] * yp + camera_intrinsics[3];

        // Compute residuals
        residuals[0] = predicted_u - T(observed_u);
        residuals[1] = predicted_v - T(observed_v);

        return true;
    }

    static ceres::CostFunction* Create(double observed_u, double observed_v) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorStereo, 2, 4, 3, 3, 3>(
            new ReprojectionErrorStereo(observed_u, observed_v)));
    }

    double observed_u;
    double observed_v;
};

int main() {
    // Step 1: Generate synthetic data
    const int num_cameras = 2; // Stereo cameras
    const int num_points = 10; // Number of 3D points

    // True intrinsic parameters for both cameras
    double true_intrinsics[num_cameras][4] = {
        {800.0, 800.0, 320.0, 240.0}, // fx, fy, cx, cy for camera 0
        {800.0, 800.0, 320.0, 240.0}  // fx, fy, cx, cy for camera 1
    };

    // True extrinsic parameters
    // Camera 0 is at the origin with identity rotation
    double true_cam_rot[num_cameras][3] = { {0, 0, 0}, {0.0, 0.0, 0.0} }; // AngleAxis rotation
    double true_cam_trans[num_cameras][3] = { {0, 0, 0}, {0.1, 0.0, 0.0} }; // Translation

    // Rotation between camera 0 and camera 1 (for stereo baseline)
    // Let's assume camera 1 is rotated around y-axis by 5 degrees
    double angle = ceres::DegToRad(5.0);
    true_cam_rot[1][1] = angle; // Rotation around y-axis

    // Generate random 3D points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> point_dist(-1.0, 1.0);

    double true_point3D[num_points][3];
    for (int i = 0; i < num_points; ++i) {
        true_point3D[i][0] = point_dist(generator);
        true_point3D[i][1] = point_dist(generator);
        true_point3D[i][2] = point_dist(generator) + 5.0; // Ensure points are in front of cameras
    }

    // Generate observations in both cameras
    std::vector<Observation> observations;
    std::normal_distribution<double> noise_dist(0.0, 1.0); // Pixel noise

    for (int cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
        double R[9];
        ceres::AngleAxisToRotationMatrix(true_cam_rot[cam_idx], R);

        for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
            double p[3];
            // Rotate and translate point
            ceres::AngleAxisRotatePoint(true_cam_rot[cam_idx], true_point3D[pt_idx], p);
            p[0] += true_cam_trans[cam_idx][0];
            p[1] += true_cam_trans[cam_idx][1];
            p[2] += true_cam_trans[cam_idx][2];

            // Project to image plane
            double xp = p[0] / p[2];
            double yp = p[1] / p[2];

            // Apply intrinsics
            double predicted_u = true_intrinsics[cam_idx][0] * xp + true_intrinsics[cam_idx][2];
            double predicted_v = true_intrinsics[cam_idx][1] * yp + true_intrinsics[cam_idx][3];

            // Add noise
            double noise_u = noise_dist(generator);
            double noise_v = noise_dist(generator);

            observations.push_back({ cam_idx, pt_idx, predicted_u + noise_u, predicted_v + noise_v });
        }
    }

    // Step 2: Set up optimization variables
    // Initial guesses for intrinsics (add some perturbation)
    double intrinsics[num_cameras][4];
    for (int cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
        intrinsics[cam_idx][0] = true_intrinsics[cam_idx][0] + 50.0; // fx
        intrinsics[cam_idx][1] = true_intrinsics[cam_idx][1] + 50.0; // fy
        intrinsics[cam_idx][2] = true_intrinsics[cam_idx][2] + 10.0; // cx
        intrinsics[cam_idx][3] = true_intrinsics[cam_idx][3] + 10.0; // cy
    }

    // Initial guesses for extrinsics
    double cam_rot[num_cameras][3];
    double cam_trans[num_cameras][3];
    for (int cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
        cam_rot[cam_idx][0] = true_cam_rot[cam_idx][0] + 0.05;
        cam_rot[cam_idx][1] = true_cam_rot[cam_idx][1] - 0.05;
        cam_rot[cam_idx][2] = true_cam_rot[cam_idx][2] + 0.05;

        cam_trans[cam_idx][0] = true_cam_trans[cam_idx][0] + 0.05;
        cam_trans[cam_idx][1] = true_cam_trans[cam_idx][1] - 0.05;
        cam_trans[cam_idx][2] = true_cam_trans[cam_idx][2] + 0.05;
    }

    // For camera 0, fix the extrinsics (it's at the origin)
    // Optionally, we can fix the extrinsics of camera 0 to reduce ambiguity
    bool fix_camera_0_extrinsics = true;

    // Initial guesses for 3D points (add some noise)
    double point3D[num_points][3];
    for (int i = 0; i < num_points; ++i) {
        point3D[i][0] = true_point3D[i][0] + point_dist(generator) * 0.1;
        point3D[i][1] = true_point3D[i][1] + point_dist(generator) * 0.1;
        point3D[i][2] = true_point3D[i][2] + point_dist(generator) * 0.1;
    }

    // Step 3: Set up Ceres problem
    ceres::Problem problem;

    for (const auto& obs : observations) {
        ceres::CostFunction* cost_function = ReprojectionErrorStereo::Create(obs.u, obs.v);

        problem.AddResidualBlock(cost_function,
                                 nullptr,
                                 intrinsics[obs.cam_idx],
                                 cam_rot[obs.cam_idx],
                                 cam_trans[obs.cam_idx],
                                 point3D[obs.pt_idx]);
    }

    // Set parameterization for rotation (to handle AngleAxis normalization)
    for (int i = 0; i < num_cameras; ++i) {
        ceres::LocalParameterization* angle_axis_parameterization =
            new ceres::AngleAxisParameterization();
        problem.SetParameterization(cam_rot[i], angle_axis_parameterization);
    }

    // Optionally, fix parameters to remove ambiguity
    if (fix_camera_0_extrinsics) {
        problem.SetParameterBlockConstant(cam_rot[0]);
        problem.SetParameterBlockConstant(cam_trans[0]);
    }

    // Step 4: Configure and solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Step 5: Output results
    std::cout << "\nFinal intrinsics:\n";
    for (int cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
        std::cout << "Camera " << cam_idx << " intrinsics:\n";
        std::cout << "fx: " << intrinsics[cam_idx][0] << ", fy: " << intrinsics[cam_idx][1]
                  << ", cx: " << intrinsics[cam_idx][2] << ", cy: " << intrinsics[cam_idx][3] << "\n";
        std::cout << "True intrinsics:\n";
        std::cout << "fx: " << true_intrinsics[cam_idx][0] << ", fy: " << true_intrinsics[cam_idx][1]
                  << ", cx: " << true_intrinsics[cam_idx][2] << ", cy: " << true_intrinsics[cam_idx][3] << "\n";
    }

    for (int i = 0; i < num_cameras; ++i) {
        std::cout << "\nCamera " << i << " extrinsics:\n";
        std::cout << "Rotation (estimated): [" << cam_rot[i][0] << ", " << cam_rot[i][1]
                  << ", " << cam_rot[i][2] << "]\n";
        std::cout << "Translation (estimated): [" << cam_trans[i][0] << ", " << cam_trans[i][1]
                  << ", " << cam_trans[i][2] << "]\n";

        std::cout << "Rotation (true): [" << true_cam_rot[i][0] << ", " << true_cam_rot[i][1]
                  << ", " << true_cam_rot[i][2] << "]\n";
        std::cout << "Translation (true): [" << true_cam_trans[i][0] << ", " << true_cam_trans[i][1]
                  << ", " << true_cam_trans[i][2] << "]\n";
    }

    // Optionally, output the estimated 3D points and compare with true points

    return 0;
}
