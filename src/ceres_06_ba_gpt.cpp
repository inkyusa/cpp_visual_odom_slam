/*
g++ -std=c++17 -o ceres_06_ba_gpt ./ceres_06_ba_gpt.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

//input, fx, fy, cx, cy
//2d points (u, v) and their correspondecnes 3D points (X,Y,Z)

//output, refined intrinsics, pose, 3D points

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

struct ReprojectionError {
    ReprojectionError(double observed_u, double observed_v)
        : observed_u(observed_u), observed_v(observed_v) {}

    template <typename T>
    bool operator()(const T* const camera_intrinsics,
                    const T* const camera_rotation,
                    const T* const camera_translation,
                    const T* const point3d,
                    T* residuals) const {
        // camera_intrinsics: [fx, fy, cx, cy]
        // camera_rotation: rotation vector (angle-axis) [rx, ry, rz]
        // camera_translation: [tx, ty, tz]
        // point3d: [X, Y, Z]

        // Rotate and translate the point to camera coordinates
        T p[3];
        ceres::AngleAxisRotatePoint(camera_rotation, point3d, p);
        p[0] += camera_translation[0];
        p[1] += camera_translation[1];
        p[2] += camera_translation[2];

        // Project to 2D using the camera intrinsics
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T predicted_u = camera_intrinsics[0] * xp + camera_intrinsics[2];
        T predicted_v = camera_intrinsics[1] * yp + camera_intrinsics[3];

        // Compute residuals
        residuals[0] = predicted_u - T(observed_u);
        residuals[1] = predicted_v - T(observed_v);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_u, const double observed_v) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3, 3>(
            new ReprojectionError(observed_u, observed_v)));
    }

    double observed_u;
    double observed_v;
};

int main() {
    srand(time(0));

    const int NUM_CAMERAS = 2;
    const int NUM_POINTS = 5;

    // Initial estimates for camera intrinsics
    double camera_intrinsics[4] = {520.0, 480.0, 300.0, 260.0}; // fx, fy, cx, cy

    // True camera intrinsics (used to generate observations)
    double true_camera_intrinsics[4] = {500.0, 500.0, 320.0, 240.0};

    // Initial camera poses (rotation and translation)
    std::vector<double> camera_rotation[NUM_CAMERAS];
    std::vector<double> camera_translation[NUM_CAMERAS];

    // Initialize camera poses with small perturbations
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        camera_rotation[i].resize(3);
        camera_translation[i].resize(3);
        for (int j = 0; j < 3; ++j) {
            camera_rotation[i][j] = ((rand() % 1000) / 100000.0); // Small random rotations
            camera_translation[i][j] = ((rand() % 1000) / 1000.0) * 0.1; // Small random translations
        }
    }

    // True camera poses (used to generate observations)
    double true_camera_rotation[NUM_CAMERAS][3] = {{0, 0, 0}, {0, 0, 0}};
    double true_camera_translation[NUM_CAMERAS][3] = {{0, 0, 0}, {1, 0, 0}};

    // Initialize 3D points with small perturbations
    std::vector<double> points3D[NUM_POINTS];
    double true_points3D[NUM_POINTS][3] = {
        {0.5, 0.5, 5.0},
        {-0.5, -0.5, 6.0},
        {1.0, -1.0, 7.0},
        {-1.0, 1.0, 8.0},
        {0.0, 0.0, 9.0}
    };
    for (int i = 0; i < NUM_POINTS; ++i) {
        points3D[i].resize(3);
        for (int j = 0; j < 3; ++j) {
            points3D[i][j] = true_points3D[i][j] + ((rand() % 1000) / 100000.0);
        }
    }

    // Observations: each observation links a camera, a point, and the observed 2D position
    struct Observation {
        int camera_index;
        int point_index;
        double observed_u;
        double observed_v;
    };
    std::vector<Observation> observations;

    // Generate observations with noise
    for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
        for (int pt_idx = 0; pt_idx < NUM_POINTS; ++pt_idx) {
            double p[3];
            ceres::AngleAxisRotatePoint(true_camera_rotation[cam_idx], true_points3D[pt_idx], p);
            p[0] += true_camera_translation[cam_idx][0];
            p[1] += true_camera_translation[cam_idx][1];
            p[2] += true_camera_translation[cam_idx][2];

            double xp = p[0] / p[2];
            double yp = p[1] / p[2];

            double predicted_u = true_camera_intrinsics[0] * xp + true_camera_intrinsics[2];
            double predicted_v = true_camera_intrinsics[1] * yp + true_camera_intrinsics[3];

            // Add noise
            double noise_u = ((rand() % 1000) / 10000.0) - 0.05;
            double noise_v = ((rand() % 1000) / 10000.0) - 0.05;

            observations.push_back({cam_idx, pt_idx, predicted_u + noise_u, predicted_v + noise_v});
        }
    }

    // Set up the problem
    ceres::Problem problem;

    // Add camera intrinsics as a parameter block
    problem.AddParameterBlock(camera_intrinsics, 4);

    // Add camera poses to the problem
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        problem.AddParameterBlock(camera_rotation[i].data(), 3);
        problem.AddParameterBlock(camera_translation[i].data(), 3);
    }

    // Add 3D points to the problem
    for (int i = 0; i < NUM_POINTS; ++i) {
        problem.AddParameterBlock(points3D[i].data(), 3);
    }

    // Add residual blocks for each observation
    for (const auto& obs : observations) {
        ceres::CostFunction* cost_function = ReprojectionError::Create(obs.observed_u, obs.observed_v);
        problem.AddResidualBlock(cost_function, nullptr,
                                 camera_intrinsics,
                                 camera_rotation[obs.camera_index].data(),
                                 camera_translation[obs.camera_index].data(),
                                 points3D[obs.point_index].data());
    }

    // Optionally fix the first camera pose to avoid gauge freedom
    problem.SetParameterBlockConstant(camera_rotation[0].data());
    problem.SetParameterBlockConstant(camera_translation[0].data());

    // Configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Output the results
    std::cout << summary.FullReport() << "\n";

    std::cout << "Estimated Camera Intrinsics:\n";
    std::cout << "fx: " << camera_intrinsics[0] << "\n";
    std::cout << "fy: " << camera_intrinsics[1] << "\n";
    std::cout << "cx: " << camera_intrinsics[2] << "\n";
    std::cout << "cy: " << camera_intrinsics[3] << "\n";

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        std::cout << "Camera " << i << " Rotation (angle-axis): "
                  << camera_rotation[i][0] << ", "
                  << camera_rotation[i][1] << ", "
                  << camera_rotation[i][2] << "\n";
        std::cout << "Camera " << i << " Translation: "
                  << camera_translation[i][0] << ", "
                  << camera_translation[i][1] << ", "
                  << camera_translation[i][2] << "\n";
    }

    for (int i = 0; i < NUM_POINTS; ++i) {
        std::cout << "Point " << i << " Position: "
                  << points3D[i][0] << ", "
                  << points3D[i][1] << ", "
                  << points3D[i][2] << "\n";
    }

    return 0;
}
