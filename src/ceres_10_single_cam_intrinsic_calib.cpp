/*
Compilation instructions:
g++ -std=c++17 -o ceres_10_single_cam_intrinsic_calib ./ceres_10_single_cam_intrinsic_calib.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

#include <iostream>
#include <vector>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

double deg2rad(double deg) {
    return deg * M_PI / 180;
}

struct Observation {
    int camIdx;
    int pointIdx;
    double u;
    double v;
};

struct ReprjError {
    ReprjError(double in_u, double in_v) : measured_u_(in_u), measured_v_(in_v) {}

    template<typename T>
    bool operator()(const T* cam_quat,
                    const T* cam_trans,
                    const T* point3D,
                    const T* intrinsics,
                    T* residuals) const {
        // Rotate and translate the point
        T p[3];
        T angle_axis[3];
        ceres::QuaternionToAngleAxis(cam_quat, angle_axis);
        ceres::AngleAxisRotatePoint(angle_axis, point3D, p);
        
        p[0] += cam_trans[0];
        p[1] += cam_trans[1];
        p[2] += cam_trans[2];

        // Perspective division
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Apply intrinsics
        T fx = intrinsics[0];
        T fy = intrinsics[1];
        T cx = intrinsics[2];
        T cy = intrinsics[3];

        T predicted_u = fx * xp + cx;
        T predicted_v = fy * yp + cy;

        // Compute residuals
        residuals[0] = predicted_u - T(measured_u_);
        residuals[1] = predicted_v - T(measured_v_);

        return true;
    }

    static ceres::CostFunction* Create(double u, double v) {
        return (new ceres::AutoDiffCostFunction<ReprjError, 2, 4, 3, 3, 4>(
            new ReprjError(u, v)));
    }

private:
    double measured_u_;
    double measured_v_;
};

int main(void) {
    // Constants
    const int N = 4; // Number of camera poses
    const int M = 10; // Number of 3D points

    // True intrinsics
    double true_intrinsics[4] = {400.0, 400.0, 200.0, 200.0}; // fx, fy, cx, cy

    // True camera rotations (Angle-Axis)
    double true_cam_rot[N][3] = {
        {deg2rad(10), deg2rad(5), deg2rad(3)},
        {deg2rad(11), deg2rad(2.5), deg2rad(15)},
        {deg2rad(2), deg2rad(30), deg2rad(20)},
        {deg2rad(5), deg2rad(8), deg2rad(6)}
    };

    // True camera translations
    double true_cam_trans[N][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 1.5, 2},
        {1.5, 1, 3}
    };

    // Random number generators
    default_random_engine generator;
    normal_distribution<double> noise_dist_2d_point(0.0, 1.0); // 1-pixel std deviation
    uniform_real_distribution<double> noise_dist_trans(-0.05, 0.05); // -10cm ~ 10cm

    // True 3D points
    double true_point3D[M][3] = {
        {0.5, 0.5, 5.0},
        {-0.5, -0.5, 6.0},
        {1.0, -1.0, 7.0},
        {-1.0, 1.0, 8.0},
        {0.0, 0.0, 9.0},
        {0.5, -1.5, 5.0},
        {-0.5, 3.5, 6.0},
        {1.35, -1.2, 1.0},
        {-1.2, 1.0, 3.0},
        {0.0, 0.25, 2.0}
    };

    // Estimated 3D points (initial guess)
    double esti_point3D[M][3];
    for (int i = 0; i < M; ++i) {
        esti_point3D[i][0] = true_point3D[i][0];
        esti_point3D[i][1] = true_point3D[i][1];
        esti_point3D[i][2] = true_point3D[i][2];
    }

    // Generate observations
    vector<Observation> observations;
    for (int cam_idx = 0; cam_idx < N; ++cam_idx) {
        for (int pt_idx = 0; pt_idx < M; ++pt_idx) {
            double p[3];
            ceres::AngleAxisRotatePoint(true_cam_rot[cam_idx], true_point3D[pt_idx], p);
            p[0] += true_cam_trans[cam_idx][0];
            p[1] += true_cam_trans[cam_idx][1];
            p[2] += true_cam_trans[cam_idx][2];

            double xp = p[0] / p[2];
            double yp = p[1] / p[2];

            double predicted_u = true_intrinsics[0] * xp + true_intrinsics[2];
            double predicted_v = true_intrinsics[1] * yp + true_intrinsics[3];

            // Add noise
            observations.push_back({
                cam_idx,
                pt_idx,
                predicted_u + noise_dist_2d_point(generator),
                predicted_v + noise_dist_2d_point(generator)
            });
        }
    }

    // Initial estimates for intrinsics (perturbed)
    double esti_intrinsics[4] = {380.0, 380.0, 180.0, 180.0};

    // Initial estimates for camera rotations and translations
    double esti_cam_rot[N][3];
    double esti_cam_trans[N][3];
    double esti_cam_quat[N][4];
    for (int i = 0; i < N; i++) {
        esti_cam_rot[i][0] = 0.0; // Small rotation
        esti_cam_rot[i][1] = 0.0;
        esti_cam_rot[i][2] = 0.0;

        esti_cam_trans[i][0] = true_cam_trans[i][0] + noise_dist_trans(generator);
        esti_cam_trans[i][1] = true_cam_trans[i][1] + noise_dist_trans(generator);
        esti_cam_trans[i][2] = true_cam_trans[i][2] + noise_dist_trans(generator);
    }

    // Build the problem
    ceres::Problem problem;

    // Add parameter blocks
    for (int i = 0; i < N; i++) {
        problem.AddParameterBlock(esti_cam_rot[i], 3);
        problem.AddParameterBlock(esti_cam_trans[i], 3);
    }
    for (int i = 0; i < M; i++) {
        problem.AddParameterBlock(esti_point3D[i], 3);
    }
    problem.AddParameterBlock(esti_intrinsics, 4);

    for (int i = 0; i < N; i++) {
        ceres::AngleAxisToQuaternion(esti_cam_rot[i], esti_cam_quat[i]);
    }

    // Create and attach the quaternion manifold
    for (int i = 0; i < N; i++) {
        auto* quaternion_manifold = new ceres::EigenQuaternionManifold();
        problem.AddParameterBlock(esti_cam_quat[i], 4);
        problem.SetManifold(esti_cam_quat[i], quaternion_manifold);
    }

    // Fix the first camera's pose to remove gauge freedom
    problem.SetParameterBlockConstant(esti_cam_rot[0]);
    problem.SetParameterBlockConstant(esti_cam_quat[0]);

    // Add residuals
    for (const auto& obs : observations) {
        ceres::CostFunction* cost_function = ReprjError::Create(obs.u, obs.v);
        problem.AddResidualBlock(cost_function,
                                 new ceres::HuberLoss(1.0),
                                 esti_cam_quat[obs.camIdx],
                                 esti_cam_trans[obs.camIdx],
                                 esti_point3D[obs.pointIdx],
                                 esti_intrinsics);
    }

    // Configure the solver
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1500;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Output the results
    std::cout << summary.FullReport() << "\n";

    std::cout << "\nEstimated Camera Intrinsics:\n";
    std::cout << "fx: " << esti_intrinsics[0] << "\n";
    std::cout << "fy: " << esti_intrinsics[1] << "\n";
    std::cout << "cx: " << esti_intrinsics[2] << "\n";
    std::cout << "cy: " << esti_intrinsics[3] << "\n";

    std::cout << "\nTrue Camera Intrinsics:\n";
    std::cout << "fx: " << true_intrinsics[0] << "\n";
    std::cout << "fy: " << true_intrinsics[1] << "\n";
    std::cout << "cx: " << true_intrinsics[2] << "\n";
    std::cout << "cy: " << true_intrinsics[3] << "\n";

    for (int i = 0; i < N; ++i) {
        std::cout << "\nCamera " << i << " Rotation (estimated): ["
                  << esti_cam_rot[i][0] << ", "
                  << esti_cam_rot[i][1] << ", "
                  << esti_cam_rot[i][2] << "]\n";
        std::cout << "Camera " << i << " Rotation (true): ["
                  << true_cam_rot[i][0] << ", "
                  << true_cam_rot[i][1] << ", "
                  << true_cam_rot[i][2] << "]\n";

        std::cout << "Camera " << i << " Translation (estimated): ["
                  << esti_cam_trans[i][0] << ", "
                  << esti_cam_trans[i][1] << ", "
                  << esti_cam_trans[i][2] << "]\n";
        std::cout << "Camera " << i << " Translation (true): ["
                  << true_cam_trans[i][0] << ", "
                  << true_cam_trans[i][1] << ", "
                  << true_cam_trans[i][2] << "]\n";
    }

    // Optionally, compare estimated 3D points with true points
    for (int i = 0; i < M; ++i) {
        std::cout << "\nPoint " << i << " Position (estimated): ["
                  << esti_point3D[i][0] << ", "
                  << esti_point3D[i][1] << ", "
                  << esti_point3D[i][2] << "]\n";
        std::cout << "Point " << i << " Position (true): ["
                  << true_point3D[i][0] << ", "
                  << true_point3D[i][1] << ", "
                  << true_point3D[i][2] << "]\n";
    }

    return 0;
}


/*
iter      cost      cost_change  |gradient|   |step|    tr_ratio  tr_radius  ls_iter  iter_time  total_time
   0  4.979720e+03    0.00e+00    1.72e+03   0.00e+00   0.00e+00  1.00e+04        0    1.93e-03    2.12e-03
   1  1.935526e+03    3.04e+03    7.92e+03   0.00e+00   1.22e+00  3.00e+04        1    3.01e-03    5.18e-03
   2  5.147906e+02    1.42e+03    4.65e+03   4.42e+01   1.46e+00  9.00e+04        1    2.76e-03    7.96e-03
   3  5.640488e+01    4.58e+02    3.25e+03   2.23e+01   1.74e+00  2.70e+05        1    2.74e-03    1.07e-02
   4  1.606368e+01    4.03e+01    7.21e+02   1.50e+01   1.76e+00  8.10e+05        1    3.34e-03    1.41e-02
   5  1.490536e+01    1.16e+00    1.64e+02   4.69e+00   1.18e+00  2.43e+06        1    2.87e-03    1.70e-02
   6  1.478712e+01    1.18e-01    1.84e+01   9.72e-01   1.60e+00  7.29e+06        1    3.32e-03    2.03e-02
   7  1.473493e+01    5.22e-02    8.50e+00   6.99e-01   1.57e+00  2.19e+07        1    2.74e-03    2.31e-02
   8  1.472408e+01    1.08e-02    3.11e+00   3.77e-01   1.36e+00  6.56e+07        1    2.74e-03    2.58e-02
   9  1.472245e+01    1.63e-03    1.87e+00   3.11e-01   1.40e+00  1.97e+08        1    2.89e-03    2.87e-02
  10  1.472217e+01    2.84e-04    1.76e+00   1.78e-01   1.44e+00  5.90e+08        1    2.87e-03    3.16e-02
  11  1.472211e+01    5.88e-05    6.75e-01   1.03e-01   1.48e+00  1.77e+09        1    2.81e-03    3.44e-02

Solver Summary (v 2.2.0-eigen-(3.4.0)-lapack-suitesparse-(7.3.1)-metis-(5.1.0)-acceleratesparse-eigensparse)

                                     Original                  Reduced
Parameter blocks                           23                       18
Parameters                                 74                       58
Effective parameters                       70                       55
Residual blocks                            40                       40
Residuals                                  80                       80

Minimizer                        TRUST_REGION
Trust region strategy     LEVENBERG_MARQUARDT
Sparse linear algebra library    SUITE_SPARSE

                                        Given                     Used
Linear solver          SPARSE_NORMAL_CHOLESKY   SPARSE_NORMAL_CHOLESKY
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                       18

Cost:
Initial                          4.979720e+03
Final                            1.472211e+01
Change                           4.964998e+03

Minimizer iterations                       12
Successful steps                           12
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.000191

  Residual only evaluation           0.000170 (12)
  Jacobian & residual evaluation     0.032462 (12)
  Linear solver                      0.001289 (12)
Minimizer                            0.034386

Postprocessor                        0.000009
Total                                0.034587

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 9.901764e-07 <= 1.000000e-06)


Estimated Camera Intrinsics:
fx: 407.921
fy: 403.813
cx: 221.313
cy: 188.212

True Camera Intrinsics:
fx: 400
fy: 400
cx: 200
cy: 200

Camera 0 Rotation (estimated): [0, 0, 0]
Camera 0 Rotation (true): [0.174533, 0.0872665, 0.0523599]
Camera 0 Translation (estimated): [0.038651, -0.257284, -0.107904]
Camera 0 Translation (true): [0, 0, 0]

Camera 1 Rotation (estimated): [0, 0, 0]
Camera 1 Rotation (true): [0.191986, 0.0436332, 0.261799]
Camera 1 Translation (estimated): [1.17072, -0.247003, -0.103827]
Camera 1 Translation (true): [1, 0, 0]

Camera 2 Rotation (estimated): [0, 0, 0]
Camera 2 Rotation (true): [0.0349066, 0.523599, 0.349066]
Camera 2 Translation (estimated): [-0.0785922, 1.33733, 1.93371]
Camera 2 Translation (true): [0, 1.5, 2]

Camera 3 Rotation (estimated): [0, 0, 0]
Camera 3 Rotation (true): [0.0872665, 0.139626, 0.10472]
Camera 3 Translation (estimated): [1.46829, 0.884319, 3.13241]
Camera 3 Translation (true): [1.5, 1, 3]

Point 0 Position (estimated): [0.653069, 0.0546601, 5.37606]
Point 0 Position (true): [0.5, 0.5, 5]

Point 1 Position (estimated): [-0.26764, -1.17039, 6.26305]
Point 1 Position (true): [-0.5, -0.5, 6]

Point 2 Position (estimated): [1.36139, -1.78171, 7.19671]
Point 2 Position (true): [1, -1, 7]

Point 3 Position (estimated): [-0.808318, 0.0428713, 8.61929]
Point 3 Position (true): [-1, 1, 8]

Point 4 Position (estimated): [0.281611, -1.07298, 9.26719]
Point 4 Position (true): [0, 0, 9]

Point 5 Position (estimated): [0.771575, -1.99037, 4.95856]
Point 5 Position (true): [0.5, -1.5, 5]

Point 6 Position (estimated): [-0.52864, 3.05206, 7.28736]
Point 6 Position (true): [-0.5, 3.5, 6]

Point 7 Position (estimated): [1.4704, -1.05449, 0.80681]
Point 7 Position (true): [1.35, -1.2, 1]

Point 8 Position (estimated): [-1.2449, 0.783154, 3.60231]
Point 8 Position (true): [-1.2, 1, 3]

Point 9 Position (estimated): [0.0331089, 0.214381, 2.2983]
Point 9 Position (true): [0, 0.25, 2]


*/