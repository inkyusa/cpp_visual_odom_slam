// imu_camera_extrinsic_calibration.cpp
/*
Compilation Instructions:

g++ -std=c++17 -o ceres_11_vi_extrinsic_calibration ./ceres_11_vi_extrinsic_calibration.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread

Make sure to have Ceres Solver and its dependencies installed.
*/

// #include <iostream>
// #include <vector>
// #include <random>
// #include <ceres/ceres.h>
// #include <ceres/rotation.h>
// #include <Eigen/Dense>

// using namespace std;
// using namespace ceres;
// using namespace Eigen;

// // Helper functions for rotation conversions
// void RotationMatrixToAngleAxis(const double R[9], double angle_axis[3]) {
//     ceres::RotationMatrixToAngleAxis(R, angle_axis);
// }

// void AngleAxisToRotationMatrix(const double angle_axis[3], double R[9]) {
//     ceres::AngleAxisToRotationMatrix(angle_axis, R);
// }

// // Structure to hold the IMU measurements
// struct IMUMeasurement {
//     double timestamp;
//     double angular_velocity[3]; // Gyroscope measurements
//     double linear_acceleration[3]; // Accelerometer measurements
// };

// // Structure to hold the poses
// struct Pose {
//     double timestamp;
//     double rotation[3];    // Angle-Axis rotation
//     double translation[3]; // Translation vector
// };

// // Gravity vector in the world frame
// const double gravity[3] = {0.0, 0.0, -9.81};

// // Function to simulate IMU measurements
// void SimulateIMUMeasurements(
//     const std::vector<Pose>& imu_poses,
//     const std::vector<double>& time_stamps,
//     std::vector<IMUMeasurement>& imu_measurements,
//     double imu_noise_std[6],
//     std::default_random_engine& generator)
// {
//     std::normal_distribution<double> gyro_noise(0.0, imu_noise_std[0]);
//     std::normal_distribution<double> accel_noise(0.0, imu_noise_std[3]);

//     for (size_t i = 1; i < imu_poses.size(); ++i) {
//         double dt = time_stamps[i] - time_stamps[i - 1];

//         // Compute angular velocity (omega = delta_theta / dt)
//         double delta_rotation[3] = {
//             imu_poses[i].rotation[0] - imu_poses[i - 1].rotation[0],
//             imu_poses[i].rotation[1] - imu_poses[i - 1].rotation[1],
//             imu_poses[i].rotation[2] - imu_poses[i - 1].rotation[2]
//         };

//         double angular_velocity[3] = {
//             delta_rotation[0] / dt,
//             delta_rotation[1] / dt,
//             delta_rotation[2] / dt
//         };

//         // Add gyroscope noise
//         angular_velocity[0] += gyro_noise(generator);
//         angular_velocity[1] += gyro_noise(generator);
//         angular_velocity[2] += gyro_noise(generator);

//         // Compute linear acceleration
//         double accel[3];
//         double velocity_prev[3] = {
//             (imu_poses[i - 1].translation[0]) / dt,
//             (imu_poses[i - 1].translation[1]) / dt,
//             (imu_poses[i - 1].translation[2]) / dt
//         };

//         double velocity_curr[3] = {
//             (imu_poses[i].translation[0]) / dt,
//             (imu_poses[i].translation[1]) / dt,
//             (imu_poses[i].translation[2]) / dt
//         };

//         double delta_velocity[3] = {
//             velocity_curr[0] - velocity_prev[0],
//             velocity_curr[1] - velocity_prev[1],
//             velocity_curr[2] - velocity_prev[2]
//         };

//         accel[0] = delta_velocity[0] / dt;
//         accel[1] = delta_velocity[1] / dt;
//         accel[2] = delta_velocity[2] / dt;

//         // Add gravity
//         accel[0] += gravity[0];
//         accel[1] += gravity[1];
//         accel[2] += gravity[2];

//         // Add accelerometer noise
//         accel[0] += accel_noise(generator);
//         accel[1] += accel_noise(generator);
//         accel[2] += accel_noise(generator);

//         // Store IMU measurement
//         IMUMeasurement imu_meas;
//         imu_meas.timestamp = time_stamps[i];
//         imu_meas.angular_velocity[0] = angular_velocity[0];
//         imu_meas.angular_velocity[1] = angular_velocity[1];
//         imu_meas.angular_velocity[2] = angular_velocity[2];

//         imu_meas.linear_acceleration[0] = accel[0];
//         imu_meas.linear_acceleration[1] = accel[1];
//         imu_meas.linear_acceleration[2] = accel[2];

//         imu_measurements.push_back(imu_meas);
//     }
// }

// // Cost function for IMU-Camera extrinsic calibration
// struct ExtrinsicCalibrationResidual {
//     ExtrinsicCalibrationResidual(const Pose& cam_pose,
//                                  const IMUMeasurement& imu_meas,
//                                  double dt)
//         : cam_rotation_{cam_pose.rotation[0], cam_pose.rotation[1], cam_pose.rotation[2]},
//           cam_translation_{cam_pose.translation[0], cam_pose.translation[1], cam_pose.translation[2]},
//           angular_velocity_{imu_meas.angular_velocity[0], imu_meas.angular_velocity[1], imu_meas.angular_velocity[2]},
//           linear_acceleration_{imu_meas.linear_acceleration[0], imu_meas.linear_acceleration[1], imu_meas.linear_acceleration[2]},
//           dt_(dt)
//           {}

//     template <typename T>
//     bool operator()(const T* const extrinsic_rotation,   // Rotation from IMU to Camera
//                     const T* const extrinsic_translation, // Translation from IMU to Camera
//                     T* residuals) const {
//         // Convert camera rotation to rotation matrix
//         T cam_R[9];
//         ceres::AngleAxisToRotationMatrix(cam_rotation_, cam_R);

//         // Convert extrinsic rotation to rotation matrix
//         T ext_R[9];
//         // ceres::AngleAxisToRotationMatrix(extrinsic_rotation, ext_R);
//         ceres::QuaternionToRotation(extrinsic_rotation, ext_R);

//         // Compute IMU rotation: imu_R = ext_R * cam_R
//         T imu_R[9];
//         // ceres::MatrixMultiply(ext_R, cam_R, imu_R);

//         Eigen::Map<const Eigen::Matrix<double, 3, 3>> ext_R_eigen(ext_R);
//         Eigen::Map<const Eigen::Matrix<double, 3, 3>> cam_R_eigen(cam_R);
//         Eigen::Map<const Eigen::Matrix<double, 3, 3>> imu_R_eigen(imu_R);
//         imu_R_eigen = ext_R_eigen * cam_R_eigen;
//         // ConvertToEigenMatrix(imu_R_eigen, imu_R);
//         Eigen::Map<const Eigen::Matrix<double, 3, 3>>{imu_R} = imu_R_eigen;



//         // Compute angular velocity in IMU frame
//         T cam_rotation_rate[3];
//         cam_rotation_rate[0] = cam_rotation_[0] / T(dt_);
//         cam_rotation_rate[1] = cam_rotation_[1] / T(dt_);
//         cam_rotation_rate[2] = cam_rotation_[2] / T(dt_);

//         T imu_rotation_rate[3];
//         ceres::AngleAxisRotatePoint(extrinsic_rotation, cam_rotation_rate, imu_rotation_rate);

//         // Residual for gyroscope (angular velocity)
//         residuals[0] = imu_rotation_rate[0] - T(angular_velocity_[0]);
//         residuals[1] = imu_rotation_rate[1] - T(angular_velocity_[1]);
//         residuals[2] = imu_rotation_rate[2] - T(angular_velocity_[2]);

//         // Compute linear acceleration in IMU frame
//         T cam_velocity[3] = {
//             cam_translation_[0] / T(dt_),
//             cam_translation_[1] / T(dt_),
//             cam_translation_[2] / T(dt_)
//         };

//         T cam_acceleration[3] = {
//             cam_velocity[0] / T(dt_),
//             cam_velocity[1] / T(dt_),
//             cam_velocity[2] / T(dt_)
//         };

//         // Rotate acceleration to IMU frame
//         T imu_acceleration[3];
//         ceres::AngleAxisRotatePoint(extrinsic_rotation, cam_acceleration, imu_acceleration);

//         // Add gravity
//         imu_acceleration[0] += T(gravity[0]);
//         imu_acceleration[1] += T(gravity[1]);
//         imu_acceleration[2] += T(gravity[2]);

//         // Residual for accelerometer (linear acceleration)
//         residuals[3] = imu_acceleration[0] - T(linear_acceleration_[0]);
//         residuals[4] = imu_acceleration[1] - T(linear_acceleration_[1]);
//         residuals[5] = imu_acceleration[2] - T(linear_acceleration_[2]);

//         return true;
//     }

//     static ceres::CostFunction* Create(const Pose& cam_pose,
//                                        const IMUMeasurement& imu_meas,
//                                        double dt) {
//         return (new ceres::AutoDiffCostFunction<ExtrinsicCalibrationResidual, 6, 3, 3>(
//             new ExtrinsicCalibrationResidual(cam_pose, imu_meas, dt)));
//     }

//     // Measured Camera pose and IMU measurements
//     double cam_rotation_[3];
//     double cam_translation_[3];
//     double angular_velocity_[3];
//     double linear_acceleration_[3];
//     double dt_;
// };

// // // Convert imu_R (double[9]) to Eigen matrix
// // Eigen::Matrix<double, 3, 3> ConvertToEigenMatrix(const double data[9]) {
// //     return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(data);
// // }

// // void ConvertToDoubleArray(const Eigen::Matrix<double, 3, 3>& matrix, double output[9]) {
// //     Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(output) = matrix;
// // }

// int main() {
//     // Simulation parameters
//     const int num_poses = 100; // Number of poses
//     const double total_time = 10.0; // Total time in seconds
//     const double dt = total_time / num_poses; // Time interval between poses

//     // True extrinsic parameters (from IMU to Camera)
//     double R_ic[3] = { 0.1, -0.05, 0.08 }; // Angle-Axis rotation
//     double t_ic[3] = { 0.05, 0.1, -0.08 }; // Translation

//     // Initialize random number generator
//     std::default_random_engine generator;
//     double imu_noise_std[6] = {0.005, 0.005, 0.005, 0.05, 0.05, 0.05}; // Noise std for gyro and accel
//     std::normal_distribution<double> gyro_noise(0.0, imu_noise_std[0]);
//     std::normal_distribution<double> accel_noise(0.0, imu_noise_std[3]);

//     // Generate simulated data
//     std::vector<Pose> cam_poses;
//     std::vector<Pose> imu_poses;
//     std::vector<double> time_stamps;

//     for (int i = 0; i < num_poses; ++i) {
//         double t = i * dt;
//         time_stamps.push_back(t);

//         // Simulate camera pose with respect to the world (e.g., following a circular path)
//         Pose cam_pose;
//         cam_pose.timestamp = t;
//         cam_pose.rotation[0] = 0.1 * sin(0.2 * t);
//         cam_pose.rotation[1] = 0.1 * cos(0.2 * t);
//         cam_pose.rotation[2] = 0.05 * sin(0.1 * t);

//         cam_pose.translation[0] = 1.0 * cos(0.1 * t);
//         cam_pose.translation[1] = 1.0 * sin(0.1 * t);
//         cam_pose.translation[2] = 0.5 * sin(0.05 * t);

//         cam_poses.push_back(cam_pose);

//         // Compute corresponding IMU pose using true extrinsics
//         Pose imu_pose;
//         imu_pose.timestamp = t;

//         // Rotate camera rotation by extrinsic rotation
//         double cam_R[9];
//         ceres::AngleAxisToRotationMatrix(cam_pose.rotation, cam_R);

//         double R_ic_matrix[9];
//         ceres::AngleAxisToRotationMatrix(R_ic, R_ic_matrix);

//         double imu_R[9];
//         Eigen::Map<const Eigen::Matrix<double, 3, 3>> R_ic_eigen(R_ic);
//         Eigen::Map<const Eigen::Matrix<double, 3, 3>> cam_R_eigen(cam_R);
//         Eigen::Map<const Eigen::Matrix<double, 3, 3>> imu_R_eigen(imu_R);

//         // Eigen::Matrix<double, 3, 3> R_ic_eigen;
//         // Eigen::Matrix<double, 3, 3> cam_R_eigen;
//         // Eigen::Matrix<double, 3, 3> imu_R_eigen;
//         // R_ic_eigen = ConvertToEigenMatrix(R_ic_matrix);
//         // cam_R_eigen = ConvertToEigenMatrix(cam_R);
//         imu_R_eigen = R_ic_eigen * cam_R_eigen;
//         // ceres::MatrixMultiply(R_ic_matrix, cam_R, imu_R);
//         // ConvertToEigenMatrix(imu_R_eigen, imu_R);
//         Eigen::Map<Eigen::Matrix<double, 3, 3>>{imu_R} = imu_R_eigen;

//         ceres::RotationMatrixToAngleAxis(imu_R, imu_pose.rotation);

//         // Transform camera translation to IMU frame
//         double temp_t[3];
//         ceres::AngleAxisRotatePoint(R_ic, cam_pose.translation, temp_t);
//         imu_pose.translation[0] = temp_t[0] + t_ic[0];
//         imu_pose.translation[1] = temp_t[1] + t_ic[1];
//         imu_pose.translation[2] = temp_t[2] + t_ic[2];

//         imu_poses.push_back(imu_pose);
//     }

//     // Simulate IMU measurements
//     std::vector<IMUMeasurement> imu_measurements;
//     SimulateIMUMeasurements(imu_poses, time_stamps, imu_measurements, imu_noise_std, generator);

//     // Set up the optimization problem
//     ceres::Problem problem;

//     // Initial estimates for extrinsic parameters (with some error)
//     double esti_R_ic[3] = { 0.0, 0.0, 0.0 };
//     double esti_t_ic[3] = { 0.0, 0.0, 0.0 };
//     double esti_R_ic_quat[4];
//     ceres::AngleAxisToQuaternion(esti_R_ic, esti_R_ic_quat);

//     // Add residual blocks
//     for (size_t i = 1; i < cam_poses.size(); ++i) {
//         // Time interval
//         double dt_i = time_stamps[i] - time_stamps[i - 1];

//         // Use previous and current camera poses to compute delta
//         Pose delta_cam_pose;
//         delta_cam_pose.rotation[0] = cam_poses[i].rotation[0] - cam_poses[i - 1].rotation[0];
//         delta_cam_pose.rotation[1] = cam_poses[i].rotation[1] - cam_poses[i - 1].rotation[1];
//         delta_cam_pose.rotation[2] = cam_poses[i].rotation[2] - cam_poses[i - 1].rotation[2];

//         delta_cam_pose.translation[0] = cam_poses[i].translation[0] - cam_poses[i - 1].translation[0];
//         delta_cam_pose.translation[1] = cam_poses[i].translation[1] - cam_poses[i - 1].translation[1];
//         delta_cam_pose.translation[2] = cam_poses[i].translation[2] - cam_poses[i - 1].translation[2];

//         ceres::CostFunction* cost_function = ExtrinsicCalibrationResidual::Create(
//             delta_cam_pose, imu_measurements[i - 1], dt_i);

//         problem.AddResidualBlock(cost_function, nullptr, esti_R_ic, esti_t_ic);
//     }

//     // Set parameterization for rotation (Angle-Axis)
//     // ceres::LocalParameterization* angle_axis_parameterization = new ceres::AngleAxisParameterization();
//     // problem.SetParameterization(esti_R_ic, angle_axis_parameterization);

//     auto* quaternion_manifold = new ceres::EigenQuaternionManifold();
//     problem.AddParameterBlock(esti_R_ic_quat, 4);
//     problem.SetManifold(esti_R_ic_quat, quaternion_manifold);


//     // Configure the solver
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_QR;
//     options.minimizer_progress_to_stdout = true;
//     options.max_num_iterations = 100;

//     // Solve
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     // Output results
//     std::cout << summary.FullReport() << "\n";

//     std::cout << "\nEstimated Extrinsic Parameters (from IMU to Camera):\n";
//     std::cout << "Rotation (Angle-Axis): [" << esti_R_ic[0] << ", " << esti_R_ic[1] << ", " << esti_R_ic[2] << "]\n";
//     std::cout << "Translation: [" << esti_t_ic[0] << ", " << esti_t_ic[1] << ", " << esti_t_ic[2] << "]\n";

//     std::cout << "\nTrue Extrinsic Parameters (from IMU to Camera):\n";
//     std::cout << "Rotation (Angle-Axis): [" << R_ic[0] << ", " << R_ic[1] << ", " << R_ic[2] << "]\n";
//     std::cout << "Translation: [" << t_ic[0] << ", " << t_ic[1] << ", " << t_ic[2] << "]\n";

//     return 0;
// }



// imu_camera_extrinsic_calibration.cpp
/*
Compilation Instructions:

g++ -std=c++17 -o ceres_11_vi_extrinsic_calibration ceres_11_vi_extrinsic_calibration.cpp \
    -I /usr/local/include \
    -lceres -lglog -pthread

Ensure you have the latest Ceres Solver and Eigen installed.
*/

// imu_camera_extrinsic_calibration.cpp
/*
Compilation Instructions:

g++ -std=c++17 -o ceres_11_vi_extrinsic_calibration ceres_11_vi_extrinsic_calibration.cpp \
    -I /usr/local/include \
    -lceres -lglog -pthread

Ensure you have the latest Ceres Solver and Eigen installed.
*/

// imu_camera_extrinsic_calibration.cpp
/*
Compilation Instructions:

g++ -std=c++17 -o ceres_11_vi_extrinsic_calibration ceres_11_vi_extrinsic_calibration.cpp \
    -I /usr/local/Cellar/ceres-solver/2.2.0_1/include \
    -I /usr/local/include/eigen3 \
    -L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
    -lceres -lglog -pthread

Ensure you have the latest Ceres Solver and Eigen installed.
*/

#include <iostream>
#include <vector>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <Eigen/Dense>

using namespace std;
using namespace ceres;
using namespace Eigen;

// Gravity vector in the world frame
const double gravity[3] = {0.0, 0.0, -9.81};

// Structure to hold the IMU measurements
struct IMUMeasurement {
    double timestamp;
    double angular_velocity[3];    // Gyroscope measurements
    double linear_acceleration[3]; // Accelerometer measurements
};

// Structure to hold the poses
struct Pose {
    double timestamp;
    double rotation[3];    // Angle-Axis rotation
    double translation[3]; // Translation vector
};

// Function to simulate IMU measurements
void SimulateIMUMeasurements(
    const std::vector<Pose>& imu_poses,
    const std::vector<double>& time_stamps,
    std::vector<IMUMeasurement>& imu_measurements,
    double imu_noise_std[6],
    std::default_random_engine& generator) {
    std::normal_distribution<double> gyro_noise(0.0, imu_noise_std[0]);
    std::normal_distribution<double> accel_noise(0.0, imu_noise_std[3]);

    for (size_t i = 1; i < imu_poses.size(); ++i) {
        double dt = time_stamps[i] - time_stamps[i - 1];

        // Compute angular velocity (omega = delta_theta / dt)
        double delta_rotation[3] = {
            imu_poses[i].rotation[0] - imu_poses[i - 1].rotation[0],
            imu_poses[i].rotation[1] - imu_poses[i - 1].rotation[1],
            imu_poses[i].rotation[2] - imu_poses[i - 1].rotation[2]};
        double angular_velocity[3] = {
            delta_rotation[0] / dt,
            delta_rotation[1] / dt,
            delta_rotation[2] / dt};

        // Add gyroscope noise
        angular_velocity[0] += gyro_noise(generator);
        angular_velocity[1] += gyro_noise(generator);
        angular_velocity[2] += gyro_noise(generator);

        // Compute linear acceleration
        double velocity_prev[3] = {
            imu_poses[i - 1].translation[0] / dt,
            imu_poses[i - 1].translation[1] / dt,
            imu_poses[i - 1].translation[2] / dt};
        double velocity_curr[3] = {
            imu_poses[i].translation[0] / dt,
            imu_poses[i].translation[1] / dt,
            imu_poses[i].translation[2] / dt};
        double delta_velocity[3] = {
            velocity_curr[0] - velocity_prev[0],
            velocity_curr[1] - velocity_prev[1],
            velocity_curr[2] - velocity_prev[2]};
        double accel[3] = {
            delta_velocity[0] / dt + gravity[0],
            delta_velocity[1] / dt + gravity[1],
            delta_velocity[2] / dt + gravity[2]};

        // Add accelerometer noise
        accel[0] += accel_noise(generator);
        accel[1] += accel_noise(generator);
        accel[2] += accel_noise(generator);

        // Store IMU measurement
        IMUMeasurement imu_meas;
        imu_meas.timestamp = time_stamps[i];
        imu_meas.angular_velocity[0] = angular_velocity[0];
        imu_meas.angular_velocity[1] = angular_velocity[1];
        imu_meas.angular_velocity[2] = angular_velocity[2];
        imu_meas.linear_acceleration[0] = accel[0];
        imu_meas.linear_acceleration[1] = accel[1];
        imu_meas.linear_acceleration[2] = accel[2];

        imu_measurements.push_back(imu_meas);
    }
}

// Cost function for IMU-Camera extrinsic calibration
struct ExtrinsicCalibrationResidual {
    ExtrinsicCalibrationResidual(const Pose& cam_pose,
                                 const IMUMeasurement& imu_meas,
                                 double dt)
        : cam_rotation_{cam_pose.rotation[0], cam_pose.rotation[1], cam_pose.rotation[2]},
          cam_translation_{cam_pose.translation[0], cam_pose.translation[1], cam_pose.translation[2]},
          angular_velocity_{imu_meas.angular_velocity[0], imu_meas.angular_velocity[1], imu_meas.angular_velocity[2]},
          linear_acceleration_{imu_meas.linear_acceleration[0], imu_meas.linear_acceleration[1], imu_meas.linear_acceleration[2]},
          dt_(dt) {}

    template <typename T>
    bool operator()(const T* const extrinsic_rotation,    // Quaternion (w, x, y, z)
                    const T* const extrinsic_translation, // Translation from IMU to Camera
                    T* residuals) const {
        // Map the extrinsic rotation to Eigen quaternion
        Eigen::Map<const Eigen::Quaternion<T>> ext_q(extrinsic_rotation);

        // Convert camera rotation (angle-axis) to quaternion
        Eigen::Matrix<T, 3, 1> cam_rot_vec;
        cam_rot_vec << T(cam_rotation_[0]), T(cam_rotation_[1]), T(cam_rotation_[2]);
        T angle = cam_rot_vec.norm();
        Eigen::Quaternion<T> cam_q;
        if (angle > T(1e-6)) {
            Eigen::Matrix<T, 3, 1> axis = cam_rot_vec / angle;
            cam_q = Eigen::Quaternion<T>(Eigen::AngleAxis<T>(angle, axis));
        } else {
            // For small angles, approximate as identity
            cam_q = Eigen::Quaternion<T>(T(1), T(0), T(0), T(0));
        }

        // Compute imu rotation: imu_q = ext_q * cam_q
        Eigen::Quaternion<T> imu_q = ext_q * cam_q;

        // Compute angular velocity in IMU frame
        Eigen::Matrix<T, 3, 1> cam_w = cam_rot_vec / T(dt_);
        Eigen::Matrix<T, 3, 1> imu_w = ext_q * cam_w;

        // Residual for gyroscope (angular velocity)
        residuals[0] = imu_w[0] - T(angular_velocity_[0]);
        residuals[1] = imu_w[1] - T(angular_velocity_[1]);
        residuals[2] = imu_w[2] - T(angular_velocity_[2]);

        // Compute linear acceleration in IMU frame
        Eigen::Matrix<T, 3, 1> cam_vel;
        cam_vel << T(cam_translation_[0]), T(cam_translation_[1]), T(cam_translation_[2]);
        cam_vel /= T(dt_);

        Eigen::Matrix<T, 3, 1> cam_accel = cam_vel / T(dt_); // Simplified acceleration

        // Rotate acceleration to IMU frame
        Eigen::Matrix<T, 3, 1> imu_accel = ext_q * cam_accel;

        // Add gravity
        imu_accel[0] += T(gravity[0]);
        imu_accel[1] += T(gravity[1]);
        imu_accel[2] += T(gravity[2]);

        // Residual for accelerometer (linear acceleration)
        residuals[3] = imu_accel[0] - T(linear_acceleration_[0]);
        residuals[4] = imu_accel[1] - T(linear_acceleration_[1]);
        residuals[5] = imu_accel[2] - T(linear_acceleration_[2]);

        return true;
    }

    static ceres::CostFunction* Create(const Pose& cam_pose,
                                       const IMUMeasurement& imu_meas,
                                       double dt) {
        return (new ceres::AutoDiffCostFunction<ExtrinsicCalibrationResidual, 6, 4, 3>(
            new ExtrinsicCalibrationResidual(cam_pose, imu_meas, dt)));
    }

    // Measured Camera pose and IMU measurements
    double cam_rotation_[3];
    double cam_translation_[3];
    double angular_velocity_[3];
    double linear_acceleration_[3];
    double dt_;
};

int main() {
    // Simulation parameters
    const int num_poses = 100;     // Number of poses
    const double total_time = 10.0; // Total time in seconds
    const double dt = total_time / num_poses; // Time interval between poses

    // True extrinsic parameters (from IMU to Camera)
    double R_ic[3] = {0.1, -0.05, 0.08}; // Angle-Axis rotation
    double t_ic[3] = {0.05, 0.1, -0.08}; // Translation

    // Initialize random number generator
    std::default_random_engine generator;
    double imu_noise_std[6] = {0.005, 0.005, 0.005, 0.05, 0.05, 0.05}; // Noise std for gyro and accel

    // Generate simulated data
    std::vector<Pose> cam_poses;
    std::vector<Pose> imu_poses;
    std::vector<double> time_stamps;

    // Convert R_ic to quaternion
    Eigen::Vector3d R_ic_vec(R_ic[0], R_ic[1], R_ic[2]);
    double R_ic_angle = R_ic_vec.norm();
    Eigen::Quaterniond R_ic_quat;
    if (R_ic_angle > 1e-6) {
        Eigen::Vector3d R_ic_axis = R_ic_vec / R_ic_angle;
        R_ic_quat = Eigen::Quaterniond(Eigen::AngleAxisd(R_ic_angle, R_ic_axis));
    } else {
        R_ic_quat = Eigen::Quaterniond::Identity();
    }

    for (int i = 0; i < num_poses; ++i) {
        double t = i * dt;
        time_stamps.push_back(t);

        // Simulate camera pose with respect to the world (e.g., following a circular path)
        Pose cam_pose;
        cam_pose.timestamp = t;
        cam_pose.rotation[0] = 0.1 * sin(0.2 * t);
        cam_pose.rotation[1] = 0.1 * cos(0.2 * t);
        cam_pose.rotation[2] = 0.05 * sin(0.1 * t);

        cam_pose.translation[0] = 1.0 * cos(0.1 * t);
        cam_pose.translation[1] = 1.0 * sin(0.1 * t);
        cam_pose.translation[2] = 0.5 * sin(0.05 * t);

        cam_poses.push_back(cam_pose);

        // Compute corresponding IMU pose using true extrinsics
        Pose imu_pose;
        imu_pose.timestamp = t;

        // Convert cam_pose.rotation to quaternion
        Eigen::Vector3d cam_rot_vec(cam_pose.rotation[0], cam_pose.rotation[1], cam_pose.rotation[2]);
        double cam_angle = cam_rot_vec.norm();
        Eigen::Quaterniond cam_q;
        if (cam_angle > 1e-6) {
            Eigen::Vector3d cam_axis = cam_rot_vec / cam_angle;
            cam_q = Eigen::Quaterniond(Eigen::AngleAxisd(cam_angle, cam_axis));
        } else {
            cam_q = Eigen::Quaterniond::Identity();
        }

        // Compute imu rotation: imu_q = R_ic_quat * cam_q
        Eigen::Quaterniond imu_q = R_ic_quat * cam_q;

        // Convert imu_q to angle-axis
        Eigen::AngleAxisd imu_aa(imu_q);
        Eigen::Vector3d imu_rot_vec = imu_aa.angle() * imu_aa.axis();
        imu_pose.rotation[0] = imu_rot_vec[0];
        imu_pose.rotation[1] = imu_rot_vec[1];
        imu_pose.rotation[2] = imu_rot_vec[2];

        // Transform camera translation to IMU frame
        Eigen::Vector3d cam_t(cam_pose.translation[0], cam_pose.translation[1], cam_pose.translation[2]);
        Eigen::Vector3d imu_t = R_ic_quat * cam_t + Eigen::Vector3d(t_ic[0], t_ic[1], t_ic[2]);
        imu_pose.translation[0] = imu_t[0];
        imu_pose.translation[1] = imu_t[1];
        imu_pose.translation[2] = imu_t[2];

        imu_poses.push_back(imu_pose);
    }

    // Simulate IMU measurements
    std::vector<IMUMeasurement> imu_measurements;
    SimulateIMUMeasurements(imu_poses, time_stamps, imu_measurements, imu_noise_std, generator);

    // Set up the optimization problem
    ceres::Problem problem;

    // Initial estimates for extrinsic parameters (with some error)

    double esti_t_ic[3] = {0.1, 0.1, 0.1};
    double esti_R_ic_quat[4] = {1.0, 0.0, 0.0, 0.0}; // Identity quaternion

    // Add residual blocks
    for (size_t i = 1; i < cam_poses.size(); ++i) {
        // Time interval
        double dt_i = time_stamps[i] - time_stamps[i - 1];

        // Use previous and current camera poses to compute delta
        Pose delta_cam_pose;
        delta_cam_pose.rotation[0] = cam_poses[i].rotation[0] - cam_poses[i - 1].rotation[0];
        delta_cam_pose.rotation[1] = cam_poses[i].rotation[1] - cam_poses[i - 1].rotation[1];
        delta_cam_pose.rotation[2] = cam_poses[i].rotation[2] - cam_poses[i - 1].rotation[2];

        delta_cam_pose.translation[0] = cam_poses[i].translation[0] - cam_poses[i - 1].translation[0];
        delta_cam_pose.translation[1] = cam_poses[i].translation[1] - cam_poses[i - 1].translation[1];
        delta_cam_pose.translation[2] = cam_poses[i].translation[2] - cam_poses[i - 1].translation[2];

        ceres::CostFunction* cost_function = ExtrinsicCalibrationResidual::Create(
            delta_cam_pose, imu_measurements[i - 1], dt_i);

        problem.AddResidualBlock(cost_function, nullptr, esti_R_ic_quat, esti_t_ic);
    }

    // Set manifold (parameterization) for quaternion
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold();
    problem.SetManifold(esti_R_ic_quat, quaternion_manifold);

    // Configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Output results
    std::cout << summary.FullReport() << "\n";

    // Convert estimated quaternion to angle-axis
    double esti_R_ic_angle_axis[3];
    ceres::QuaternionToAngleAxis(esti_R_ic_quat, esti_R_ic_angle_axis);

    std::cout << "\nEstimated Extrinsic Parameters (from IMU to Camera):\n";
    std::cout << "Rotation (Angle-Axis): [" << esti_R_ic_angle_axis[0] << ", " << esti_R_ic_angle_axis[1] << ", " << esti_R_ic_angle_axis[2] << "]\n";
    std::cout << "Translation: [" << esti_t_ic[0] << ", " << esti_t_ic[1] << ", " << esti_t_ic[2] << "]\n";

    std::cout << "\nTrue Extrinsic Parameters (from IMU to Camera):\n";
    std::cout << "Rotation (Angle-Axis): [" << R_ic[0] << ", " << R_ic[1] << ", " << R_ic[2] << "]\n";
    std::cout << "Translation: [" << t_ic[0] << ", " << t_ic[1] << ", " << t_ic[2] << "]\n";

    return 0;
}

