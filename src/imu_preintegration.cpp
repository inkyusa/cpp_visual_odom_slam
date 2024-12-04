/*
g++ -std=c++17 -o imu_preintegration ./imu_preintegration.cpp \
-I /usr/local/include/eigen3
*/

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>

double rad2deg(double rad) {
    return rad * 180 / M_PI;
}

double deg2rad(double deg) {
    return deg * M_PI / 180;
}

// Constants
const double dt = 0.01; // Time step (100 Hz)
const int num_steps = 10000; // Number of steps for observation
const Eigen::Vector3d gravity(0, 0, -9.81); // Gravity vector
// Preintegration interval (e.g., integrate over N steps before updating state)
const int preint_interval = 100; // Adjust as needed
// #define NO_NOISE //uncomment if you want to test without any noise.

// IMU noise parameters
#ifdef NO_NOISE
const Eigen::Vector3d accel_noise_std(0.0, 0.0, 0.0); // Accelerometer noise standard deviation
const Eigen::Vector3d gyro_noise_std(deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)); // Gyroscope noise standard deviation
#else
const Eigen::Vector3d accel_noise_std(0.1, 0.1, 0.1); // Accelerometer noise standard deviation
const Eigen::Vector3d gyro_noise_std(deg2rad(1.0), deg2rad(1.0), deg2rad(1.0)); // Gyroscope noise standard deviation
#endif
// Skew-symmetric matrix
Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<    0, -v.z(),  v.y(),
            v.z(),     0, -v.x(),
        -v.y(),  v.x(),     0;
    return m;
}

// Exponential map for SO(3) as a standalone function
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& omega) {
    double theta = omega.norm();
    Eigen::Matrix3d omega_hat;
    omega_hat <<    0, -omega.z(),  omega.y(),
                omega.z(),     0, -omega.x(),
               -omega.y(),  omega.x(),     0;

    if (theta < 1e-5) {
        // Use first-order approximation
        return Eigen::Matrix3d::Identity() + omega_hat;
    } else {
        // Rodrigues' rotation formula
        return Eigen::Matrix3d::Identity() +
               (sin(theta) / theta) * omega_hat +
               ((1 - cos(theta)) / (theta * theta)) * omega_hat * omega_hat;
    }
}



struct IMUData {
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};

class IMUPreintegration {
public:
    IMUPreintegration() {
        delta_R.setIdentity();
        delta_v.setZero();
        delta_p.setZero();
    }

    void integrate(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt) {
        // Update rotation
        Eigen::Matrix3d R = ExpSO3(gyro * dt);
        delta_R = delta_R * R;

        // Compute acceleration in world frame without gravity
        Eigen::Vector3d accel_world = delta_R * accel;

        // Store previous velocity
        Eigen::Vector3d delta_v_prev = delta_v;

        // Update velocity
        delta_v += accel_world * dt;

        // Update position using previous velocity
        delta_p += delta_v_prev * dt + 0.5 * accel_world * dt * dt;
    }

    Eigen::Matrix3d getDeltaR() const { return delta_R; }
    Eigen::Vector3d getDeltaV() const { return delta_v; }
    Eigen::Vector3d getDeltaP() const { return delta_p; }

    void reset() {
        delta_R.setIdentity();
        delta_v.setZero();
        delta_p.setZero();
    }

private:
    Eigen::Matrix3d delta_R;
    Eigen::Vector3d delta_v;
    Eigen::Vector3d delta_p;   
};

class IMUIntegration {
public:
    IMUIntegration() {
        R.setIdentity();
        v.setZero();
        p.setZero();
    }

    void integrate(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt) {
        // Update rotation
        Eigen::Matrix3d R_inc = ExpSO3(gyro * dt);
        R = R * R_inc;

        // Compute acceleration in world frame
        Eigen::Vector3d accel_world = R * accel + gravity;

        // Store previous velocity
        Eigen::Vector3d v_prev = v;

        // Update velocity
        v += accel_world * dt;

        // Update position using previous velocity
        p += v_prev * dt + 0.5 * accel_world * dt * dt;
    }

    Eigen::Matrix3d getRotation() const { return R; }
    Eigen::Vector3d getVelocity() const { return v; }
    Eigen::Vector3d getPosition() const { return p; }

private:
    Eigen::Matrix3d R;
    Eigen::Vector3d v;
    Eigen::Vector3d p;
};

Eigen::Vector3d getTrueAccel(double t) {
    // Nonlinear acceleration: oscillatory motion with varying frequency
    return Eigen::Vector3d(
        0.1 * sin(0.1 * t),
        0.2 * cos(0.2 * t),
        0.3 * sin(0.3 * t) * cos(0.1 * t)
    );
}

Eigen::Vector3d getTrueGyro(double t) {
    // Nonlinear angular velocity: combined oscillatory and constant rotation
    return Eigen::Vector3d(
        0.01 + 0.005 * sin(0.05 * t),
        0.02 * cos(0.1 * t),
        0.03 * sin(0.1 * t) * cos(0.05 * t)
    );
}

int main() {
    // Generate synthetic IMU data
    std::vector<IMUData> imu_data;
    imu_data.reserve(num_steps);

    for (int i = 0; i < num_steps; ++i) {
        double t = i * dt;
        IMUData data;
        data.accel = getTrueAccel(t) + accel_noise_std.cwiseProduct(Eigen::Vector3d::Random());
        data.gyro = getTrueGyro(t) + gyro_noise_std.cwiseProduct(Eigen::Vector3d::Random());
        imu_data.push_back(data);
    }

    // Initialize integration methods
    IMUPreintegration preint;
    IMUIntegration ordinary_int;

    // Containers to store positions over time
    std::vector<Eigen::Vector3d> positions_ground_truth;
    std::vector<Eigen::Vector3d> positions_preint;
    std::vector<Eigen::Vector3d> positions_ordinary;

    // Initialize variables for ground truth
    Eigen::Matrix3d R_gt = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v_gt = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_gt = Eigen::Vector3d::Zero();

    // Initialize variables for preintegration global state
    Eigen::Matrix3d R_preint = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v_preint = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_preint = Eigen::Vector3d::Zero();

    // Main loop
    for (int i = 0; i < num_steps; ++i) {
        double t = i * dt;

        // Get nonlinear motion parameters
        Eigen::Vector3d true_accel = getTrueAccel(t);
        Eigen::Vector3d true_gyro = getTrueGyro(t);

        // Ground truth update
        Eigen::Matrix3d R_inc_gt = ExpSO3(true_gyro * dt); // Use ExpSO3 for consistency
        R_gt = R_gt * R_inc_gt;
        Eigen::Vector3d accel_world_gt = R_gt * true_accel + gravity;
        Eigen::Vector3d v_gt_prev = v_gt;
        v_gt += accel_world_gt * dt;
        p_gt += v_gt_prev * dt + 0.5 * accel_world_gt * dt * dt;

        positions_ground_truth.push_back(p_gt);

        // Ordinary integration
        ordinary_int.integrate(imu_data[i].accel, imu_data[i].gyro, dt);
        positions_ordinary.push_back(ordinary_int.getPosition());

        // Preintegration
        preint.integrate(imu_data[i].accel, imu_data[i].gyro, dt);

        // At preintegration interval, update state and reset preintegrator
        if ((i + 1) % preint_interval == 0) {
            double delta_t = preint_interval * dt;

            // Store previous velocity
            Eigen::Vector3d v_preint_prev = v_preint;

            // Update global velocity
            v_preint += gravity * delta_t + R_preint * preint.getDeltaV();

            // Update global position
            p_preint += v_preint_prev * delta_t + 0.5 * gravity * delta_t * delta_t + R_preint * preint.getDeltaP();

            // Update global rotation
            R_preint = R_preint * preint.getDeltaR();

            // Record the updated global position
            positions_preint.push_back(p_preint);

            // Reset preintegrator
            preint.reset();
        } else {
            // For plotting purposes, maintain the latest position
            positions_preint.push_back(p_preint);
        }
    }

    // Write positions to CSV files
    std::ofstream gt_file("ground_truth.csv");
    std::ofstream ord_file("ordinary_integration.csv");
    std::ofstream pre_file("preintegration.csv");

    for (int i = 0; i < num_steps; ++i) {
        gt_file << positions_ground_truth[i].transpose() << "\n";
        ord_file << positions_ordinary[i].transpose() << "\n";
        pre_file << positions_preint[i].transpose() << "\n";
    }

    gt_file.close();
    ord_file.close();
    pre_file.close();

    std::cout << "Data saved to CSV files.\n";

    return 0;
}
