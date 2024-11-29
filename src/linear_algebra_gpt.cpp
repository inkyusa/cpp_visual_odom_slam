#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <iostream>

int main() {
    // Rotation matrix (identity)
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    // Translation vector
    Eigen::Vector3d t(1, 0, 0);

    // Create SE(3) transformation
    Sophus::SE3d SE3_Rt(R, t);

    // Log map to get Lie algebra (tangent space)
    Eigen::Matrix<double, 6, 1> se3 = SE3_Rt.log();

    std::cout << "se3 (Lie algebra):\n" << se3 << std::endl;

    // Apply perturbation
    Eigen::Matrix<double, 6, 1> delta_se3;
    delta_se3.setZero();
    delta_se3(0) = 0.01; // Small rotation around x-axis
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(delta_se3) * SE3_Rt;

    std::cout << "Updated SE(3) matrix:\n" << SE3_updated.matrix() << std::endl;

    return 0;
}
