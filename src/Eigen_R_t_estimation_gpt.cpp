#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Function to normalize 2D points
void normalizePoints(const std::vector<Eigen::Vector2d>& points,
                     std::vector<Eigen::Vector2d>& normalizedPoints,
                     Eigen::Matrix3d& T) {
    // Compute centroid
    Eigen::Vector2d centroid(0, 0);
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= points.size();

    // Compute scale
    double scale = 0;
    for (const auto& p : points) {
        scale += (p - centroid).norm();
    }
    scale = std::sqrt(2) * points.size() / scale;

    // Compute normalization matrix T
    T << scale,     0, -scale * centroid.x(),
            0, scale, -scale * centroid.y(),
            0,     0,                    1;

    // Normalize points
    normalizedPoints.clear();
    for (const auto& p : points) {
        Eigen::Vector3d p_homogeneous(p.x(), p.y(), 1);
        Eigen::Vector3d p_normalized = T * p_homogeneous;
        normalizedPoints.emplace_back(p_normalized.x() / p_normalized.z(),
                                      p_normalized.y() / p_normalized.z());
    }
}

// Function to estimate Essential matrix using the normalized eight-point algorithm
Eigen::Matrix3d estimateEssentialMatrix(const std::vector<Eigen::Vector2d>& points1,
                                        const std::vector<Eigen::Vector2d>& points2) {
    // Normalize points
    std::vector<Eigen::Vector2d> norm_points1, norm_points2;
    Eigen::Matrix3d T1, T2;
    normalizePoints(points1, norm_points1, T1);
    normalizePoints(points2, norm_points2, T2);

    // Build matrix A for homogeneous equation system Ax = 0
    Eigen::MatrixXd A(points1.size(), 9);
    for (size_t i = 0; i < points1.size(); ++i) {
        double x1 = norm_points1[i].x();
        double y1 = norm_points1[i].y();
        double x2 = norm_points2[i].x();
        double y2 = norm_points2[i].y();
        A.row(i) << x1 * x2, x1 * y2, x1,
                    y1 * x2, y1 * y2, y1,
                         x2,      y2,  1;
    }

    // Solve for the null space of A using SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd e = svd.matrixV().col(8);
    Eigen::Matrix3d E_norm;
    E_norm << e(0), e(1), e(2),
              e(3), e(4), e(5),
              e(6), e(7), e(8);

    // Enforce rank-2 constraint on Essential matrix
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E_norm, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd_E.singularValues();
    singular_values(2) = 0; // Set the smallest singular value to zero
    Eigen::Matrix3d E_rank2 = svd_E.matrixU() * singular_values.asDiagonal() * svd_E.matrixV().transpose();

    // Denormalize Essential matrix
    Eigen::Matrix3d E = T2.transpose() * E_rank2 * T1;

    return E;
}

// Function to decompose Essential matrix into possible rotations and translations
void decomposeEssentialMatrix(const Eigen::Matrix3d& E,
                              std::vector<Eigen::Matrix3d>& rotations,
                              std::vector<Eigen::Vector3d>& translations) {
    // SVD of Essential matrix
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Ensure proper rotation matrices
    if (svd.matrixU().determinant() < 0) svd.matrixU().col(2) *= -1;
    if (svd.matrixV().determinant() < 0) svd.matrixV().col(2) *= -1;

    // Possible rotations
    Eigen::Matrix3d W;
    W <<  0, -1,  0,
          1,  0,  0,
          0,  0,  1;

    Eigen::Matrix3d R1 = svd.matrixU() * W * svd.matrixV().transpose();
    Eigen::Matrix3d R2 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();

    // Possible translations (up to scale)
    Eigen::Vector3d t = svd.matrixU().col(2);

    rotations = { R1, R2 };
    translations = { t, -t };
}

int main() {
    // Toy example with 8 point correspondences
    std::vector<Eigen::Vector2d> points1 = {
        {0.5, 0.2}, {1.0, 0.8}, {0.2, 1.0}, {0.9, 0.3},
        {0.4, 0.7}, {0.7, 0.5}, {0.6, 0.9}, {0.3, 0.4}
    };

    // Define intrinsic camera matrix (assuming same for both cameras)
    Eigen::Matrix3d K;
    K << 800,   0, 320,
          0, 800, 240,
          0,   0,   1;

    // Define a known rotation and translation
    Eigen::Matrix3d true_R = Eigen::AngleAxisd(M_PI / 8, Eigen::Vector3d::UnitY()).toRotationMatrix(); // 22.5 degrees around Y-axis
    Eigen::Vector3d true_t(0.5, 0.0, 0.0); // Translation along X-axis

    // Project points into the second image
    std::vector<Eigen::Vector2d> points2;
    for (const auto& p : points1) {
        // Convert to normalized image coordinates
        Eigen::Vector3d p_cam = K.inverse() * Eigen::Vector3d(p.x(), p.y(), 1);
        // Apply rotation and translation
        Eigen::Vector3d p_cam2 = true_R * p_cam + true_t;
        // Project back to pixel coordinates
        Eigen::Vector3d p_img2 = K * p_cam2;
        points2.emplace_back(p_img2.x() / p_img2.z(), p_img2.y() / p_img2.z());
    }

    // Estimate Essential matrix
    Eigen::Matrix3d E = estimateEssentialMatrix(points1, points2);

    // Decompose Essential matrix
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> translations;
    decomposeEssentialMatrix(E, rotations, translations);

    // Output the results
    std::cout << "Estimated Essential Matrix:\n" << E << "\n\n";

    std::cout << "Possible Rotations:\n";
    for (const auto& R : rotations) {
        std::cout << R << "\n\n";
    }

    std::cout << "Possible Translations (up to scale):\n";
    for (const auto& t : translations) {
        std::cout << t.transpose() << "\n";
    }

    std::cout << "\nTrue Rotation:\n" << true_R << "\n\n";
    std::cout << "True Translation (up to scale):\n" << true_t.transpose() << "\n";

    return 0;
}
