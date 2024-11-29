/*

g++ -std=c++17 -o BA_practice ./BA_practice.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann -lceres -lglog -lprotobuf -pthread

*/

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <random>
#include <iostream>
using namespace std;
using namespace cv;
using namespace ceres;




// Function to create a rotation matrix from Euler angles (roll, pitch, yaw)
Mat createRotationMatrix(double roll, double pitch, double yaw) {
    Mat R_x = (Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(roll), -sin(roll),
        0, sin(roll), cos(roll)
    );

    Mat R_y = (Mat_<double>(3, 3) <<
        cos(pitch), 0, sin(pitch),
        0, 1, 0,
        -sin(pitch), 0, cos(pitch)
    );

    Mat R_z = (Mat_<double>(3, 3) <<
        cos(yaw), -sin(yaw), 0,
        sin(yaw), cos(yaw), 0,
        0, 0, 1
    );

    Mat R = R_z * R_y * R_x;
    return R;
}
double rad2deg(double rad) {
    return rad * 180 / CV_PI;
}

double deg2rad(double deg) {
    return deg * CV_PI / 180;
}


pair<vector<Mat>, vector<Mat>> generateCorrespondences(
    int N,
    int num_points,
    vector<Point3d>& worldPoints,
    vector<vector<Point2d>>& projections,
    Mat& K,
    double noise_std_dev = 1.0 // Standard deviation of Gaussian noise
) {
    // 1. Generate 3D World Points
    default_random_engine generator;
    uniform_real_distribution<double> distribution_xy(-5.0, 5.0);
    uniform_real_distribution<double> distribution_z(5.0, 15.0);

    for (int i = 0; i < num_points; ++i) {
        double x = distribution_xy(generator);
        double y = distribution_xy(generator);
        double z = distribution_z(generator);
        worldPoints.emplace_back(x, y, z);
    }

    // 2. Generate N Camera Poses
    vector<Mat> R_list, t_list;

    double delta_x = 0.5; // Translation along X-axis per view
    double delta_angle = 5 * CV_PI / 180.0; // Rotation around Y-axis per view

    for (int i = 0; i < N; ++i) {
        double angle = i * delta_angle;
        Mat R = createRotationMatrix(0, angle, 0);
        Mat t = (Mat_<double>(3, 1) << i * delta_x, 0, 0);

        R_list.push_back(R);
        t_list.push_back(t);
    }

    // 3. Project Points onto Image Planes for Each View
    projections.resize(N);

    // Prepare Gaussian noise generator
    normal_distribution<double> noise_distribution(0.0, noise_std_dev);

    for (int i = 0; i < N; ++i) {
        Mat R = R_list[i];
        Mat t = t_list[i];

        Mat Rt;
        hconcat(R, t, Rt); // Combine rotation and translation

        Mat P = K * Rt; // Projection matrix for current view

        vector<Point2d>& imagePoints = projections[i];

        for (const auto& P3D : worldPoints) {
            // Convert 3D point to homogeneous coordinates
            Mat Pw = (Mat_<double>(4, 1) << P3D.x, P3D.y, P3D.z, 1.0);

            // Project point
            Mat p_homogeneous = P * Pw;
            Point2d p(p_homogeneous.at<double>(0, 0) / p_homogeneous.at<double>(2, 0),
                      p_homogeneous.at<double>(1, 0) / p_homogeneous.at<double>(2, 0));

            // Add Gaussian noise
            p.x += noise_distribution(generator);
            p.y += noise_distribution(generator);

            imagePoints.push_back(p);
        }
    }

    return {R_list, t_list};
}


class ReprojectionError {
public:
    Point2d observed;
    Mat K;

    ReprojectionError(const Point2d& observed_, const Mat& K_)
        : observed(observed_), K(K_) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        // camera[3,4,5] are the translation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // Apply the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        // Project to 2D
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        // Apply intrinsic parameters
        T fx = T(K.at<double>(0, 0));
        T fy = T(K.at<double>(1, 1));
        T cx = T(K.at<double>(0, 2));
        T cy = T(K.at<double>(1, 2));
        T u = fx * xp + cx;
        T v = fy * yp + cy;
        // Compute residuals
        residuals[0] = u - T(observed.x);
        residuals[1] = v - T(observed.y);
        return true;
    }

    static ceres::CostFunction* Create(const Point2d& observed, const Mat& K) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(observed, K)));
    }
};

int main() {
    // // Assuming we have initialized objectPoints (3D points) and imagePoints (2D observations)
    // vector<Point3d> objectPoints; // 3D points
    // vector<Point2d> imagePoints;  // 2D observations
    // // ... (Fill objectPoints and imagePoints with your data)

    // Camera intrinsic matrix K
    Mat K = (Mat_<double>(3,3) << 800, 0, 320,
        0, 800, 240,
        0, 0, 1);

    int N = 50; // Number of views (you can set this to any integer)
    int num_points = 150; // Number of world points

    vector<Point3d> worldPoints;
    vector<vector<Point2d>> projections; // projections[i][j] is the projection of point j in view i

    double noise_std_dev = 1.0; // Standard deviation of Gaussian noise in pixels

    auto [R_list, t_list] = generateCorrespondences(N, num_points, worldPoints, projections, K, noise_std_dev);

    cout << "# of world points = " << worldPoints.size() << endl;
    cout << "# of views = " << N << endl;

    default_random_engine generator;
    normal_distribution<double> rot_noise_distribution(0.0, deg2rad(0));
    normal_distribution<double> trans_noise_distribution(0.0, 0.00);

    // Prepare camera poses (rotation + translation for each view)
    vector<double> cameras(6 * N, 0.0); // N poses, each with 6 parameters (rotation and translation)
    for (int i = 0; i < N; ++i) {
        Mat R;
        Rodrigues(R_list[i], R); // Convert rotation matrix to angle-axis representation
        cameras[6 * i + 0] = R.at<double>(0) + rot_noise_distribution(generator);
        cameras[6 * i + 1] = R.at<double>(1) + rot_noise_distribution(generator);;
        cameras[6 * i + 2] = R.at<double>(2) + rot_noise_distribution(generator);;
        cameras[6 * i + 3] = t_list[i].at<double>(0) + trans_noise_distribution(generator);
        cameras[6 * i + 4] = t_list[i].at<double>(1) + trans_noise_distribution(generator);
        cameras[6 * i + 5] = t_list[i].at<double>(2) + trans_noise_distribution(generator);
    }

    // Copy objectPoints to a format suitable for Ceres
    vector<double> points3D(worldPoints.size() * 3);
    for (size_t i = 0; i < worldPoints.size(); ++i) {
        points3D[3 * i + 0] = worldPoints[i].x;
        points3D[3 * i + 1] = worldPoints[i].y;
        points3D[3 * i + 2] = worldPoints[i].z;
    }

    ceres::Problem problem;
    for (int i = 0; i < N; ++i) { // Iterate over views
        for (size_t j = 0; j < projections[i].size(); ++j) { // Iterate over points
            ceres::CostFunction* cost_function = ReprojectionError::Create(projections[i][j], K);
            // Add residual block
            problem.AddResidualBlock(cost_function, nullptr,
                                      &cameras[6 * i],         // Camera parameters for view i
                                      &points3D[3 * j]);       // 3D point j
        }
    }
    

    // Configure solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << "\n";
    // Print optimized poses
    for (int i = 0; i < N; ++i) {
        cout << "Camera " << i << ":\n";
        cout << "Rotation (angle-axis): [" << cameras[6 * i + 0] << ", "
             << cameras[6 * i + 1] << ", " << cameras[6 * i + 2] << "]\n";
        cout << "Translation: [" << cameras[6 * i + 3] << ", "
             << cameras[6 * i + 4] << ", " << cameras[6 * i + 5] << "]\n";
        cout << "GT translation: [" << t_list[i].at<double>(0) << ", "
             << t_list[i].at<double>(1) << ", " << t_list[i].at<double>(2) << "]"<<endl;
    }

    // Print optimized 3D points
    for (size_t i = 0; i < worldPoints.size(); ++i) {
        cout << "3D Point " << i << ": ["
             << points3D[3 * i + 0] << ", "
             << points3D[3 * i + 1] << ", "
             << points3D[3 * i + 2] << "]\n";
    }
    return 0;
}