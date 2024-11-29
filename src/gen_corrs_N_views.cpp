/*
g++ -std=c++17 -o gen_corrs_N_views ./gen_corrs_N_views.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-I /usr/local/include/eigen3 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

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

pair<vector<Mat>, vector<Mat>> generateCorrespondences(
    int N,
    int num_points,
    vector<Point3d>& worldPoints,
    vector<vector<Point2d>>& projections,
    Mat& K,
    double noise_std_dev = 1.0 // Standard deviation of Gaussian noise
) {
    // 1. Generate 3D World Points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_xy(-5.0, 5.0);
    std::uniform_real_distribution<double> distribution_z(5.0, 15.0);

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
    std::normal_distribution<double> noise_distribution(0.0, noise_std_dev);

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

int main(void) {
    Mat K = (Mat_<double>(3, 3) << 800, 0, 320,
                                   0, 800, 240,
                                   0, 0, 1);

    int N = 5; // Number of views (you can set this to any integer)
    int num_points = 150; // Number of world points

    vector<Point3d> worldPoints;
    vector<vector<Point2d>> projections; // projections[i][j] is the projection of point j in view i

    double noise_std_dev = 1.0; // Standard deviation of Gaussian noise in pixels

    auto [R_list, t_list] = generateCorrespondences(N, num_points, worldPoints, projections, K, noise_std_dev);

    cout << "# of world points = " << worldPoints.size() << endl;
    cout << "# of views = " << N << endl;

    for (int i = 0; i < N; ++i) {
        cout << "Camera Pose " << i + 1 << ":" << endl;
        cout << "R = " << endl << R_list[i] << endl;
        cout << "t = " << endl << t_list[i] << endl;
    }

    // Optionally, you can save the data to files or further process them

    return 0;
}
