/*
g++ -std=c++17 -o gen_corrs ./gen_corrs.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-I /usr/local/include/eigen3 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/
#include <iostream>
#include <opencv2/opencv.hpp>

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
void generateCorrespondences(vector<Point2d>& pt1, vector<Point2d>& pt2, Mat& K){
    // 1. Simulate 3D Points in World Coordinates
    vector<Point3d> worldPoints;
    worldPoints.emplace_back(0.0, 0.0, 5.0);
    worldPoints.emplace_back(1.0, 0.0, 5.0);
    worldPoints.emplace_back(0.0, 1.0, 5.0);
    worldPoints.emplace_back(1.0, 1.0, 5.0);
    worldPoints.emplace_back(0.0, 0.5, 5.0);
    worldPoints.emplace_back(1.0, 1.0, 3.0);
    worldPoints.emplace_back(2.0, 1.0, 2.0);
    worldPoints.emplace_back(1.0, 2.0, 3.0);

    // 3. Simulate Camera Poses at t=1 and t=2
    // At t=1, the camera is at the origin looking along the Z-axis
    Mat R1 = Mat::eye(3, 3, CV_64F); // Identity rotation
    Mat t1 = Mat::zeros(3, 1, CV_64F); // Zero translation

    // At t=2, the camera has moved along X-axis and rotated slightly
    double angle = 5 * CV_PI / 180.0; // Rotate 5 degrees around Y-axis
    Mat R2 = createRotationMatrix(0, angle, 0);
    Mat t2 = (Mat_<double>(3, 1) << 0.5, 0, 0); // Translate 0.5 units along X-axis

    // 4. Project Points onto Image Planes at t=1 and t=2
    // vector<Point2d> imagePoints1, imagePoints2;

    for (const auto& P : worldPoints) {
        // Convert 3D point to homogeneous coordinates
        Mat Pw = (Mat_<double>(4, 1) << P.x, P.y, P.z, 1.0);

        // Projection matrix at t=1
        Mat P1 = K * (R1 * Mat::eye(3, 4, CV_64F));

        // Projection matrix at t=2
        Mat Rt;
        hconcat(R2, t2, Rt); // Combine rotation and translation
        Mat P2 = K * Rt;

        // Project point at t=1
        Mat p1_homogeneous = P1 * Pw.rowRange(0, 4);
        Point2d p1(p1_homogeneous.at<double>(0, 0) / p1_homogeneous.at<double>(2, 0),
                   p1_homogeneous.at<double>(1, 0) / p1_homogeneous.at<double>(2, 0));
        pt1.push_back(p1);

        // Project point at t=2
        Mat p2_homogeneous = P2 * Pw.rowRange(0, 4);
        Point2d p2(p2_homogeneous.at<double>(0, 0) / p2_homogeneous.at<double>(2, 0),
                   p2_homogeneous.at<double>(1, 0) / p2_homogeneous.at<double>(2, 0));
        pt2.push_back(p2);
    }
}
int main(void) {
    Mat K = (Mat_<double>(3,3) << 800, 0, 320,
        0, 800, 240,
        0, 0, 1);
    vector<Point2d> pts1, pts2;
    generateCorrespondences(pts1, pts2, K);
    cout << "# of correspondences = " << pts1.size() << endl;
}