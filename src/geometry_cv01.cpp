/*
g++ -std=c++17 -o geometry_cv ./geometry_cv.cpp \
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
pair<Mat, Mat> generateCorrespondences(vector<Point2d>& pt1, vector<Point2d>& pt2, Mat& K){
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
    return {R2, t2};
}
int main(void) {
    Mat K = (Mat_<double>(3,3) << 800, 0, 320,
        0, 800, 240,
        0, 0, 1);
    vector<Point2d> pts1, pts2;
    auto [R_gt, t_gt] = generateCorrespondences(pts1, pts2, K);
    cout << "# of correspondences = " << pts1.size() << endl;

    //Find Essential matrix given correspondences
    Mat E = findEssentialMat(pts1, pts2, K, RANSAC, 0.9999, 0.5);

    //verification of E
    Mat x2_mat = (Mat_<double>(3, 1) << pts2[0].x, pts2[0].y, 1);
    Mat x1_mat = (Mat_<double>(3, 1) << pts1[0].x, pts1[0].y, 1);
    Mat result = (K.inv()*x2_mat).t() * E * K.inv()*x1_mat;
    // Point2d x2 = pts2[0];
    // Point2d x1 = pts1[0];
    cout << result << endl;

    //Recover R and t from it
    Mat R, t;
    int inliers = recoverPose(E, pts1, pts2, K, R, t);
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;

    cout << "R_gt = " << R_gt << endl;
    cout << "t_gt = " << t_gt << endl;

    //Triangulation
    //Given two poses P1 and P2 and their correspondences
    Mat P1 = K * Mat::eye(3, 4, CV_64F); //initial pose, 3x4
    Mat temp;
    hconcat(R_gt, t_gt, temp);
    Mat P2 = K * temp;
    Mat reconPoints;
    triangulatePoints(P1, P2, pts1, pts2, reconPoints);
    cout << reconPoints.rows << reconPoints.cols << endl;
    vector<Point3d> triPts;
    for (int i = 0; i < reconPoints.cols; i++) {
        Mat point = reconPoints.col(i); //4x1
        point = point / point.at<double>(3, 0);//
        Point3d pt;
        pt.x = point.at<double>(0, 0);
        pt.y = point.at<double>(1, 0);
        pt.z = point.at<double>(2, 0);
        triPts.push_back(pt);
    }
    for (const auto& pt : triPts) {
        cout << pt << endl;
    }
    //PnP problem
    //Given X2 and x2 correspondences, estimate R and t
    Mat R_pnp, t_pnp;
    solvePnP(triPts, pts2, K, Mat(), R_pnp, t_pnp);
    cout << R_pnp << endl; //x, y, z euler angle (rad)
    cout << t_pnp << endl; //x, y, z


}