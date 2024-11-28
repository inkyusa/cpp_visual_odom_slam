/*
g++ -std=c++17 -o geometry_practice_cv ./geometry_practice_cv.cpp \
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
double matDiff(const Mat& mat1, const Mat& mat2) {
    if (mat1.size() == mat2.size() && mat1.type() == mat2.type()) {
        return norm(mat1, mat2, NORM_L2); // Or cv::NORM_L1, cv::NORM_INF
    } else {
        throw std::invalid_argument("Matrices must have the same size and type.");
    }
}

pair<Mat, Mat> generateCorrespondences(vector<Point2d>& pt1, vector<Point2d>& pt2, Mat& K, vector<Point3d>& worldPoints){
    // 1. Simulate 3D Points in World Coordinates
    // vector<Point3d> worldPoints;
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
    double angle_x = 15 * CV_PI / 180.0; // Rotate 5 degrees around Y-axis
    double angle_y = 7 * CV_PI / 180.0; // Rotate 5 degrees around Y-axis
    double angle_z = 30 * CV_PI / 180.0; // Rotate 5 degrees around Y-axis
    Mat R2 = createRotationMatrix(angle_x, angle_y, angle_z);
    Mat t2 = (Mat_<double>(3, 1) << 0.5, 1, 1.3); // Translate 0.5 units along X-axis

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
    vector<Point3d> worldPoints;
    auto[R_gt, t_gt] = generateCorrespondences(pts1, pts2, K, worldPoints);
    cout << "# of correspondences = " << pts1.size() << endl;


    // 1. estimate E given correspondences
    Mat E = findEssentialMat(pts1, pts2, K, RANSAC, 0.999, 1);
    cout << "E=" << E << endl;

    Mat R_est, t_est;
    recoverPose(E, pts1, pts2, K, R_est, t_est);
    cout << "\nRecovered Rotation R:\n" << R_est << endl;
    cout << "\nRecovered Translation t:\n" << t_est << endl; 

    cout << "\nGT Rotation R:\n" << R_gt << endl;
    cout << "\nGT Translation t:\n" << t_gt << endl;
    cout << "R diff = " << matDiff(R_gt, R_est) << endl; //this should be very close
    cout << "t diff = " << matDiff(t_gt, t_est) << endl; //This can't be same as ground truth


    // 2. Triangulate Points
    // Projection matrices for triangulation
    Mat P1 = K * Mat::eye(3, 4, CV_64F);
    Mat Rt;
    hconcat(R_gt, t_gt, Rt); // GT rotation and translation
    Mat P2 = K * Rt;
    // Convert points to homogeneous coordinates for triangulation
    vector<Point2f> pts1_homo, pts2_homo;
    for (size_t i = 0; i < pts1.size(); ++i) {
        pts1_homo.push_back(pts1[i]);
        pts2_homo.push_back(pts2[i]);
    }

    Mat points4D;
    triangulatePoints(P1, P2, pts1_homo, pts2_homo, points4D);
    // Convert homogeneous coordinates to 3D points
    vector<Point3d> triangulatedPoints;
    for (int i = 0; i < points4D.cols; ++i) {
        Mat col = points4D.col(i);
        col /= col.at<float>(3, 0); // Normalize
        Point3d pt;
        pt.x = col.at<float>(0, 0);
        pt.y = col.at<float>(1, 0);
        pt.z = col.at<float>(2, 0);
        triangulatedPoints.push_back(pt);
    }
    // Display triangulated points
    cout << "\nTriangulated 3D Points:" << endl;
    for (size_t i = 0; i < triangulatedPoints.size(); ++i) {
        cout << "GT" << worldPoints[i] << ", estimate" << triangulatedPoints[i] << endl;
    }

    // 8. Perform PnP Algorithm to Estimate Camera Pose at t=2
    // Use triangulated 3D points and their 2D projections at t=2
    Mat Rvec, tvec;
    solvePnP(worldPoints, pts2, K, Mat(), Rvec, tvec);

    // Convert rotation vector to matrix
    Mat R_pnp;
    Rodrigues(Rvec, R_pnp);

    cout << "\nPnP Estimated Rotation R_pnp:\n" << R_pnp << endl;
    cout << "\nPnP Estimated Translation t_pnp:\n" << tvec << endl;

    // 9. Compare Estimated Pose with Ground Truth
    cout << "\nGround Truth Rotation R_gt:\n" << R_gt << endl;
    cout << "\nGround Truth Translation t_gt:\n" << t_gt << endl;

}