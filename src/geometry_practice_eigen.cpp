/*
g++ -std=c++17 -o geometry_practice_eigen ./geometry_practice_eigen.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-I /usr/local/include/eigen3 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

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

Mat findE(vector<Point2d>& pts1, vector<Point2d>& pts2) {

    Mat T1, T2;
    vector<Point2d> pts1_norm;
    vector<Point2d> pts2_norm;

    int N = pts1.size();
    MatrixXd A(N, 9);
    for (int i = 0; i < N; i++) {
        double x1 = pts1[i].x;
        double y1 = pts1[i].y;
        double x2 = pts2[i].x;
        double y2 = pts2[i].y;
        A(i, 0) = x2 * x1;
        A(i, 1) = x2 * y1;
        A(i, 2) = x2;
        A(i, 3) = y2 * x1;
        A(i, 4) = y2 * y1;
        A(i, 5) = y2;
        A(i, 6) = x1;
        A(i, 7) = y1;
        A(i, 8) = 1.0;
    }
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    VectorXd e = svd.matrixV().col(8);
    Matrix3d E;
    E << e(0), e(1), e(2), e(3), e(4), e(5),e(6), e(7), e(8);
    JacobiSVD<MatrixXd> svd_E(E, ComputeFullU | ComputeFullV);
    VectorXd singular_values = svd_E.singularValues();
    singular_values(2) = 0; //enforce rank2
    Matrix3d E_ = svd_E.matrixU() * singular_values.asDiagonal() * svd_E.matrixV().transpose();

    MatrixXd x1(3,1);
    x1 << pts1[0].x, pts1[0].y, 1;
    MatrixXd x2(3,1);
    x2 << pts2[0].x, pts2[0].y, 1;
    cout <<"Verification" << x2.transpose() * E_ * x1 << endl;

    Mat E_cv;
    eigen2cv(E, E_cv);
    return E_cv;
}
Mat getSkew(Point2d& pt) {
    Mat ret = (Mat_<double>(3, 3) << 0 , 1, -pt.y, -1, 0, pt.x, pt.y, -pt.x, 0);
    return ret;
}
int main(void) {
    Mat K = (Mat_<double>(3,3) << 800, 0, 320,
        0, 800, 240,
        0, 0, 1);
    MatrixXd K_eigen;
    cv2eigen(K, K_eigen);
    vector<Point2d> pts1, pts2;
    vector<Point3d> worldPoints;
    auto[R_gt, t_gt] = generateCorrespondences(pts1, pts2, K, worldPoints);
    cout << "# of correspondences = " << pts1.size() << endl;


    // 1. estimate E given correspondences
    Mat E_eigen = findE(pts1, pts2);
    cout << "E_eigen=" << E_eigen << endl;

    Mat E_cv = findEssentialMat(pts1, pts2, K, RANSAC, 0.999, 1);
    cout << "E_cv=" << E_cv << endl;


    // Mat R_est, t_est;
    // recoverPose(E, pts1, pts2, K, R_est, t_est);
    // cout << "\nRecovered Rotation R:\n" << R_est << endl;
    // cout << "\nRecovered Translation t:\n" << t_est << endl; 

    // cout << "\nGT Rotation R:\n" << R_gt << endl;
    // cout << "\nGT Translation t:\n" << t_gt << endl;
    // cout << "R diff = " << matDiff(R_gt, R_est) << endl; //this should be very close
    // cout << "t diff = " << matDiff(t_gt, t_est) << endl; //This can't be same as ground truth


    // // 2. Triangulate Points
    // // Projection matrices for triangulation
    Mat P1 = K * Mat::eye(3, 4, CV_64F); //3x4
    Mat P2; //3x4
    hconcat(R_gt, t_gt, P2);
    P2 = K * P2;
    vector<Vector3d> worldPoints_esti;
    for (int i = 0; i < pts1.size(); i++) {
        Mat x1 = getSkew(pts1[i]);
        Mat x2 = getSkew(pts2[i]);
        Mat A;
        vconcat(x1 * P1, x2 * P2, A); //construct 3*(# of views) x 4 matrix, A
        MatrixXd A_eigen;
        cv2eigen(A, A_eigen);
        JacobiSVD<MatrixXd> svd(A_eigen, ComputeFullU | ComputeFullV);
        Vector4d X = svd.matrixV().col(3);
        X = X / X(3); //last should be 1
        Vector3d X_esti_eigen = X.head<3>();
        cout << "Global point = " << worldPoints[i] << " triangulated = " << X_esti_eigen << endl;
        worldPoints_esti.push_back(X_esti_eigen);
    }

    //Pnp algorithm
    //normalise pts2
    vector<Point2d> pts2_norm;
    Mat K_inv = K.inv();
    for (const auto& pt : pts2) {
        Mat pt_homog = (Mat_<double>(3,1) << pt.x, pt.y, 1.0);
        Mat pt_norm = K_inv * pt_homog;
        pts2_norm.push_back(Point2d(pt_norm.at<double>(0), pt_norm.at<double>(1)));
    }

    int N = pts2.size();
    MatrixXd A_concat(2 * N, 12);

    for (int i = 0; i < N; i++) {
        double X = worldPoints_esti[i](0);
        double Y = worldPoints_esti[i](1);
        double Z = worldPoints_esti[i](2);
        double u = pts2_norm[i].x;
        double v = pts2_norm[i].y;

        // First equation
        A_concat(2 * i, 0) = X;
        A_concat(2 * i, 1) = Y;
        A_concat(2 * i, 2) = Z;
        A_concat(2 * i, 3) = 1;
        A_concat(2 * i, 4) = 0;
        A_concat(2 * i, 5) = 0;
        A_concat(2 * i, 6) = 0;
        A_concat(2 * i, 7) = 0;
        A_concat(2 * i, 8) = -u * X;
        A_concat(2 * i, 9) = -u * Y;
        A_concat(2 * i, 10) = -u * Z;
        A_concat(2 * i, 11) = -u;

        // Second equation
        A_concat(2 * i + 1, 0) = 0;
        A_concat(2 * i + 1, 1) = 0;
        A_concat(2 * i + 1, 2) = 0;
        A_concat(2 * i + 1, 3) = 0;
        A_concat(2 * i + 1, 4) = X;
        A_concat(2 * i + 1, 5) = Y;
        A_concat(2 * i + 1, 6) = Z;
        A_concat(2 * i + 1, 7) = 1;
        A_concat(2 * i + 1, 8) = -v * X;
        A_concat(2 * i + 1, 9) = -v * Y;
        A_concat(2 * i + 1, 10) = -v * Z;
        A_concat(2 * i + 1, 11) = -v;
    }

    JacobiSVD<MatrixXd> svd_pnp(A_concat, ComputeFullU | ComputeFullV);
    VectorXd Pvec = svd_pnp.matrixV().col(11); // Last column
    MatrixXd P(3, 4);
    P.row(0) = Pvec.segment<4>(0); //get 4 element from 0 idx,
    P.row(1) = Pvec.segment<4>(4);
    P.row(2) = Pvec.segment<4>(8);


    MatrixXd Rt = P;
    MatrixXd R_pnp = Rt.block<3,3>(0,0);
    VectorXd t_pnp = Rt.col(3);

    // Ensure R_pnp is a valid rotation matrix
    JacobiSVD<MatrixXd> svd_R(R_pnp, ComputeFullU | ComputeFullV);
    R_pnp = svd_R.matrixU() * svd_R.matrixV().transpose();

    // Adjust the scale
    double det = R_pnp.determinant();
    if (det < 0) {
        R_pnp = -R_pnp;
        t_pnp = -t_pnp;
    }
    double scale = t_gt.at<double>(0) / t_pnp(0);
    t_pnp = scale * t_pnp;
    cout << "scale = " << scale << endl;

    cout << "R_pnp:\n" << R_pnp << endl;
    cout << "t_pnp:\n" << t_pnp << endl;

    cout << "\nGround Truth Rotation R_gt:\n" << R_gt << endl;
    cout << "\nGround Truth Translation t_gt:\n" << t_gt << endl;
}