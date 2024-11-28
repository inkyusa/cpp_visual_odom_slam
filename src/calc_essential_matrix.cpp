/*
g++ -std=c++17 -o essentialMatrix ./calc_essential_matrix.cpp \
    -I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
    -L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d
*/
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat calcEssentialMatrix(vector<Point2f>& p1, vector<Point2f>& p2, Mat& K) {
    Mat F = findFundamentalMat(p1, p2, FM_8POINT);
    cout << "K type: " << K.type() << endl;
    cout << "F type: " << F.type() << endl;
    cout << "K.t() type: " << K.t().type() << endl;
    Mat E = K.t() * F * K;
    return E;
}

int main(void) {
    vector<Point2f> points1 = {
    {400, 400},
    {480, 400},
    {560, 400},
    {400, 480},
    {480, 480},
    {560, 480},
    {400, 560},
    {480, 560}
    };
    vector<Point2f> points2 = {
    {401.25, 400},
    {481.25, 400},
    {561.25, 400},
    {401.25, 480},
    {481.25, 480},
    {561.25, 480},
    {401.25, 560},
    {481.25, 560}
    };    
    Mat K = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    Mat E = calcEssentialMatrix(points1, points2, K);
    cout << "E = " << E << endl;
    return 0;
}


/*
// Function to compute the essential matrix using the eight-point algorithm
Matrix3d computeEssentialMatrix(const std::vector<Vector3d>& points1, const std::vector<Vector3d>& points2) {
    assert(points1.size() == points2.size() && points1.size() >= 8);

    int N = points1.size();
    MatrixXd A(N, 9);

    // Construct matrix A
    for (int i = 0; i < N; ++i) {
        double x1 = points1;
        double y1 = points1;
        double x2 = points2;
        double y2 = points2;

        A(i, 0) = x1 * x2;
        A(i, 1) = x1 * y2;
        A(i, 2) = x1;
        A(i, 3) = y1 * x2;
        A(i, 4) = y1 * y2;
        A(i, 5) = y1;
        A(i, 6) = x2;
        A(i, 7) = y2;
        A(i, 8) = 1.0;
    }

    // Solve for the null space of A (the smallest singular value)
    JacobiSVD<MatrixXd> svd(A, ComputeFullV);
    VectorXd f = svd.matrixV().col(8);

    // Reshape f into a 3x3 matrix
    Matrix3d E;
    E << f(0), f(1), f(2),
         f(3), f(4), f(5),
         f(6), f(7), f(8);

    // Enforce the rank-2 constraint on E by performing SVD and setting the smallest singular value to zero
    JacobiSVD<MatrixXd> svd_E(E, ComputeFullU | ComputeFullV);
    Vector3d singular_values = svd_E.singularValues();
    singular_values(2) = 0.0; // Set the smallest singular value to zero
    E = svd_E.matrixU() * singular_values.asDiagonal() * svd_E.matrixV().transpose();

    return E;
}

// Example usage
int main() {
    // Example normalized correspondences (homogeneous coordinates)
    std::vector<Vector3d> points1 = {
        {0.5, 0.3, 1.0}, {0.6, 0.2, 1.0}, {0.4, 0.5, 1.0}, {0.7, 0.4, 1.0},
        {0.3, 0.6, 1.0}, {0.2, 0.7, 1.0}, {0.8, 0.1, 1.0}, {0.9, 0.2, 1.0}
    };

    std::vector<Vector3d> points2 = {
        {0.52, 0.31, 1.0}, {0.61, 0.22, 1.0}, {0.43, 0.53, 1.0}, {0.69, 0.42, 1.0},
        {0.31, 0.62, 1.0}, {0.19, 0.68, 1.0}, {0.81, 0.12, 1.0}, {0.88, 0.21, 1.0}
    };

    Matrix3d E = computeEssentialMatrix(points1, points2);

    std::cout << "Essential Matrix E:\n" << E << std::endl;

    return 0;
}
*/