/*
g++ -std=c++17 -o ceres_04_fit_line ./ceres_04_fit_line.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann -lceres -lglog -lprotobuf -pthread
*/

#include <ceres/ceres.h>
#include <vector>
#include <iostream>
struct Residual {
    Residual(double x, double y) : x_(x), y_(y) {}
    template<typename T>
    bool operator()(const T* theta, T* residual) const {
        residual[0] =  T(y_) - (T(theta[0]) * T(x_) + T(theta[1]));
        return true;
    }
private:
    double x_;
    double y_;
};
using namespace std;
int main(void) {
    vector<pair<double, double>> points = {{0, 1}, {1, 2}, {2, 2.8}, {3, 4.1}};
    double theta[2] = {0, 0};
    ceres::Problem problem;
    for (int i = 0; i < points.size(); i++) {
        ceres::CostFunction* fn = new ceres::AutoDiffCostFunction<Residual, 1, 2>(new Residual(points[i].first, points[i].second));
        problem.AddResidualBlock(fn, NULL, theta);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    cout << "theta[0] = " << theta[0] << " theta[1] = " << theta[1] << endl;
    //theta[0] = 1.01 theta[1] = 0.96
    return 0;
}


