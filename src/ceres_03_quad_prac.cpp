/*
g++ -std=c++17 -o ceres_03_quad_prac ./ceres_03_quad_prac.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann -lceres -lglog -lprotobuf -pthread
*/

#include <ceres/ceres.h>
#include <iostream>
#include <vector>

// r(x) = x - 5;
using namespace std;

double model(double x) {
    return (double)(x - 5);
}
struct Residual {
    Residual(double x, double y) : x_(x), y_(y) {}
    template<typename T>
    bool operator() (const T* m, T* residual) const {
        residual[0] = T(y_) - ( T(m[0]) * T(x_) + T(m[1]));
        return true;
    };
private:
    double x_;
    double y_;
};

int main(void) {
    vector<double> x_data, y_data;
    for (int i = -50; i < 50; i++) {
        x_data.push_back(double(i));
        y_data.push_back(model(i));
    }
    int n = x_data.size();
    ceres::Problem problem;
    double m[2] = {0.0, 0.0}; //initial value;
    for (int i = 0; i < n; i++) {
        ceres::CostFunction* fn = new ceres::AutoDiffCostFunction<Residual, 1, 2> (new Residual(x_data[i], y_data[i]));
        problem.AddResidualBlock(fn, NULL, m);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    cout << m[0] << "," << m[1] << endl; //it prints 1, -5

    return 0;
}