/*
g++ -std=c++17 -o ceres_02 ./ceres_02.cpp \
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

using namespace std;

double model(double x) {
    return 5 * x * x + 4;
}
struct Residual {
    Residual(double x, double y) : x_(x), y_(y) {}
    template<typename T>
    bool operator() (const T* m, T* residual) const {
        residual[0] = T(y_) - (T(m[0]) * T(x_) * T(x_) + T(m[1]));
        return true;
    };

private:
    double x_;
    double y_;
};
int main(void) {

    vector<double> x_data, y_data;
    int n = 100;
    for (int i = -100; i < 100; i++) {
        x_data.push_back(double(i));
        y_data.push_back(model(i));
    }
    double m[2] = {0.0, 0.0}; //initial params
    ceres::Problem problem;
    for (int i = 0; i < x_data.size(); i++) {
        ceres::CostFunction* fn = new ceres::AutoDiffCostFunction<Residual, 1, 2>(new Residual(x_data[i], y_data[i]));
        problem.AddResidualBlock(fn, NULL, m);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    cout << summary.FullReport() << endl;
    cout << m[0] << "," << m[1] <<endl;


    return 0;
}

