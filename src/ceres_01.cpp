/*
g++ -std=c++17 -o ceres_01 ./ceres_01.cpp \
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
    return 3 * x * x + 2;
}
struct LSError {
    LSError(double x_in, double y_in){
        x = x_in;
        y = y_in;
    }
    template <typename T>
    bool operator()(const T* m, T* residual) const {
        residual[0] = T(y) - (m[0] * T(x) * T(x) + m[1]);
        return true;
    }
    //operator(): This is a function call operator, allowing instances
    //of the LSError struct to be used like a function. 
    //first const 
    //last const: Marks the method as read-only, 
    //meaning it does not modify any member variables of the struct.
double x;
double y;

};
int main() {
    // Data points
    std::vector<double> x_data, y_data;
    for (int i = -10; i < 10; i++) {
        x_data.push_back(i);
        y_data.push_back(model(i));
    }

    // Initial estimates for parameters m[0] and m[1]
    double m[2] = {0.0, 0.0};
    ceres::Problem problem;

    for (int i = 0; i < x_data.size(); i++) {
        ceres::CostFunction* fn = new ceres::AutoDiffCostFunction<
            LSError, 1, 2>(new LSError(x_data[i], y_data[i]));
        //1 indicate: The number of residuals produced by this cost function. 
        //Since the error for each data point is scalar, this is 1. 
        //2: The number of parameters being optimized. In this case, 
        //the model parameters m[0] (coefficient) and m[1]
        problem.AddResidualBlock(fn, NULL, m);
        //NULL, This represents the loss function.
        //Passing NULL means no robust loss is applied, 
        //so a standard squared error is used. residual^2

    }
    

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    cout << m[0] << "," << m[1] << endl;
}
