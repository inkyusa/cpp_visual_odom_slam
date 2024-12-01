/*
g++ -std=c++17 -o ceres_05_fit_exp ./ceres_05_fit_exp.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

#include<ceres/ceres.h>
#include<vector>
#include<iostream>
#include<utility>

using namespace std;
struct Residual{
    Residual(double x, double y) : x_(x), y_(x) {}
    template<typename T>
    bool operator()(const T* m, T* residual) const {
        residual[0] = T(y_) - (T(m[0]) * ceres::exp(T(m[1]) * T(x_)));
        return true;
    };
private:
    double x_;
    double y_;
};
int main(void) {
    std::vector<std::pair<double, double>> points = {{0, 1}, {1, 2.718}, {2, 7.389}, {3, 20.085}};

    double m[2] = {0.0, 0.0};//initial params
    ceres::Problem problem;
    for (int i = 0; i < points.size(); i++) {
        ceres::CostFunction* fn = 
        new ceres::AutoDiffCostFunction<Residual, 1, 2>(new Residual(points[i].first, points[i].second));
        problem.AddResidualBlock(fn, NULL, m);
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    cout << "m[0] =" << m[0] << " m[1] = " << m[1] << endl;
    //m[0] =0.486494 m[1] = 0.619855
    return 0;
}

