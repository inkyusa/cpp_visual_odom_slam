#include <ceres/ceres.h>
#include <vector>
#include <iostream>

struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const m, T* residual) const {
        residual[0] = T(y_) - ceres::exp(m[0] * T(x_) + m[1]);
        return true;
    }
private:
    const double x_;
    const double y_;
};

int main() {
    // Data points
    std::vector<double> x_data, y_data;
    // ... (Fill x_data and y_data with your data)

    // Initial estimates for parameters m[0] and m[1]
    double m[2] = {0.0, 0.0};

    ceres::Problem problem;
    for (size_t i = 0; i < x_data.size(); ++i) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 2>(
                new ExponentialResidual(x_data[i], y_data[i]));
        problem.AddResidualBlock(cost_function, nullptr, m);
    }

    // Solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";
    std::cout << "Estimated parameters: m[0]=" << m[0] << ", m[1]=" << m[1] << std::endl;
    return 0;
}
