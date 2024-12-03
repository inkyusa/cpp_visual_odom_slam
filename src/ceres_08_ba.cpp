/*
g++ -std=c++17 -o ceres_08_ba ./ceres_08_ba.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

//input, fx, fy, cx, cy
//2d points (u, v) and their correspondecnes 3D points (X,Y,Z)

//output, refined intrinsics, pose, 3D points

#include <iostream>
#include <vector>
#include <random>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>



using namespace std;
using namespace Eigen;

struct ReprjError{

    ReprjError(double in_u, double in_v) : measured_u_(in_u), measured_v_(in_v) {}
    template<typename T>
    bool operator()(const T* cam_rot,
                    const T* cam_trans,
                    const T* point3D,
                    const T* intrinsics, T* residuals) const {
                        Matrix<T, 3, 3> R;
                        ceres::AngleAxisToRotationMatrix(cam_rot, R.data());
                        Matrix<T, 3, 3> K;
                        K << intrinsics[0], T(0), intrinsics[2],
                             T(0), intrinsics[1], intrinsics[3],
                             T(0), T(0), T(1);
                        Matrix<T, 3, 1> t;
                        t << cam_trans[0], cam_trans[1], cam_trans[2];
                        Matrix<T, 3, 4> P; //projection matrix
                        P.template block<3, 3>(0, 0) = R;
                        P.template block<3, 1>(0, 3) = t;

                        P = K * P; // K*[R | t]; 3x4
                        Matrix<T, 4, 1> X;
                        X << point3D[0], point3D[1], point3D[2], T{1};
                        Matrix<T, 3, 1> x_prj;
                        x_prj = P * X;
                        x_prj /= x_prj(2, 0);

                        residuals[0] = T(measured_u_) - T(x_prj(0, 0));
                        residuals[1] = T(measured_v_) - T(x_prj(1, 0));
                        return true;
                    }
    static ceres::CostFunction* create(double u, double v) {
        ceres::CostFunction* fn = new ceres::AutoDiffCostFunction<ReprjError, 2, 3, 3, 3, 4>(new ReprjError(u, v));
        return fn;
    }
    private:
    double measured_u_;
    double measured_v_;
};

struct observation {
    int camIdx;
    int pointIdx;
    double u;
    double v;
};

int main(void) {

    /////////////////////////////////////////
    //Assume these are given
    const int N = 2; //camera pose
    const int M = 5; //# of 3D point in world
    double intrinsics[4] = {400.0, 400.0, 200.0, 200.0}; //fx, fy, cx, cy
    double cam_rot[N][3] = {{0, 0, 0}, {0, 0, 0}};
    double cam_trans[N][3] = {{0, 0, 0}, {1, 0, 0}};
    double point3D[M][3] = {
        {0.5, 0.5, 5.0},
        {-0.5, -0.5, 6.0},
        {1.0, -1.0, 7.0},
        {-1.0, 1.0, 8.0},
        {0.0, 0.0, 9.0}
    };

    vector<observation> observations;
    for (int cam_idx = 0; cam_idx < N; ++cam_idx) {
        for (int pt_idx = 0; pt_idx < M; ++pt_idx) {
            double p[3];
            ceres::AngleAxisRotatePoint(cam_rot[cam_idx], point3D[pt_idx], p);
            p[0] += cam_trans[cam_idx][0];
            p[1] += cam_trans[cam_idx][1];
            p[2] += cam_trans[cam_idx][2];

            double xp = p[0] / p[2];
            double yp = p[1] / p[2];

            double predicted_u = intrinsics[0] * xp + intrinsics[2];
            double predicted_v = intrinsics[1] * yp + intrinsics[3];

            // Add noise
            double noise_u = ((rand() % 1000) / 10000.0) - 0.05;
            double noise_v = ((rand() % 1000) / 10000.0) - 0.05;

            observations.push_back({cam_idx, pt_idx, predicted_u + noise_u, predicted_v + noise_v});
        }
    }

    ceres::Problem problem;
    //add parameter blocks
    for (int i = 0; i < N; i++) {
        problem.AddParameterBlock(cam_rot[i], 3);
        problem.AddParameterBlock(cam_trans[i], 3);
    }
    for (int i = 0; i < M; i++) {
        problem.AddParameterBlock(point3D[i], 3);
    }
    problem.AddParameterBlock(intrinsics, 4);

    //add residual blocks
    for (const auto& obs : observations) {
        ceres::CostFunction* fn = ReprjError::create(obs.u, obs.v);
        problem.AddResidualBlock(fn, NULL, 
                                    cam_rot[obs.camIdx],
                                    cam_trans[obs.camIdx],
                                    point3D[obs.pointIdx],
                                    intrinsics
                                    );
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;









    ////////////////////////////////
    // Assume these print provided
    std::cout << "Estimated Camera Intrinsics:\n";
    std::cout << "fx: " << intrinsics[0] << "\n";
    std::cout << "fy: " << intrinsics[1] << "\n";
    std::cout << "cx: " << intrinsics[2] << "\n";
    std::cout << "cy: " << intrinsics[3] << "\n";

    for (int i = 0; i < N; ++i) {
        std::cout << "Camera " << i << " Rotation (angle-axis): "
                  << cam_rot[i][0] << ", "
                  << cam_rot[i][1] << ", "
                  << cam_rot[i][2] << "\n";
        std::cout << "Camera " << i << " Translation: "
                  << cam_trans[i][0] << ", "
                  << cam_trans[i][1] << ", "
                  << cam_trans[i][2] << "\n";
    }

    for (int i = 0; i < M; ++i) {
        std::cout << "Point " << i << " Position: "
                  << point3D[i][0] << ", "
                  << point3D[i][1] << ", "
                  << point3D[i][2] << "\n";
    }
    /////////
    return 0;
}
