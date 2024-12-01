/*
g++ -std=c++17 -o ceres_06_ba ./ceres_06_ba.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

//input, fx, fy, cx, cy
//2d points (u, v) and their correspondecnes 3D points (X,Y,Z)

//output, refined intrinsics, pose, 3D points

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <iostream>


using namespace std;
// struct pose {
//     double t[3];
//     double quat[4];
// };
// struct Residual {
//     Residual(pose pose, Point2d pt2d, Point3d pt3d, double* intrinsic) : 
//     pose_(pose), pt2d_(pt2d), pt3d_(pt3d), intrinsic_(intrinsic) {}
//     template<typename T>
//     bool operator()(const T* params, T* residual) const {
//         double fx = intrinsic_[0];
//         double fy = intrinsic_[1];
//         double cx = intrinsic_[2];
//         double cy = intrinsic_[3];

//         double X = point3d_.x;
//         double Y = point3d_.y;
//         double Z = point3d_.z;

//         double u = pt2d.x;
//         double v = pt2d.y;
        

//         residual[0] = T(u) - ((T(fx) * T(X) + T(cx)) / T(Z));
//         residual[2] = T(v) - ((T(fy) * T(Y) + T(cy)) / T(Z));
//         return true;
//     };
// private:
//     pose pose_;
//     Point2d pt2d_;
//     Point3d pt3d_;
//     double intrinsic_[4];
// };

struct ReprojErr{
    ReprojErr(double input_u, double input_v) : measured_u(input_u), measured_v(input_v){}
    template<typename T>
    bool operator()(const T* intrinsics,
                    const T* trans,
                    const T* rot,
                    const T* point3D, T* residual) const {
                        T p[3]; //a 3D point
                        ceres::AngleAxisRotatePoint(rot, point3D, p);
                        p[0] += trans[0];
                        p[1] += trans[1];
                        p[2] += trans[2];

                        T px = p[0] / p[2];
                        T py = p[1] / p[2];
                        T fx = intrinsics[0];
                        T fy = intrinsics[1];
                        T cx = intrinsics[2];
                        T cy = intrinsics[3];


                        T pred_u = T(fx) * T(px) + T(cx);
                        T pred_v = T(fy) * T(py) + T(cy);
                        residual[0] = T(measured_u) - pred_u;
                        residual[1] = T(measured_v) - pred_v;
                        return true;
                    }
    static ceres::CostFunction* Create(double u, double v) {
        return (new ceres::AutoDiffCostFunction<ReprojErr, 2, 4, 3, 3, 3>(new ReprojErr(u, v)));
    }
private:
    double measured_u;
    double measured_v;
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
    /////////////////////////////////////////

    ceres::Problem problem;

    //Add parameters
    problem.AddParameterBlock(intrinsics, 4);
    for (int i = 0; i < N; i++) {
        problem.AddParameterBlock(cam_trans[i], 3);
        problem.AddParameterBlock(cam_rot[i], 3);
    }
    for (int i = 0; i < M; i++) {
        problem.AddParameterBlock(point3D[i], 3);
    }

    //add residual
    for (const auto& obs : observations) {
        ceres::CostFunction* fn = ReprojErr::Create(obs.u, obs.v);
        problem.AddResidualBlock(fn, NULL,
                                  intrinsics,
                                  cam_trans[obs.camIdx],
                                  cam_rot[obs.camIdx],
                                  point3D[obs.pointIdx]
                                  );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
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
