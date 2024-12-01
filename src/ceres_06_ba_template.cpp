/*
g++ -std=c++17 -o ceres_06_ba_template ./ceres_06_ba_template.cpp \
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
