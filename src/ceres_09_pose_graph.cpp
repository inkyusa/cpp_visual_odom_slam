/*
g++ -std=c++17 -o ceres_09_pose_graph ./ceres_09_pose_graph.cpp \
-I /usr/local/Cellar/ceres-solver/2.2.0_1/include/ceres \
-I /usr/local/include/eigen3 \
-L /usr/local/Cellar/ceres-solver/2.2.0_1/lib \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lceres -lglog -lprotobuf -pthread
*/

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <fstream>


// Normalizes the angle in radians between [-pi and pi).
template <typename T>
inline T NormalizeAngle(const T& angle_radians) {
  // Use ceres::floor because it is specialized for double and Jet types.
  T two_pi(2.0 * ceres::constants::pi);
  return angle_radians -
         two_pi * ceres::floor((angle_radians + T(ceres::constants::pi)) / two_pi);
}


using namespace std;
const double positionNoiseStd = 0.05;  // meters
const double orientationNoiseStd = (0.5 * M_PI) / 180.0;  // radians
std::default_random_engine generator;
// Random number generator for Gaussian noise
std::normal_distribution<double> positionNoise(0.0, positionNoiseStd);
std::normal_distribution<double> orientationNoise(0.0, orientationNoiseStd);


struct Pose2D {
    int id;
    double x;
    double y;
    double theta;  // Orientation in radians
};

struct Constraint2D {
    int id_begin;      // ID of the first pose
    int id_end;        // ID of the second pose
    Eigen::Vector3d t; // Relative transformation [dx, dy, dtheta]
    Eigen::Matrix3d information; // Information matrix (inverse of covariance)
};

// vector<Pose2D> generateGTPoses() {
//     const int numPoses = 100;
//     std::vector<Pose2D> poses;

//     // Segment lengths
//     const double lengths[] = {30.0, 20.0, 30.0, 20.0};  // meters
//     const int numSegments = sizeof(lengths) / sizeof(lengths[0]);
//     const int posesPerSegment = numPoses / numSegments;

//     // Starting position
//     double x = 0.0, y = 0.0, theta = 0.0;

//     int poseId = 0;
//     for (int i = 0; i < numSegments; ++i) {
//         double dx = (i % 2 == 0) ? lengths[i] : 0.0;
//         double dy = (i % 2 == 1) ? lengths[i] : 0.0;
//         theta = (M_PI / 2.0) * i;  // Orientations: 0, 90, 180, 270 degrees

//         int steps = posesPerSegment;
//         double stepSize = lengths[i] / steps;

//         for (int j = 0; j < steps; ++j) {
//             // Incremental position
//             if (dx != 0.0) x += stepSize;
//             if (dy != 0.0) y += stepSize;

//             Pose2D pose;
//             pose.id = poseId++;
//             pose.x = x;
//             pose.y = y;
//             pose.theta = theta;
//             poses.push_back(pose);
//         }
//     }

//     // Handle remaining poses if any
//     while (poses.size() < numPoses) {
//         poses.push_back(poses.back());
//     }
//     return poses;
// }

vector<Pose2D> generateGTPoses() {
    const int numPoses = 300;       // Total number of poses
    const int numLoops = 3;         // Number of loops in the trajectory
    const double loopRadius = 10.0; // Radius of each loop (meters)
    std::vector<Pose2D> poses;

    // Loop through each segment of the trajectory
    int poseId = 0;
    double x = 0.0, y = 0.0, theta = 0.0;

    for (int loop = 0; loop < numLoops; ++loop) {
        double offsetX = loop * 2.5 * loopRadius; // Offset each loop in the x direction
        for (int i = 0; i < numPoses / numLoops; ++i) {
            // Generate a smooth circular motion within each loop
            double angle = (2.0 * M_PI * i) / (numPoses / numLoops);
            x = offsetX + loopRadius * cos(angle);
            y = loopRadius * sin(angle);
            theta = angle + M_PI / 2.0; // Orientation tangential to the circle

            // Store the generated pose
            Pose2D pose;
            pose.id = poseId++;
            pose.x = x;
            pose.y = y;
            pose.theta = theta;
            poses.push_back(pose);
        }
    }

    // Add loop closure points (revisit starting points with slight offsets)
    for (int loop = 1; loop < numLoops; ++loop) {
        Pose2D loopClosurePose = poses[(loop - 1) * (numPoses / numLoops)];
        loopClosurePose.x += 0.1;  // Add small noise for realism
        loopClosurePose.y -= 0.1;
        loopClosurePose.theta += 0.05;
        poses.push_back(loopClosurePose);
    }

    return poses;
}



vector<Constraint2D> generateOdom(vector<Pose2D>& poses) {
    // Noise parameters
    
    std::vector<Constraint2D> constraints;
    for (size_t i = 0; i < poses.size() - 1; ++i) {
        const Pose2D& pose1 = poses[i];
        const Pose2D& pose2 = poses[i + 1];

        // Compute relative motion
        double dx = pose2.x - pose1.x;
        double dy = pose2.y - pose1.y;
        double dtheta = pose2.theta - pose1.theta;

        // Add noise
        dx += positionNoise(generator);
        dy += positionNoise(generator);
        dtheta += orientationNoise(generator);

        // Create constraint
        Constraint2D constraint;
        constraint.id_begin = pose1.id;
        constraint.id_end = pose2.id;
        constraint.t = Eigen::Vector3d(dx, dy, dtheta);

        // Information matrix (inverse of covariance)
        Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
        information(0, 0) = 1.0 / (positionNoiseStd * positionNoiseStd);
        information(1, 1) = 1.0 / (positionNoiseStd * positionNoiseStd);
        information(2, 2) = 1.0 / (orientationNoiseStd * orientationNoiseStd);

        constraint.information = information;
        constraints.push_back(constraint);
    }
    return constraints;
}

vector<Constraint2D> generateLoopClosure(vector<Pose2D>& poses) {
    std::vector<Constraint2D> constraints;
    // Poses involved in the loop closure
    const Pose2D& poseStart = poses.front();  // Starting pose
    const Pose2D& poseEnd = poses.back();     // Ending pose

    // Compute relative motion
    double dx = poseStart.x - poseEnd.x;
    double dy = poseStart.y - poseEnd.y;
    double dtheta = poseStart.theta - poseEnd.theta;
    
    // Add noise
    dx += positionNoise(generator);
    dy += positionNoise(generator);
    dtheta += orientationNoise(generator);

    // Create loop closure constraint
    Constraint2D loopClosureConstraint;
    loopClosureConstraint.id_begin = poseEnd.id;
    loopClosureConstraint.id_end = poseStart.id;
    loopClosureConstraint.t = Eigen::Vector3d(dx, dy, dtheta);

    // Information matrix
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    information(0, 0) = 1.0 / (positionNoiseStd * positionNoiseStd);
    information(1, 1) = 1.0 / (positionNoiseStd * positionNoiseStd);
    information(2, 2) = 1.0 / (orientationNoiseStd * orientationNoiseStd);

    loopClosureConstraint.information = information;
    constraints.push_back(loopClosureConstraint);
    return constraints;
}
struct LSError {
    LSError(const Eigen::Vector3d t_ij, const Eigen::Matrix3d info) : t_ij_(t_ij), info_(info) {}
    template<typename T>
    bool operator () (const T* p_i, const T* p_j, T* residual) const {
        
        T dx = p_j[0] - p_i[0];
        T dy = p_j[1] - p_i[1];
        T dtheta =  p_j[2] - p_i[2];
        NormalizeAngle(dtheta);

        // T residual[3];
        residual[0] = dx - T(t_ij_[0]);
        residual[1] = dy - T(t_ij_[1]);
        residual[2] = dtheta - T(t_ij_[2]);
        return true;
    }
    static ceres::CostFunction* create(const Eigen::Vector3d& t, const Eigen::Matrix3d& info) {
        ceres::CostFunction* fn = new ceres::AutoDiffCostFunction<LSError, 3, 3, 3> (
            new LSError(t, info)
        );
        return fn;
    }
private:
    Eigen::Vector3d t_ij_; // Relative transformation [dx, dy, dtheta] between from node i to node j
    Eigen::Matrix3d info_; 
};

// Helper function to write poses to a CSV file
void WritePosesToCSV(const std::string& filename, const std::vector<Pose2D>& poses) {
    std::ofstream file(filename);
    file << "id,x,y,theta\n";  // Header
    for (const auto& pose : poses) {
        file << pose.id << "," << pose.x << "," << pose.y << "," << pose.theta << "\n";
    }
    file.close();
}

int main() {
    vector<Pose2D> poses_gt = generateGTPoses();
    vector<Constraint2D> pose_odom = generateOdom(poses_gt);
    vector<Constraint2D> loop_closure = generateLoopClosure(poses_gt);
    pose_odom.insert(pose_odom.end(), loop_closure.begin(), loop_closure.end());


    std::vector<Pose2D> initial_estimates;
    Pose2D current_pose = poses_gt.front();  // Start from the first pose
    initial_estimates.push_back(current_pose);

    for (size_t i = 0; i < pose_odom.size(); ++i) {  // Exclude loop closure constraint
        const Constraint2D& constraint = pose_odom[i];
        // Update the pose using the measurement
        current_pose.x += constraint.t(0);
        current_pose.y += constraint.t(1);
        current_pose.theta += constraint.t(2);

        // Normalize angle to [-pi, pi]
        current_pose.theta = atan2(sin(current_pose.theta), cos(current_pose.theta));

        current_pose.id = constraint.id_end;
        initial_estimates.push_back(current_pose);
    }
    int N = initial_estimates.size();

    //set up ceres
    ceres::Problem problem;
    unordered_map<int, double*> pose_params; //k:id, v:params
    vector<array<double, 3>> params_storage(N); // Store parameters

    for (int i = 0; i < N; i++) {
        params_storage[i] = {initial_estimates[i].x, initial_estimates[i].y, initial_estimates[i].theta};
        pose_params[initial_estimates[i].id] = params_storage[i].data();

        // double params[3];
        // params[0] = initial_estimates[i].x;
        // params[1] = initial_estimates[i].y;
        // params[2] = initial_estimates[i].theta;
        // pose_params[initial_estimates[i].id] = params;
        if (i == 0) { //fix the first pose, otherwise it will float...
            params_storage[i] = {initial_estimates[i].x, initial_estimates[i].y, initial_estimates[i].theta};
            pose_params[initial_estimates[i].id] = params_storage[i].data();
        }

    }
    //add parameters
    for (const auto& odom : pose_odom) {
        ceres::CostFunction* fn = LSError::create(odom.t, odom.information);
        double* pose_start = pose_params[odom.id_begin];
        double* pose_end = pose_params[odom.id_end];
        problem.AddResidualBlock(fn, NULL, pose_start, pose_end);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;

    // Write ground truth poses
    WritePosesToCSV("ground_truth_poses.csv", poses_gt);

    // Write initial estimates
    WritePosesToCSV("initial_estimates.csv", initial_estimates);

    // Write optimized poses
    std::vector<Pose2D> optimized_poses;
    for (int i = 0; i < N; ++i) {
        Pose2D pose;
        pose.id = i;
        pose.x = params_storage[i][0];
        pose.y = params_storage[i][1];
        pose.theta = params_storage[i][2];
        optimized_poses.push_back(pose);
    }
    WritePosesToCSV("optimized_poses.csv", optimized_poses);



    return 0;
}