/*
g++ -std=c++17 -o vo_others_gpt_converted ./vo_others_gpt_converted.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-I /usr/local/include/eigen3 \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/



#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>

class VisualOdometry {
public:
    VisualOdometry(const std::string& data_dir);
    void run();

private:
    cv::Mat K_;  // Intrinsic camera matrix
    cv::Mat P_;  // Projection matrix
    std::vector<cv::Mat> gt_poses_;  // Ground truth poses
    std::vector<cv::Mat> images_;    // Grayscale images
    cv::Ptr<cv::ORB> orb_;           // ORB detector
    cv::FlannBasedMatcher flann_;    // FLANN matcher

    void loadCalibration(const std::string& filepath);
    void loadPoses(const std::string& filepath);
    void loadImages(const std::string& filepath);
    cv::Mat formTransformation(const cv::Mat& R, const cv::Mat& t);
    void getMatches(int idx, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2);
    cv::Mat getPose(const std::vector<cv::Point2f>& q1, const std::vector<cv::Point2f>& q2);
};

VisualOdometry::VisualOdometry(const std::string& data_dir) {
    loadCalibration(data_dir + "/calib.txt");
    loadPoses(data_dir + "/poses.txt");
    loadImages(data_dir + "/image_l");

    orb_ = cv::ORB::create(3000);

    // Set up FLANN matcher with LSH index
    cv::Ptr<cv::flann::IndexParams> index_params = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
    cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>(50);
    flann_ = cv::FlannBasedMatcher(index_params, search_params);
}

void VisualOdometry::loadCalibration(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open calibration file: " << filepath << std::endl;
        return;
    }
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::vector<double> params((std::istream_iterator<double>(iss)), std::istream_iterator<double>());
    P_ = cv::Mat(params).reshape(1, 3);
    K_ = P_(cv::Rect(0, 0, 3, 3)).clone();
    file.close();
}

void VisualOdometry::loadPoses(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open poses file: " << filepath << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> T_values((std::istream_iterator<double>(iss)), std::istream_iterator<double>());
        cv::Mat T = cv::Mat(T_values).reshape(1, 3);
        cv::Mat last_row = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
        cv::vconcat(T, last_row, T);
        gt_poses_.push_back(T);
    }
    file.close();
}

void VisualOdometry::loadImages(const std::string& filepath) {
    std::vector<std::string> image_paths;
    cv::glob(filepath + "/*.png", image_paths, false);
    if (image_paths.empty()) {
        std::cerr << "No images found in: " << filepath << std::endl;
        return;
    }
    // Load images as grayscale
    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }
        images_.push_back(img);
    }
}

cv::Mat VisualOdometry::formTransformation(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t.copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

void VisualOdometry::getMatches(int idx, std::vector<cv::Point2f>& q1, std::vector<cv::Point2f>& q2) {
    // Detect and compute keypoints and descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb_->detectAndCompute(images_[idx - 1], cv::noArray(), kp1, des1);
    orb_->detectAndCompute(images_[idx], cv::noArray(), kp2, des2);

    if (des1.empty() || des2.empty()) {
        std::cerr << "Descriptors are empty at frame " << idx << std::endl;
        return;
    }

    // Match descriptors using FLANN
    std::vector<std::vector<cv::DMatch>> matches;
    flann_.knnMatch(des1, des2, matches, 2);

    // Ratio test to keep good matches
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (matches[i][0].distance < 0.8f * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }

    // Extract matched keypoints positions
    for (size_t i = 0; i < good_matches.size(); ++i) {
        q1.push_back(kp1[good_matches[i].queryIdx].pt);
        q2.push_back(kp2[good_matches[i].trainIdx].pt);
    }

    // Optional: Draw matches
    cv::Mat img_matches;
    cv::drawMatches(images_[idx - 1], kp1, images_[idx], kp2, good_matches, img_matches);
    cv::imshow("Matches", img_matches);
    cv::waitKey(1);  // Adjust delay as needed
}

cv::Mat VisualOdometry::getPose(const std::vector<cv::Point2f>& q1, const std::vector<cv::Point2f>& q2) {
    // Compute the Essential matrix
    cv::Mat E, mask;
    E = cv::findEssentialMat(q1, q2, K_, cv::RANSAC, 0.999, 1.0, mask);

    // Recover pose from Essential matrix
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, q1, q2, K_, R, t, mask);

    // Form the transformation matrix
    cv::Mat transformation = formTransformation(R, t);
    return transformation;
}

void VisualOdometry::run() {
    std::ofstream traj_file("trajectory_.txt");
    std::vector<cv::Point2f> gt_path, estimated_path;
    cv::Mat cur_pose = cv::Mat::eye(4, 4, CV_64F);
    std::cout << K_ << std::endl;

    for (size_t i = 0; i < images_.size(); ++i) {
        if (i == 0) {
            cur_pose = gt_poses_[i];
            traj_file << 0 << " " << 0 << " " << 0 << std::endl; 
        } else {
            std::vector<cv::Point2f> q1, q2;
            getMatches(i, q1, q2);
            if (q1.size() < 5 || q2.size() < 5) {
                std::cerr << "Not enough matches between frames " << i - 1 << " and " << i << std::endl;
                continue;
            }
            cv::Mat transformation = getPose(q1, q2);
            std::cout << "transformation" << transformation << std::endl;
            cur_pose = cur_pose * transformation.inv();
        }
        // Extract positions
        gt_path.push_back(cv::Point2f(gt_poses_[i].at<double>(0, 3), gt_poses_[i].at<double>(2, 3)));
        estimated_path.push_back(cv::Point2f(cur_pose.at<double>(0, 3), cur_pose.at<double>(2, 3)));
        traj_file << cur_pose.at<double>(0, 3) << " " << cur_pose.at<double>(1, 3) << " " << cur_pose.at<double>(2, 3) << std::endl;
        // trajFile << x << " " << y << " " << z << std::endl;
    }

    // Visualization
    int path_size = static_cast<int>(gt_path.size());
    cv::Mat traj = cv::Mat::zeros(600, 600, CV_8UC3);

    for (int i = 0; i < path_size; ++i) {
        int x_gt = static_cast<int>(gt_path[i].x) + 300;
        int y_gt = static_cast<int>(gt_path[i].y) + 100;
        int x_est = static_cast<int>(estimated_path[i].x) + 300;
        int y_est = static_cast<int>(estimated_path[i].y) + 100;

        cv::circle(traj, cv::Point(x_gt, y_gt), 1, CV_RGB(0, 255, 0), 2);
        cv::circle(traj, cv::Point(x_est, y_est), 1, CV_RGB(255, 0, 0), 2);
        cv::imshow("Trajectory", traj);
        cv::waitKey(1);
    }
    cv::waitKey(0);
}

int main() {
    std::string data_dir = "../dataset/KITTI_sequence_2";  // Update the path as needed
    VisualOdometry vo(data_dir);
    vo.run();
    return 0;
}
