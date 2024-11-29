/*
g++ -std=c++17 -o vo_cv.cpp ./vo_cv.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <fstream>

// using namespace std;
// using namespace cv;

using namespace std;

// Function to draw matches with inliers and outliers differentiated by color
void drawInOutliers(
    cv::Mat& img1_in, cv::Mat& img2_in,
    vector<cv::Point2f>& pts1, vector<cv::Point2f>& pts2,
    vector<bool>& inliers) {
    // Convert images to RGB if they are grayscale
    cv::Mat img1, img2;
    if (img1_in.channels() == 1) {
        cv::cvtColor(img1_in, img1, cv::COLOR_GRAY2BGR);
    } else {
        img1 = img1_in;
    }

    if (img2_in.channels() == 1) {
        cv::cvtColor(img2_in, img2, cv::COLOR_GRAY2BGR);
    } else {
        img2 = img2_in;
    }

    // Combine the two images for visualization
    cv::Size size(img1.cols + img2.cols, max(img1.rows, img2.rows));
    cv::Mat output = cv::Mat::zeros(size, img1.type());
    img1.copyTo(output(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(output(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Point2f pt1 = pts1[i];
        cv::Point2f pt2 = pts2[i];
        pt2.x += img1.cols; // Shift x-coordinate for img2

        // Choose color based on whether the match is an inlier or outlier
        cv::Scalar color = inliers[i] ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255); // Blue for inliers, Red for outliers

        // Draw a line connecting the matched points
        cv::line(output, pt1, pt2, color, 1);

        // Draw circles around the matched points
        cv::circle(output, pt1, 2, color, cv::FILLED);
        cv::circle(output, pt2, 2, color, cv::FILLED);
    }
    cv::Mat resized_output;
    cv::resize(output, resized_output, cv::Size(output.cols / 2, output.rows / 2), 0, 0, cv::INTER_LINEAR);
    cv::imshow("Feature Matches", resized_output);
    cv::waitKey(0);
}

// Function to convert rotation and translation to homogeneous transformation matrix
cv::Mat convertToHomo(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = cv::Mat::eye(4, 4, R.type()); // Initialize as identity matrix
    // Copy rotation matrix R to the top-left 3x3 block
    R.copyTo(T.rowRange(0, 3).colRange(0, 3));

    // Copy translation vector t to the top-right 3x1 block
    t.copyTo(T.rowRange(0, 3).col(3));
    return T;
}

int main() {
    string folderPath = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/";
    vector<string> imgPaths;
    cv::glob(folderPath + "*", imgPaths, false);
    int n = imgPaths.size();
    bool init = false;
    cv::Mat img1, img2;
    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    cv::Mat desc1, desc2;
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    vector<cv::DMatch> matches;
    cv::Mat mask;
    cv::Mat R, t;
    R = cv::Mat::eye(3, 3, CV_32F);
    t = cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat K = (cv::Mat_<float>(3,3) << 9.842439e+02, 0.000000e+00, 6.900000e+02,
                                          0.000000e+00, 9.808141e+02, 2.331966e+02,
                                          0.000000e+00, 0.000000e+00, 1.000000e+0);
    vector<cv::Point2f> pts1, pts2;
    vector<cv::Mat> traj_list;
    cv::Mat cur_pose = cv::Mat::eye(4, 4, CV_32F);

    // Open a file to write the trajectory positions
    std::ofstream trajFile("trajectory.txt");

    for(int i = 0; i < n; i++) {
        if (!init) {
            init = true;
            img1 = cv::imread(imgPaths[i], cv::IMREAD_GRAYSCALE);
            detector->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
            traj_list.push_back(cur_pose);

            // Write the initial position (0,0,0) to the file
            trajFile << 0 << " " << 0 << " " << 0 << std::endl;
        }
        else {
            img2 = cv::imread(imgPaths[i], cv::IMREAD_GRAYSCALE);
            detector->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

            matcher.match(desc1, desc2, matches);

            // Extract matched keypoints
            pts1.clear();
            pts2.clear();
            for (const auto& match : matches) {
                pts1.push_back(kpts1[match.queryIdx].pt);
                pts2.push_back(kpts2[match.trainIdx].pt);
            }

            // Compute the Essential matrix
            cv::Mat E, mask;
            E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

            // Recover the pose
            cv::recoverPose(E, pts1, pts2, K, R, t, mask);

            R.convertTo(R, CV_32F);
            t.convertTo(t, CV_32F);

            // Convert R and t to homogeneous transformation matrix
            cv::Mat T = convertToHomo(R, t);

            // Update current pose
            cur_pose = cur_pose * T;

            // Extract translation components
            float x = cur_pose.at<float>(0, 3);
            float y = cur_pose.at<float>(1, 3);
            float z = cur_pose.at<float>(2, 3);

            // Write the position to the file
            trajFile << x << " " << y << " " << z << std::endl;

            traj_list.push_back(cur_pose); // Add updated pose

            // Prepare for next iteration
            img1 = img2.clone();
            kpts1 = kpts2;
            desc1 = desc2.clone();
        }
    }

    trajFile.close(); // Close the file
    return 0;
}