#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    cv::VideoCapture cap("video.mp4");
    if (!cap.isOpened()) return -1;

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Camera intrinsic parameters
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                            0, fy, cy,
                                            0,  0,  1);

    cv::Mat R_f = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_f = cv::Mat::zeros(3, 1, CV_64F);

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Feature detection using FAST
        std::vector<cv::Point2f> pts1, pts2;
        cv::goodFeaturesToTrack(prevGray, pts1, 2000, 0.01, 10);

        // Feature tracking using Optical Flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prevGray, gray, pts1, pts2, status, err);

        // Remove points for which tracking failed
        std::vector<cv::Point2f> good_pts1, good_pts2;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                good_pts1.push_back(pts1[i]);
                good_pts2.push_back(pts2[i]);
            }
        }

        // Compute the essential matrix
        cv::Mat E, mask;
        E = cv::findEssentialMat(good_pts1, good_pts2, K, cv::RANSAC, 0.999, 1.0, mask);

        // Recover the pose from the essential matrix
        cv::Mat R, t;
        cv::recoverPose(E, good_pts1, good_pts2, K, R, t, mask);

        // Update the total rotation and translation
        t_f += R_f * t;
        R_f = R * R_f;

        prevGray = gray.clone();

        // Visualize the trajectory
        // ... (Implement visualization if needed)

        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}