/*
g++ -std=c++17 -o feature_matching_cv_desc02 ./feature_matching_cv_desc02.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>


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


int main() {
    string image_path1 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000000.png";
    string image_path2 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000001.png";
    cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_GRAYSCALE);

    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    detector->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
    detector->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    matcher.match(desc1, desc2, matches);

    int n = 20;
    auto cmp = [](cv::DMatch& a, cv::DMatch& b) {
        return a.distance < b.distance; //max-heap, max on top
    };
    priority_queue<cv::DMatch, vector<cv::DMatch>, decltype(cmp)> pq(cmp);
    for (const auto& m : matches) {
        pq.push(m);
        if (pq.size() > n) {
            pq.pop();
        }
    }
    vector<cv::DMatch> good_matches;
    while (!pq.empty()) {
        good_matches.push_back(pq.top());
        pq.pop();
    }

    vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good_matches) {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }

    cout << pts1.size() << " " << pts2.size() << endl;
    
    //compute E
    cv::Mat K = (cv::Mat_<double>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02, 
                                   0.000000e+00, 9.808141e+02, 2.331966e+02, 
                                   0.000000e+00, 0.000000e+00, 1.000000e+00);
    
    cv::Mat mask; //inlier mask after RANSAC
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.9999, 0.1, mask);
    vector<cv::Point2f> pts1_inlier, pts2_inlier;
    vector<bool> mask_idx;
    for (int i = 0; i < mask.rows; i++) { //mask is mx1 matrix (bool)
        if (mask.at<bool>(i, 0) == 1) {
            pts1_inlier.push_back(pts1[i]);
            pts2_inlier.push_back(pts2[i]);
            mask_idx.push_back(true);
        }
        else {
            mask_idx.push_back(false);
        }
    }
    cout << "# of inliers = " <<pts1_inlier.size() << "/" << pts1.size() << endl;

    drawInOutliers(img1, img2, pts1, pts2, mask_idx);
    //Recover relative R, t
    cv::Mat R_esti, t_esti;
    int inliers = cv::recoverPose(E, pts1_inlier, pts2_inlier, K, R_esti, t_esti);
    cout << "R_esti = " << R_esti << " t_esti = " << t_esti << endl;
    return 0;
}
