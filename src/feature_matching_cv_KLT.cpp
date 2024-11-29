/*
g++ -std=c++17 -o feature_matching_cv_KLT ./feature_matching_cv_KLT.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann -lopencv_video
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <queue>

using namespace std;
using namespace cv;

vector<DMatch> findCorrespondences(const string &image_path1, const string &image_path2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2) {
    // Read the two images
    Mat img1 = imread(image_path1, IMREAD_GRAYSCALE);
    Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        cerr << "Error: Could not open or find one of the images!" << endl;
        return {};
    }

    // Step 1: Detect keypoints and compute descriptors
    Ptr<Feature2D> detector = ORB::create(); // ORB is used here for feature detection

    // vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // Step 2: Match descriptors using BFMatcher
    BFMatcher matcher(NORM_HAMMING, true); // NORM_HAMMING is appropriate for ORB descriptors
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Step 3: Filter good matches using distance threshold
    double max_dist = 0, min_dist = 100;

    // Find min and max distances between keypoints
    for (const auto &match : matches) {
        double dist = match.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    vector<DMatch> good_matches;
    for (const auto &match : matches) {
        if (match.distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(match);
        }
    }
    #if 0
    // Step 4: Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

    // Display matches
    imshow("Feature Matches", img_matches);
    waitKey(0);

    // Save matches to a file (optional)
    imwrite("matches.jpg", img_matches);
    #endif

    // Print summary
    cout << "Total matches found: " << matches.size() << endl;
    cout << "Good matches after filtering: " << good_matches.size() << endl;
    return good_matches;
}

int main() {
    string image_path1 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000000.png";
    string image_path2 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000001.png";
    Mat img1 = imread(image_path1, IMREAD_GRAYSCALE);
    Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);

    vector<Point2f> temp_pts1, temp_pts2;
    goodFeaturesToTrack(img1, temp_pts1, 2000, 0.01, 10);
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(img1, img2, temp_pts1, temp_pts2, status, err);
    // Remove points for which tracking failed
    vector<Point2f> pts1, pts2;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            pts1.push_back(temp_pts1[i]);
            pts2.push_back(temp_pts2[i]);
        }
    }

    cout <<"pts1.size() = " << pts1.size() << endl;
    cout <<"pts2.size() = " << pts2.size() << endl;

    // vector<DMatch> matches = findCorrespondences(image_path1, image_path2, kps1, kps2);
    // cout << "kps1.size() = " << kps1.size() << ", kps2.size() = " << kps2.size() << endl;

    return 0;
}
