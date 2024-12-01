/*
g++ -std=c++17 -o feature_matching_cv_desc04_bucketing ./feature_matching_cv_desc04_bucketing.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;


// Function to draw matches with inliers and outliers differentiated by color
void drawInOutliers(
    Mat& img1_in, Mat& img2_in,
    vector<Point2f>& pts1, vector<Point2f>& pts2,
    vector<bool>& inliers) {
    // Convert images to RGB if they are grayscale
    Mat img1, img2;
    if (img1_in.channels() == 1) {
        cvtColor(img1_in, img1, COLOR_GRAY2BGR);
    } else {
        img1 = img1_in;
    }

    if (img2_in.channels() == 1) {
        cvtColor(img2_in, img2, COLOR_GRAY2BGR);
    } else {
        img2 = img2_in;
    }

    // Combine the two images for visualization
    Size size(img1.cols + img2.cols, max(img1.rows, img2.rows));
    Mat output = Mat::zeros(size, img1.type());
    img1.copyTo(output(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(output(Rect(img1.cols, 0, img2.cols, img2.rows)));
    for (size_t i = 0; i < pts1.size(); i++) {
        Point2f pt1 = pts1[i];
        Point2f pt2 = pts2[i];
        pt2.x += img1.cols; // Shift x-coordinate for img2

        // Choose color based on whether the match is an inlier or outlier
        Scalar color = inliers[i] ? Scalar(255, 0, 0) : Scalar(0, 0, 255); // Blue for inliers, Red for outliers

        // Draw a line connecting the matched points
        line(output, pt1, pt2, color, 1);

        // Draw circles around the matched points
        circle(output, pt1, 2, color, FILLED);
        circle(output, pt2, 2, color, FILLED);
    }
    Mat resized_output;
    resize(output, resized_output, Size(output.cols / 2, output.rows / 2), 0, 0, INTER_LINEAR);
    imshow("Feature Matches", resized_output);
    waitKey(0);
}


// Function to perform feature bucketing
void bucketFeatures(
    const vector<KeyPoint>& keypoints,
    int image_width, int image_height,
    int bucket_size, int max_features_per_bucket,
    vector<KeyPoint>& bucketed_keypoints) {

    // Calculate the number of buckets in x and y directions
    int num_buckets_x = (image_width + bucket_size - 1) / bucket_size;
    int num_buckets_y = (image_height + bucket_size - 1) / bucket_size;

    // Create a 2D vector of buckets
    vector<vector<vector<KeyPoint>>> buckets(
        num_buckets_y, vector<vector<KeyPoint>>(num_buckets_x));

    // Assign keypoints to buckets
    for (const auto& kp : keypoints) {
        int x_idx = static_cast<int>(kp.pt.x) / bucket_size;
        int y_idx = static_cast<int>(kp.pt.y) / bucket_size;
        buckets[y_idx][x_idx].push_back(kp);
    }

    // Collect top features from each bucket
    for (int y = 0; y < num_buckets_y; ++y) {
        for (int x = 0; x < num_buckets_x; ++x) {
            auto& bucket = buckets[y][x];
            // Sort keypoints in the bucket by response
            sort(bucket.begin(), bucket.end(),
                      [](const KeyPoint& a, const KeyPoint& b) {
                          return a.response > b.response;
                      });
            // Select top N features from the bucket
            int features_to_copy = min(max_features_per_bucket, static_cast<int>(bucket.size()));
            bucketed_keypoints.insert(bucketed_keypoints.end(), bucket.begin(), bucket.begin() + features_to_copy);
        }
    }
}

using namespace cv;
int main() {
    string image_path1 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000000.png";
    string image_path2 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000001.png";

    Mat img1 = imread(image_path1, IMREAD_GRAYSCALE);
    Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);

    // ORB Detector Parameters
    Ptr<ORB> detector = ORB::create(5000); // Increase to detect more features initially
    vector<KeyPoint> kpts1_all, kpts2_all;
    Mat desc1_all, desc2_all;

    // Detect keypoints and compute descriptors
    detector->detectAndCompute(img1, noArray(), kpts1_all, desc1_all);
    detector->detectAndCompute(img2, noArray(), kpts2_all, desc2_all);

    // Parameters for feature bucketing
    int bucket_size = 50; // Size of each bucket in pixels
    int max_features_per_bucket = 5; // Max features to retain per bucket

    // Perform feature bucketing
    vector<KeyPoint> kpts1_bucketed, kpts2_bucketed;
    bucketFeatures(kpts1_all, img1.cols, img1.rows, bucket_size, max_features_per_bucket, kpts1_bucketed);
    bucketFeatures(kpts2_all, img2.cols, img2.rows, bucket_size, max_features_per_bucket, kpts2_bucketed);

    // Compute descriptors for bucketed keypoints
    Mat desc1, desc2;
    detector->compute(img1, kpts1_bucketed, desc1);
    detector->compute(img2, kpts2_bucketed, desc2);



    BFMatcher matcher(NORM_HAMMING, true); //binary desc, cross-matching on 
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    int topK = 50; //maintain min-distance, 50
    auto cmp = [](DMatch& a, DMatch& b) {
        return a.distance < b.distance; //max-heap, max on top
    };
    priority_queue<DMatch, vector<DMatch>, decltype(cmp)> pq(cmp);
    for (const auto& match : matches) {
        pq.push(match);
        if (pq.size() > topK) {
            pq.pop();
        }
    }
    vector<DMatch> good_matches;
    while (!pq.empty()) {
        good_matches.push_back(pq.top());
        pq.pop();
    }
    vector<Point2f> pts1, pts2;
    for (const auto& match : good_matches) {
        pts1.push_back(kpts1_bucketed[match.queryIdx].pt);
        pts2.push_back(kpts2_bucketed[match.trainIdx].pt);
    }//found 50 correspondences

    Mat K = (Mat_<float>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02, 
                                   0.000000e+00, 9.808141e+02, 2.331966e+02, 
                                   0.000000e+00, 0.000000e+00, 1.000000e+00);
    //find esseitla matrix
    //Mat mask
    Mat mask;
    Mat E = findEssentialMat(pts1, pts2, K, RANSAC, 0.999, 0.1, mask); // using RANSAC with 0.999 confidence, and 1 pixel threhold
    vector<bool> mask_bool;
    for (int i = 0; i < mask.rows; i++) {
        mask_bool.push_back(mask.at<bool>(i, 0));
    }
    Mat R_esti, t_esti;
    recoverPose(E, pts1, pts2, K, R_esti, t_esti);
    cout << "R_esti=" << R_esti << endl;
    cout << "t_esti=" << t_esti << endl;

    drawInOutliers(img1, img2, pts1, pts2, mask_bool);
    return 0;
}
