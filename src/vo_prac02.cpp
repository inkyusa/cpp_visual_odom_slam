/*
g++ -std=c++17 -o vo_prac02 ./vo_prac02.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <queue>

using namespace std;
using namespace cv;

int main() {
    string folderPath = "../dataset/KITTI_sequence_2/image_l/";
    vector<string> imgPaths;
    cv::glob(folderPath + "*", imgPaths, false);
    int n = imgPaths.size();

    Mat K = (Mat_<double>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02,
                                          0.000000e+00, 9.808141e+02, 2.331966e+02,
                                          0.000000e+00, 0.000000e+00, 1.000000e+0);
    cout << n << endl;
    bool init = false;
    Mat img1, img2;
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<Feature2D> detector = ORB::create(5000);
    BFMatcher matcher(NORM_HAMMING, true); //hamming distance for binary desc, cross matching check on
    int topK = 300; //maintain min-distance, 50
    vector<Mat> trajectory;
    trajectory.push_back(Mat::eye(4, 4, CV_64F)); //initial pose
    // Open a file to write the trajectory positions
    std::ofstream trajFile("trajectory.txt");
    for (int i = 0; i < n; i++) {
        if (!init) {
            init = true;
            img1 = imread(imgPaths[i], IMREAD_GRAYSCALE);
            detector->detectAndCompute(img1, noArray(), kpts1, desc1); //no mask
            // Write the initial position (0,0,0) to the file
            trajFile << 0 << " " << 0 << " " << 0 << std::endl;
        }
        else {
            img2 = imread(imgPaths[i], IMREAD_GRAYSCALE);
            detector->detectAndCompute(img2, noArray(), kpts2, desc2); //no mask
            // //Matching

            // FLANN parameters for LSH (Locality Sensitive Hashing)
            FlannBasedMatcher matcher(makePtr<flann::LshIndexParams>(12, 20, 2));

            // Perform knnMatch
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(desc1, desc2, knnMatches, 2); // k=2 for the two nearest neighbors

            // Apply Lowe's Ratio Test
            vector<DMatch> goodMatches;
            const float ratioThresh = 0.8f; // Lowe's ratio threshold
            for (size_t i = 0; i < knnMatches.size(); i++) {
                if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
                    goodMatches.push_back(knnMatches[i][0]);
                }
            }

            // Output the number of good matches
            cout << "Number of good matches: " << goodMatches.size() << endl;



            // vector<DMatch> matches;
            // matcher.match(desc1, desc2, matches);
            // auto cmp = [](DMatch& a, DMatch& b) {
            //     return a.distance < b.distance; //max-heap, max on top
            // };
            // priority_queue<DMatch, vector<DMatch>, decltype(cmp)> pq(cmp);
            // for (const auto& match : matches) {
            //     pq.push(match);
            //     if (pq.size() > topK) {
            //         pq.pop();
            //     }
            // }
            // vector<DMatch> good_matches;
            // while (!pq.empty()) {
            //     good_matches.push_back(pq.top());
            //     pq.pop();
            // }
            vector<Point2d> pts1, pts2;
            for (const auto& m : goodMatches) {
                pts1.push_back(kpts1[m.queryIdx].pt);
                pts2.push_back(kpts2[m.trainIdx].pt);
            }
            Mat mask;
            Mat R, t;
            Mat E = findEssentialMat(pts1, pts2, K, RANSAC, 0.99, 1, mask); //0.5 pixel threhold, 0.999 confidence
            // vector<Point2d> pts1_inliers, pts2_inliers;
            // for (int i = 0; i < mask.rows; i++) {
            //     if (mask.at<bool>(i, 0)) {
            //         pts1_inliers.push_back(pts1[i]);
            //         pts2_inliers.push_back(pts2[i]);
            //     }
            // }
            recoverPose(E, pts1, pts2, K, R, t, mask);
            cout <<"R = " << R << endl;
            cout <<"t = " << t << endl;
            Mat T = Mat::eye(4, 4, CV_64F);
            R.copyTo(T.rowRange(0, 3).colRange(0, 3));
            t.copyTo(T.rowRange(0, 3).col(3));
            // Write the position to the file
            // cout << "T = " << T << endl;
            Mat currPose = trajectory.back();
            currPose = currPose * T.inv();
            trajectory.push_back(currPose);
            double x = currPose.at<double>(0, 3);
            double y = currPose.at<double>(1, 3);
            double z = currPose.at<double>(2, 3);
            trajFile << x << " " << y << " " << z << std::endl;
            

            img1 = img2.clone();
            desc1 = desc2.clone();
            kpts1 = kpts2;
        }
    }
    trajFile.close(); // Close the file
    return 0;
}
