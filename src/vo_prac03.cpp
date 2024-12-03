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
    cv::FlannBasedMatcher flann_;
    // Set up FLANN matcher with LSH index
    cv::Ptr<cv::flann::IndexParams> index_params = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
    cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>(50);
    flann_ = cv::FlannBasedMatcher(index_params, search_params);

    // BFMatcher matcher(NORM_HAMMING, true); //hamming distance for binary desc, cross matching check on
    int topK = 300; //maintain min-distance, 50
    vector<Mat> trajectory;
    trajectory.push_back(Mat::eye(4, 4, CV_64F)); //initial pose
    // Open a file to write the trajectory positions
    std::ofstream trajFile("trajectory.txt");
    vector<Point3d> prev_point3D;
    for (int frame_cnt = 0; frame_cnt < n; frame_cnt++) {
        if (!init) {
            init = true;
            img1 = imread(imgPaths[frame_cnt], IMREAD_GRAYSCALE);
            detector->detectAndCompute(img1, noArray(), kpts1, desc1); //no mask
            // Write the initial position (0,0,0) to the file
            trajFile << 0 << " " << 0 << " " << 0 << std::endl;
        }
        else {
            img2 = imread(imgPaths[frame_cnt], IMREAD_GRAYSCALE);
            detector->detectAndCompute(img2, noArray(), kpts2, desc2); //no mask
            // //Matching

            // FLANN parameters for LSH (Locality Sensitive Hashing)
            FlannBasedMatcher matcher(makePtr<flann::LshIndexParams>(12, 20, 2));

            // Perform knnMatch
            vector<vector<DMatch>> knnMatches;
            flann_.knnMatch(desc1, desc2, knnMatches, 2); // k=2 for the two nearest neighbors

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
            cout << "11111" << endl;
            vector<Point2d> pts1, pts2;
            for (const auto& m : goodMatches) {
                pts1.push_back(kpts1[m.queryIdx].pt);
                pts2.push_back(kpts2[m.trainIdx].pt);
            }
            cout << "2222" << endl;
            Mat mask;
            Mat R, t;
            Mat E = findEssentialMat(pts1, pts2, K, RANSAC, 0.99, 1, mask); //0.5 pixel threhold, 0.999 confidence
            vector<Point2d> pts1_inliers, pts2_inliers;
            for (int row = 0; row < mask.rows; row++) {
                if (mask.at<bool>(row, 0)) {
                    pts1_inliers.push_back(pts1[row]);
                    pts2_inliers.push_back(pts2[row]);
                }
            }
            cout << "3333" << endl;
            recoverPose(E, pts1, pts2, K, R, t, mask);

            cout << "44444" << endl;
            // //performing triangulation
            // Mat P1 = K * Mat::eye(3, 4, CV_64F);
            // Mat P2 = K * Mat::eye(3, 4, CV_64F);
            // R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
            // t.copyTo(P2.rowRange(0, 3).col(3));
            // Mat point3D_homo;
            // triangulatePoints(P1, P2, pts1_inliers, pts2_inliers, point3D_homo);
            // vector<Point3d> point3D;
            // for (int col = 0; col < point3D_homo.cols; col++) {
            //     Point3d pt;
            //     Mat x = point3D_homo.col(col);
            //     if (x.at<double>(3, 0) != 0) {
            //         x /= x.at<double>(3, 0);
            //         pt.x = x.at<double>(0,0);
            //         pt.y = x.at<double>(1, 0);
            //         pt.z = x.at<double>(2, 0);
            //         point3D.push_back(pt);
            //     }
            // }
            // //estimate scale
            // double scale = 0.0;
            // int cnt = 0;
            // if (!prev_point3D.empty()) {
            //     for (int i = 0; i < point3D.size(); i++) {
            //         if (mask.at<uchar>(i)) {
            //             double dist_curr = norm(point3D[i]);
            //             double dist_prev = norm(prev_point3D[i]);
            //             if (dist_curr > 0 and dist_prev> 0) {
            //                 scale += dist_curr / dist_prev;
            //                 cnt++;
            //             }
            //         }
            //     }
            //     if (cnt > 0) {
            //         scale = scale / cnt;
            //     }
            //     else {
            //         scale = 1.0;
            //     }
            // }
            // else {
            //     scale = 1.0;
            // }
            // cout <<"Scale = " << scale << "ith = " << frame_cnt << endl;
            // prev_point3D = point3D;
            // t = t * scale;

            // Corrected Projection Matrices
            Mat P1 = K * Mat::eye(3, 4, CV_64F);
            Mat P2 = K * (Mat_<double>(3, 4) <<
                R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
            // cout << "55555" << endl;
            // Triangulate Points
            Mat points4D;
            triangulatePoints(P1, P2, pts1_inliers, pts2_inliers, points4D);
            // cout << "66666" << endl;
            // Convert to Euclidean Coordinates
            vector<Point3d> point3D;
            for (int i = 0; i < points4D.cols; i++) {
                Mat x = points4D.col(i);
                x /= x.at<double>(3, 0); // Normalize
                Point3d pt(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0));
                point3D.push_back(pt);
            }
            // cout << "7777" << endl;
            // Estimate Scale
            double scale = 0.0;
            int cnt = 0;
            if (!prev_point3D.empty()) {
                // cout << "point3D.size() = " << point3D.size() << endl;
                // cout << "prev_point3D.size() = " << prev_point3D.size() << endl;
                for (size_t i = 0; i < point3D.size(); i++) {
                    if (mask.at<uchar>(i)) {
                        // cout << "8888" << endl;
                        double dist_curr = norm(point3D[i]);
                        // cout << "10101010" << endl;
                        // cout <<"prev_point3D.size() = " << prev_point3D.size() << endl;
                        // cout <<"i = " << i << endl;
                        double dist_prev = norm(prev_point3D[i]);
                        // cout << "12121212" << endl;
                        if (dist_curr > 0 && dist_prev > 0) {
                            // cout << "13131313" << endl;
                            scale += dist_prev / dist_curr;
                            // cout << "141414141" << endl;
                            cnt++;
                        }
                    }
                }
                // cout << "9999" << endl;
                if (cnt > 0) {
                    scale = scale / cnt;
                } else {
                    scale = 1.0;
                }
            } else {
                scale = 1.0;
            }
            // cout << "Scale = " << scale << " ith = " << frame_cnt << endl;
            prev_point3D = point3D;

            // Scale the translation vector
            // t = t


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
