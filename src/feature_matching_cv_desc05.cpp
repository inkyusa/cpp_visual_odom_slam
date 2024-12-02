/*
g++ -std=c++17 -o feature_matching_cv_desc05 ./feature_matching_cv_desc05.cpp \
-I /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/include/opencv4 \
-L /Users/inkyusa/opt/miniconda3/envs/conda_pytorch_edu_p37/lib \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <queue>

using namespace std;
using namespace cv;

int main() {
    string image_path1 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000000.png";
    string image_path2 = "../dataset/2011_09_26-1/2011_09_26_drive_0001_sync/image_00/data/0000000001.png";
    Mat img1 = imread(image_path1, IMREAD_GRAYSCALE);
    Mat img2 = imread(image_path2, IMREAD_GRAYSCALE);
    Mat K = (Mat_<double>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02, 
                                   0.000000e+00, 9.808141e+02, 2.331966e+02, 
                                   0.000000e+00, 0.000000e+00, 1.000000e+00);


    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<Feature2D> detector = ORB::create(5000);
    detector->detectAndCompute(img1, noArray(), kpts1, desc1); //no mask
    detector->detectAndCompute(img2, noArray(), kpts2, desc2);

    //Matching
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING, true); //hamming distance for binary desc, cross matching check on
    matcher.match(desc1, desc2, matches);
    vector<Point2d> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }
    Mat mask;
    Mat R, t;
    Mat E = findEssentialMat(pts1, pts2, K, RANSAC, 0.999, 0.5, mask); //0.5 pixel threhold, 0.999 confidence
    vector<Point2d> pts1_inliers, pts2_inliers;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<bool>(i, 0)) {
            pts1_inliers.push_back(pts1[i]);
            pts2_inliers.push_back(pts2[i]);
        }
    }
    recoverPose(E, pts1_inliers, pts2_inliers, K, R, t);
    cout <<"R = " << R << endl;
    cout <<"t = " << t << endl;
    Mat T = Mat::zeros(4, 4, CV_64F);
    cout << T.type() << t.type() << endl;
    R.copyTo(T.rowRange(0, 3).colRange(0, 3));
    t.copyTo(T.rowRange(0, 3).col(3));
    T.at<double>(3, 3) = (double)(1);
    cout << "T = " << T << endl;
    return 0;
}
