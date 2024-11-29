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
