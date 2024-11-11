//
// Created by ysc on 2023/2/28.
//

#ifndef NATIVE_OPENCV_ANDROID_TEMPLATE_MASTER_HIGHLIGHTREMOVAL_H
#define NATIVE_OPENCV_ANDROID_TEMPLATE_MASTER_HIGHLIGHTREMOVAL_H
#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace chrono;

class highlightRemoval
{
public:
    Mat brightImage;
    Mat stitchedImage;
    highlightRemoval();
    ~highlightRemoval();
    bool findBrightImage(Mat &frame);
    bool findMatches(Mat &frame);
private:



    Rect findHighLight(Mat image, int lower_threshold=180);
    Rect findBlackBox(Mat image);
    void highlightImageStitching(Mat image1, Mat image2);
    bool highlightImageInnerStitching(Mat image1, Mat image2, int setMatches=300);
    double Point2PointDist(const cv::Point2f& a, const cv::Point2f& b);

};
#endif //NATIVE_OPENCV_ANDROID_TEMPLATE_MASTER_HIGHLIGHTREMOVAL_H
