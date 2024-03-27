//
//  main.cpp
//  HighlightRemoval
//
//  Created by ysc on 2023/2/12.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "chrono"
using namespace std;
using namespace cv;
using namespace chrono;

Rect findHighLight(Mat image, int lower_threshold=180)
{
    Mat gray, imageBin;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, imageBin, lower_threshold, 255, THRESH_BINARY);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(imageBin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int maxArea = 0;
    Rect maxRect;
    for(int i=0;i<contours.size();i++)
    {
        Rect rect = boundingRect(contours[i]);
        if(maxArea<rect.area())
        {
            maxArea = rect.area();
            maxRect = rect;
        }
       
    }

    return maxRect;
}

Rect findBlackBox(Mat image)
{
    cvtColor(image, image, COLOR_BGR2GRAY);
    blur(image, image, Size(3,3));
    Mat edge;
    Canny(image, edge, 150, 200, 3);//150,200
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(edge, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Rect balckRect;
    int rectNum = 0;
    for(int i=0;i<contours.size();i++)
    {
        Rect rect = boundingRect(contours[i]);
        //面积上界要比较大
        if(rect.area()>4000&&rect.area()<8000&&abs(rect.width-rect.height)<5)
        {
            balckRect = rect;
            rectNum++;
        }
        
                   
    }
    cout<<"BoxNum:"<<rectNum<<"area"<<balckRect.area()<<endl;
    return balckRect;
}



void highlightImageStitching(Mat image1, Mat image2)
{
    auto start = system_clock::now();
    Ptr<ORB> detector1 = ORB::create(10000);
    Ptr<ORB> detector2 = ORB::create(10000);
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptor1;
    Mat descriptor2;
    detector1->detectAndCompute(image1, Mat(), keypoints1, descriptor1);
    detector2->detectAndCompute(image2, Mat(), keypoints2, descriptor2);
    if(descriptor1.type()!=CV_32F) {
        descriptor1.convertTo(descriptor1, CV_32F);
    }

    if(descriptor2.type()!=CV_32F) {
        descriptor2.convertTo(descriptor2, CV_32F);
    }
    
    
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    Mat matchImg;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, matchImg, Scalar(0, 255, 0), Scalar::all(-1));
    
    //筛选匹配，找出距离小于一半最大距离的match
    double sum = 0;
    double maxDist = 0;
    double minDist = 0;
    for (auto &match : matches)
    {
        double dist = match.distance;
        maxDist = max(maxDist, dist);
        minDist = min(minDist, dist);
    }
   
 
    std::vector<DMatch> goodMatches;
    double threshold = 0.5;
    for (auto &match : matches)
    {
        if (match.distance < threshold * maxDist)
            goodMatches.emplace_back(match);
    }

    
    
    //step 5.1 align feature points and convet to float
    std::vector<KeyPoint> R_keypoint01, R_keypoint02;
    for (auto &match : goodMatches)
    {
        R_keypoint01.emplace_back(keypoints1[match.queryIdx]);
        R_keypoint02.emplace_back(keypoints2[match.trainIdx]);
    }
    std::vector<Point2f> p01, p02;
    for (int i = 0; i < goodMatches.size(); ++i)
    {
        p01.emplace_back(R_keypoint01[i].pt);
        p02.emplace_back(R_keypoint02[i].pt);
    }
    cout<<"goodmatches:"<<goodMatches.size()<<endl;
    //计算单应性
    std::vector<uchar> RansacStatus;
    Mat fundamental = findHomography(p01, p02, RansacStatus, RANSAC);
    Mat image1Calib;//反射变换后的image1,和image2在一个平面上
    //透视变换,填充必须是白色,白色填充直接用image2覆盖
    warpPerspective(image1, image1Calib, fundamental, Size(image1.cols, image1.rows),INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
 
    void warpPerspective( InputArray src, OutputArray dst,
                                       InputArray M, Size dsize,
                                       int flags = INTER_LINEAR,
                                       int borderMode = BORDER_CONSTANT,
                                       const Scalar& borderValue = Scalar());
   
    //删除错误匹配特征点
    std::vector<KeyPoint> RR_keypoint01, RR_keypoint02;
    std::vector<DMatch> RR_matches;
    int idx = 0;
    for (int i = 0; i < goodMatches.size(); ++i)
    {
        if (RansacStatus[i] != 0)
        {
            RR_keypoint01.emplace_back(R_keypoint01[i]);
            RR_keypoint02.emplace_back(R_keypoint02[i]);
            goodMatches[i].queryIdx = idx;
            goodMatches[i].trainIdx = idx;
            RR_matches.emplace_back(goodMatches[i]);
            ++idx;
        }
    }
    
    Mat imgRRMatches;
    drawMatches(image1, RR_keypoint01, image2, RR_keypoint02, RR_matches, imgRRMatches, Scalar(0, 255, 0), Scalar::all(-1));
    imshow("final match", imgRRMatches);
    Mat image1Calib_copy = image1Calib.clone();
    Mat output = min(image1Calib_copy(Rect(0, 0, image2.cols, image2.rows)),image2);
    
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    //hours(小时), minutes（分钟）, seconds（秒）, milliseconds（毫秒）, nanoseconds（纳秒）
    auto cost = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    cout<<"time:"<<cost<<endl;
    imshow("stitching image", output);
    waitKey(0);
    
}

void reducePixelValue(Mat& image, int threshold, float factor)
{

    for (int row = 0; row < image.rows; row++)
    {
        for (int col = 0; col < image.cols; col++)
        {
            Vec3b& pixel = image.at<Vec3b>(row, col);
            for (int channel = 0; channel < 3; channel++)
            {
                if (pixel[channel] > threshold)
                {
                    pixel[channel] = static_cast<uchar>(pixel[channel] * factor);
                }
            }
        }
    }
}

Mat sharpenImage(const Mat& inputImage) {
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    Mat laplacianImage;
    Laplacian(grayImage, laplacianImage, CV_32F, 3, 1, 0, BORDER_DEFAULT);
    Mat normalizedLaplacianImage;
    normalize(laplacianImage, normalizedLaplacianImage, 0, 255, NORM_MINMAX, CV_8U);
    Mat sharpenedImage;
    addWeighted(grayImage, 1.5, normalizedLaplacianImage, -0.5, 0, sharpenedImage);
    return sharpenedImage;
}


void highlightImageInnerStitching(Mat image1, Mat image2, int setMatches=300)
{
    auto start = system_clock::now();
    Ptr<ORB> detector1 = ORB::create(10000);
    Ptr<ORB> detector2 = ORB::create(10000);
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptor1;
    Mat descriptor2;
    detector1->detectAndCompute(image1, Mat(), keypoints1, descriptor1);
    detector2->detectAndCompute(image2, Mat(), keypoints2, descriptor2);
    if(descriptor1.type()!=CV_32F) {
        descriptor1.convertTo(descriptor1, CV_32F);
    }

    if(descriptor2.type()!=CV_32F) {
        descriptor2.convertTo(descriptor2, CV_32F);
    }
    
    
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    Mat matchImg;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, matchImg, Scalar(0, 255, 0), Scalar::all(-1));
    
    //筛选匹配，找出距离小于一半最大距离的match
    double sum = 0;
    double maxDist = 0;
    double minDist = 0;
    for (auto &match : matches)
    {
        double dist = match.distance;
        maxDist = max(maxDist, dist);
        minDist = min(minDist, dist);
    }
   
 
    std::vector<DMatch> goodMatches;
    double threshold = 0.5;
    for (auto &match : matches)
    {
        if (match.distance < threshold * maxDist)
            goodMatches.emplace_back(match);
    }

    
    
    //将matches转换float
    std::vector<KeyPoint> R_keypoint01, R_keypoint02;
    for (auto &match : goodMatches)
    {
        R_keypoint01.emplace_back(keypoints1[match.queryIdx]);
        R_keypoint02.emplace_back(keypoints2[match.trainIdx]);
    }
    std::vector<Point2f> p01, p02;
    for (int i = 0; i < goodMatches.size(); ++i)
    {
        p01.emplace_back(R_keypoint01[i].pt);
        p02.emplace_back(R_keypoint02[i].pt);
    }
    cout<<"goodmatches:"<<goodMatches.size()<<endl;
    if(goodMatches.size()<setMatches)
    {
        cout<<"few matches"<<endl;
        return;
    }
    
    //计算单应性
    std::vector<uchar> RansacStatus;
    Mat fundamental = findHomography(p01, p02, RansacStatus, RANSAC);
    Mat image1Calib;//反射变换后的image1,和image2在一个平面上
    //透视变换,填充必须是白色,白色填充直接用image2覆盖
    warpPerspective(image1, image1Calib, fundamental, Size(image1.cols, image1.rows),INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

    void warpPerspective( InputArray src, OutputArray dst,
                                       InputArray M, Size dsize,
                                       int flags = INTER_LINEAR,
                                       int borderMode = BORDER_CONSTANT,
                                       const Scalar& borderValue = Scalar());
   
    //删除误匹配特征点
    std::vector<KeyPoint> RR_keypoint01, RR_keypoint02;
    std::vector<DMatch> RR_matches;
    int idx = 0;
    for (int i = 0; i < goodMatches.size(); ++i)
    {
        if (RansacStatus[i] != 0)
        {
            RR_keypoint01.emplace_back(R_keypoint01[i]);
            RR_keypoint02.emplace_back(R_keypoint02[i]);
            goodMatches[i].queryIdx = idx;
            goodMatches[i].trainIdx = idx;
            RR_matches.emplace_back(goodMatches[i]);
            ++idx;
        }
    }
    
    Mat imgRRMatches;
    drawMatches(image1, RR_keypoint01, image2, RR_keypoint02, RR_matches, imgRRMatches, Scalar(0, 255, 0), Scalar::all(-1));
    imshow("final match", imgRRMatches);
 
    //image1Calib_copy和image拼接得到output
    Mat image1Calib_copy = image1Calib.clone();
    Mat output = min(image1Calib_copy(Rect(0, 0, image2.cols, image2.rows)),image2);
    imshow("stitching image1", output);
    //output和image1Calib_copy(blackBox)拼接得到output
    Rect blackBox = findBlackBox(image1Calib_copy);
    if(blackBox.area()==0)
    {
        cout<<"no balckBox detected"<<endl;
        return;
    }
    
    //调高image1Calib_copy(blackBox)区域亮度
    Mat innerCode = image1Calib_copy(blackBox);
    //中值滤波
    //medianBlur(innerCode, innerCode, 3);
    //均值滤波
    //blur(innerCode, innerCode, Size(3, 3));
    //高斯滤波
    GaussianBlur(innerCode, innerCode, Size(3, 3), 0);
    
    

    reducePixelValue(innerCode, 150, 0.7);
    innerCode*=1.5;//提亮，平滑等相关运算
    //直方图增强
    //cvtColor(innerCode, innerCode, COLOR_BGR2GRAY);
    //equalizeHist(innerCode, innerCode);
    //cvtColor(output, output, COLOR_BGR2GRAY);
    innerCode.copyTo(output(blackBox));
    
    imshow("stitching image2", output);
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    //hours(小时), minutes（分钟）, seconds（秒）, milliseconds（毫秒）, nanoseconds（纳秒）
    auto cost = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    cout<<"time:"<<cost<<endl;
    waitKey(0);
    
}
double Point2PointDist(const cv::Point2f& a, const cv::Point2f& b)
{
    double res = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    return res;
}


bool findBrightImage(VideoCapture capture, Mat &brightImage)
{
    int count = 0;
    while (true)
    {
        Mat frame;
        capture >> frame;
        if(frame.empty())
        {
            break;
        }
        if(count%1==0)
        {
            Rect blackBox = findBlackBox(frame);
            Rect highLight = findHighLight(frame, 240);
            if(blackBox.area()!=0&&highLight.area()!=0)
            {
                if(highLight.x>blackBox.x+blackBox.width-10&&
                   highLight.x<blackBox.x+blackBox.width+10&&
                   highLight.y>blackBox.y+blackBox.height-10&&
                   highLight.y<blackBox.y+blackBox.height+10)
                {
                    brightImage = frame;
                    return true;
                   
                }
                
            }
        }
        count++;
    
    }
        return false;
}

void findMatches(VideoCapture capture, Mat image1)
{
    
    Rect highLight1 = findHighLight(image1, 220);
    Point2f highLight1Center((highLight1.x+highLight1.width)/2, (highLight1.y+highLight1.height)/2);
    int count = 0;
    while (true)
    {
        Mat image2;
        capture >> image2;
        if(image2.empty())
        {
            break;
        }
        //设置采样率
        if(count%10==0)
        {
            Rect highLight2 = findHighLight(image2, 220);
            Point2f highLight2Center((highLight2.x+highLight2.width)/2, (highLight2.y+highLight2.height)/2);
            if(Point2PointDist(highLight1Center, highLight2Center)>40)
            {
                highlightImageInnerStitching(image1, image2, 200);
            }
            cout<<"************distance"<<Point2PointDist(highLight1Center, highLight2Center)<<"***highLightArea"<<highLight2.area()<<endl;
        }
        count++;
    }
}


int main(int argc, const char * argv[]) {
    VideoCapture capture;
    capture.open("/Users/ysc/Desktop/HighlightRemoval/HighlightRemoval/highlightVideo.mp4");
    Mat brightImage;
    findBrightImage(capture, brightImage);
    findMatches(capture, brightImage);
    
    return 0;
}

