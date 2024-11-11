//
// Created by ysc on 2023/2/28.
//
#include "highlightRemoval.h"


highlightRemoval::highlightRemoval() {



}
highlightRemoval::~highlightRemoval() {

}

Rect highlightRemoval::findHighLight(Mat image, int lower_threshold)
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

Rect highlightRemoval::findBlackBox(Mat image)
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

void highlightRemoval::highlightImageStitching(Mat image1, Mat image2)
{

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
    //step 5.2 compute homography
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

    //step 5.3  delete mismatched points
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
    //imshow("final match", imgRRMatches);
    Mat image1Calib_copy = image1Calib.clone();
    Mat output = min(image1Calib_copy(Rect(0, 0, image2.cols, image2.rows)),image2);
    //imshow("stitching image", output);
    //waitKey(0);

}

bool highlightRemoval::highlightImageInnerStitching(Mat image1, Mat image2, int setMatches)
{

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
    if(goodMatches.size()<setMatches)
    {
        cout<<"few matches"<<endl;
        return false;
    }

    //step 5.2 compute homography
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

    //step 5.3  delete mismatched points
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


    //image1Calib_copy和image拼接得到output
    Mat image1Calib_copy = image1Calib.clone();
    Mat output = min(image1Calib_copy(Rect(0, 0, image2.cols, image2.rows)),image2);

    //output和image1Calib_copy(blackBox)拼接得到output
    Rect blackBox = findBlackBox(image1Calib_copy);
    if(blackBox.area()==0)
    {
        cout<<"no balckBox detected"<<endl;
        return false;
    }

    //调高image1Calib_copy(blackBox)区域亮度
    Mat innerCode = image1Calib_copy(blackBox);
    innerCode*=1.6;//提亮，平滑等相关运算
    innerCode.copyTo(output(blackBox));
    stitchedImage = output;
    return true;


}

double highlightRemoval::Point2PointDist(const Point2f& a, const Point2f& b)
{
    double res = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    return res;
}

bool highlightRemoval::findBrightImage(Mat &frame)
{
    Rect blackBox = findBlackBox(frame);
    rectangle(frame, blackBox, Scalar(100, 100, 100));
    Rect highLight = findHighLight(frame, 240);
    rectangle(frame, highLight, Scalar(255, 0, 0));
    if(blackBox.area()!=0 && highLight.area()!=0) {
        if (highLight.x > blackBox.x + blackBox.width - 50 &&
            highLight.x < blackBox.x + blackBox.width + 50 &&
            highLight.y > blackBox.y + blackBox.height - 50 &&
            highLight.y < blackBox.y + blackBox.height + 50) {
            brightImage = frame;
            return true;
        }
        else
        {
            return false;
        }


    }
    else
    {
        return false;
    }

}



bool highlightRemoval::findMatches(Mat &frame)
{

    Rect highLight1 = findHighLight(brightImage, 240);
    Point2f highLight1Center((highLight1.x+highLight1.width)/2, (highLight1.y+highLight1.height)/2);
    Rect highLight2 = findHighLight(frame, 240);
    Point2f highLight2Center((highLight2.x+highLight2.width)/2, (highLight2.y+highLight2.height)/2);
    bool stitchFlag = false;
    if(Point2PointDist(highLight1Center, highLight2Center)>40)
    {
        stitchFlag = highlightImageInnerStitching(brightImage, frame, 200);
        if(stitchFlag)
        {
            frame = stitchedImage;
        }
        return stitchFlag;
    }
    else
    {
        return false;
    }


}
