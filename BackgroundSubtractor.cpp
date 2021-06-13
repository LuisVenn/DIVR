#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/types.hpp>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"

#include <string>

#include <fstream>
using namespace std;
using namespace cv;
using namespace cv::detail;

void match_features(cv::Mat &img1, cv::Mat &img2, cv::Mat &output, cv::Mat mask1, cv::Mat mask2)
{
	//Features dectection and matching
	Ptr<ORB> detector = cv::ORB::create();
	std::vector<KeyPoint> keypoints1,keypoints2;
	Mat descriptors1,descriptors2;

	detector->detectAndCompute( img1, mask1, keypoints1, descriptors1 );
	detector->detectAndCompute( img2, mask2, keypoints2, descriptors2 );

	//Matching FEatures
	BFMatcher matcher(NORM_L2);
	//matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	std::vector<vector<DMatch> > matches;

	matcher.knnMatch(descriptors1, descriptors2, matches,2);

	std::vector<DMatch> match1;
	std::vector<DMatch> match2;
	int i2 = 0;
	
	match1.push_back(matches[0][0]);
	for(int i=1; i<matches.size(); i++)
	{
		if( matches[i][0].distance < match1[0].distance ){
			match1[0]=matches[i][0];
			
			
		}
		
	}
	
	cv::drawMatches(img1, keypoints1, img2, keypoints2, match1, output);
}

void getMask(cv::Mat &frame1, cv::Mat &frame2, cv::Mat &mask)
{
	cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(frame2, frame2, cv::COLOR_BGR2GRAY);
	
	//create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    
    pBackSub = createBackgroundSubtractorMOG2(500,200,true);
    //pBackSub = createBackgroundSubtractorKNN();
    
    //update the background model
    cv::Mat fgMask;
    
    pBackSub->apply(frame1, fgMask);
    pBackSub->apply(frame2, fgMask);
    
    //show the current frame and the fg masks

    cv::Mat diffrence = frame1 -frame2;
    mask = cv::Mat::zeros(diffrence.size(), CV_8U);
    
    mask.setTo(255, diffrence > 25);

    //Morphologic operations
    cv::Mat element = getStructuringElement(MORPH_RECT,Size(20,20),Point(9,9));
    cv::morphologyEx(mask,mask,MORPH_OPEN,element);
    cv::morphologyEx(mask,mask,MORPH_CLOSE,element);
    cv::morphologyEx(mask,mask,MORPH_DILATE,element);
}

int main(){
	
	cv::Mat frame1, frame2;
	cv::Mat frame12, frame22;
	
	cv::namedWindow("frame",cv::WINDOW_NORMAL);
	cv::resizeWindow("frame",960,536);
	cv::namedWindow("fgMask",cv::WINDOW_NORMAL);
	cv::resizeWindow("fgMask",960,536);
	cv::namedWindow("mask",cv::WINDOW_NORMAL);
	cv::resizeWindow("mask",960,536);
	
	frame2 = cv::imread("./Images/L4_squarefloor.jpg");
	frame1 = cv::imread("./Images/L1_squarefloor.jpg");
	frame22 = cv::imread("./Images/R4_squarefloor.jpg");
	frame12 = cv::imread("./Images/R1_squarefloor.jpg");
	
	cv::Mat mask, mask2;
	
	getMask(frame1,frame2,mask);
    getMask(frame12,frame22,mask2);
    
    cv::Mat result;
    
    match_features(frame2,frame22,result, mask, mask2);
    
    cv::imshow("fgMask", result);
    cv::imshow("frame", mask);
    cv::imshow("mask", mask2);
    cv::waitKey();
    
    //Labeling
	//cv::Mat stat,centroid,mask2;
	//int nLabels = connectedComponentsWithStats(mask, mask2, stat,centroid,8, CV_16U);
	//vector<Rect> rComp;
	//std::cout << nLabels <<std::endl;
	//for (int i=1;i<nLabels;i++)
	//{
	     //Rect r(Rect(stat.at<int>(i,CC_STAT_LEFT ),stat.at<int>(i,CC_STAT_TOP),stat.at<int>(i,CC_STAT_WIDTH ),stat.at<int>(i,CC_STAT_HEIGHT)));
		//rComp.push_back(r);
		//std::cout<<" area: " << r << std::endl;
		//std::cout<< i <<" area: " << stat.at<int>(i,CC_STAT_AREA) << std::endl;
		//std::cout << i << " point left: " << stat.at<int>(i,CC_STAT_LEFT) << std::endl;
		//std::cout << i << " point top: " << stat.at<int>(i,CC_STAT_TOP) << std::endl;
		//std::cout << i << " point width: " << stat.at<int>(i,CC_STAT_WIDTH) << std::endl;
		//std::cout << i << " point height: " << stat.at<int>(i,CC_STAT_HEIGHT) << std::endl;
		//cv::rectangle(mask,r,Scalar::all(255),1);

	//}
//// you can draw all rect
	//cv::Mat frame2_masked = cv::Mat::zeros(frame2.size(), CV_8U);
	//frame2.copyTo(frame2_masked, mask);
    //cv::imshow("fgMask", diffrence);
    //cv::imshow("frame", mask);
    //cv::waitKey();
    //cv::imshow("fgMask", frame2_masked);
    //cv::imshow("frame", mask);
    //cv::waitKey();
}
