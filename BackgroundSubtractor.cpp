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

int main(){
	
	cv::Mat frame1, frame2, fgMask;
	
	cv::namedWindow("frame",cv::WINDOW_NORMAL);
	cv::resizeWindow("frame",960,536);
	cv::namedWindow("fgMask",cv::WINDOW_NORMAL);
	cv::resizeWindow("fgMask",960,536);
	
	frame2 = cv::imread("./Images/R4_squarefloor.jpg");
	frame1 = cv::imread("./Images/R1_squarefloor.jpg");
	
	cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(frame2, frame2, cv::COLOR_BGR2GRAY);
	
	//create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    
    
    pBackSub = createBackgroundSubtractorMOG2(500,200,true);
    //pBackSub = createBackgroundSubtractorKNN();
    
    //update the background model
    pBackSub->apply(frame1, fgMask);
    pBackSub->apply(frame2, fgMask);
    
    //show the current frame and the fg masks
    cv::imshow("frame", fgMask);
    cv::imshow("fgMask", fgMask);
    cv::waitKey();
    cv::Mat diffrence = frame1 -frame2;
    cv::Mat diffrence2 = frame2 -frame1;
    cv::Mat mask = cv::Mat::zeros(diffrence.size(), CV_8U);
    
    mask.setTo(255, diffrence > 25);
    cv::imshow("frame", mask);
    cv::waitKey();
    
    //Morphologic operations
    cv::Mat element = getStructuringElement(MORPH_RECT,Size(20,20),Point(9,9));
    cv::morphologyEx(mask,mask,MORPH_OPEN,element);
    cv::morphologyEx(mask,mask,MORPH_CLOSE,element);
    
    cv::imshow("frame", mask);
    cv::waitKey();
    
    //Labeling
	cv::Mat stat,centroid,mask2;
	int nLabels = connectedComponentsWithStats(mask, mask2, stat,centroid,8, CV_16U);
	vector<Rect> rComp;
	std::cout << nLabels <<std::endl;
	for (int i=1;i<nLabels;i++)
	{
	     Rect r(Rect(stat.at<int>(i,CC_STAT_LEFT ),stat.at<int>(i,CC_STAT_TOP),stat.at<int>(i,CC_STAT_WIDTH ),stat.at<int>(i,CC_STAT_HEIGHT)));
		rComp.push_back(r);
		std::cout<<" area: " << r << std::endl;
		std::cout<< i <<" area: " << stat.at<int>(i,CC_STAT_AREA) << std::endl;
		std::cout << i << " point left: " << stat.at<int>(i,CC_STAT_LEFT) << std::endl;
		std::cout << i << " point top: " << stat.at<int>(i,CC_STAT_TOP) << std::endl;
		std::cout << i << " point width: " << stat.at<int>(i,CC_STAT_WIDTH) << std::endl;
		std::cout << i << " point height: " << stat.at<int>(i,CC_STAT_HEIGHT) << std::endl;
		cv::rectangle(mask,r,Scalar::all(255),1);

	}
// you can draw all rect
	cv::Mat frame2_masked = cv::Mat::zeros(frame2.size(), CV_8U);
	frame2.copyTo(frame2_masked, mask);
    cv::imshow("fgMask", diffrence);
    cv::imshow("frame", mask);
    cv::waitKey();
    cv::imshow("fgMask", frame2_masked);
    cv::imshow("frame", mask);
    cv::waitKey();
}
