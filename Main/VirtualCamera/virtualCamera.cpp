
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

cv::Size patternSize(6,9);

//vector calcs
void multVector(vector<Point2f> &v, float k){
     for(int i=0; i<v.size(); i++)
     {
		v[i].x = v[i].x*k ;
		v[i].y = v[i].y*k ;
	 }
  }
  
 void diffVectors(vector<Point2f> &a, vector<Point2f> &b){
	 
     for(int i=0; i<a.size(); i++)
     {
		a[i].y = a[i].y - b[i].y;
		a[i].x = a[i].x - b[i].x;
	 }
 }
 
 void sumVectors(vector<Point2f> &a, vector<Point2f> &b){
	 
     for(int i=0; i<a.size(); i++)
     {
		a[i].y = a[i].y + b[i].y;
		a[i].x = a[i].x + b[i].x;
	 }
 }
 
 //(estimates the transformation between the real plane and the estimated, returns mask size for warping)
 cv::Size getTransform(cv::Mat &img1, vector<Point2f> &corners1, vector<Point2f> &corners2, cv::Mat &Ht)
{
	//Get the homography matrix
	cv::Mat H = cv::findHomography(corners1, corners2);

	//Find mask size for warping
	vector<Point2f> original_corners, mask_corners;
	original_corners.push_back(cv::Point2f(0,0));
	original_corners.push_back(cv::Point2f(img1.cols-1,0));
	original_corners.push_back(cv::Point2f(img1.cols-1,img1.rows-1));
	original_corners.push_back(cv::Point2f(0,img1.rows-1));

	cv::perspectiveTransform(original_corners, mask_corners, H);

	cv::Size size_mask_max(-1000,-1000),size_mask_min(1000,1000), size_mask(0,0);

	for(int i=0; i<mask_corners.size(); i++)
	{
		//std::cout << "x:" << mask_corners[i].x << "y:" << mask_corners[i].y << std::endl;
		size_mask_min.height= min((int)mask_corners[i].y,(int)size_mask_min.height);
		size_mask_max.height= max((int)mask_corners[i].y,(int)size_mask_max.height);
		
		size_mask_min.width= min((int)mask_corners[i].x,(int)size_mask_min.width);
		size_mask_max.width= max((int)mask_corners[i].x,(int)size_mask_max.width);
	}

	size_mask.height = size_mask_max.height - size_mask_min.height;
	size_mask.width = size_mask_max.width - size_mask_min.width;

	cv::Mat T = (Mat_<double>(3,3) << 1, 0, -1*size_mask_min.width , 0, 1, -1* size_mask_min.height, 0, 0, 1);
	Ht = T * H; 

	return size_mask;
}

//(defines the angle of warping)
void estimateTransform(cv::Mat imgL,cv::Mat imgR, vector<Point2f> &cornersC_estimated, vector<Point2f> &cornersL, vector<Point2f> &cornersR, vector<Point2f> &cornersL_new, vector<Point2f> &cornersR_new, cv::Mat &Ht_L, cv::Mat &Ht_R, cv::Size &warpedL_size, cv::Size &warpedR_size, float k)
{ 
	
	cornersC_estimated = cornersL;
	//Estimate by assuming linear the corners of C
	diffVectors(cornersC_estimated, cornersR);
	multVector(cornersC_estimated,k);
	sumVectors(cornersC_estimated,cornersR);

	
	warpedL_size = getTransform(imgL, cornersL, cornersC_estimated, Ht_L);
	warpedR_size = getTransform(imgR, cornersR, cornersC_estimated, Ht_R);
	
	cv::perspectiveTransform(cornersL, cornersL_new, Ht_L);
	cv::perspectiveTransform(cornersR, cornersR_new, Ht_R);

}

int horizontal_stitching_calib(cv::Mat &img1, cv::Mat &img2, vector<Point2f> &corners1, vector<Point2f> &corners2)
{	
		
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += img1.cols - corners1[i].x + corners2[i].x;
	}
	
	int avg = round(sum/corners1.size());
	return avg;
}

int vertically_allign_calib(cv::Mat &img1, cv::Mat &img2, vector<Point2f> &corners1, vector<Point2f> &corners2)
{
	
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += corners1[i].y - corners2[i].y;
	}
	int avg = round(sum/corners1.size());

	return avg;
}

void vertically_allign_apply(cv::Mat &img1, cv::Mat &img2, int avg)
{
//Create buffer
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    cv::Scalar value(255,255,255);
  
	//Create Border
	if (avg > 0)
		copyMakeBorder( img2, img2, avg, no, no, no, borderType, value );
	else if (avg < 0)
		copyMakeBorder( img1, img1, -avg, no, no, no, borderType, value );
}

cv::Mat horizontal_stitching_apply(cv::Mat &img1, cv::Mat &img2,int avg)
{
	cv::Mat output = img1.clone();
	
	//Create buffer
	int right = img2.cols - avg;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    int buff = 0;
    cv::Scalar value(255,255,255);
	if(img2.rows >img1.rows) buff = img2.rows-img1.rows;
	
	//Create Border

	copyMakeBorder( img1, output, no, buff, no, right, borderType, value );
	
	//Create Mask
	cv::Mat mask = cv::Mat::zeros(img2.size(), CV_8U);
	cv::Mat mask_output = cv::Mat::zeros(output.size(), CV_8U);
	cv::Mat img2_gray, output_gray;
	cv::cvtColor( img2, img2_gray, cv::COLOR_RGB2GRAY );
	cv::cvtColor( output, output_gray, cv::COLOR_RGB2GRAY);
	
	mask.setTo(255, img2_gray > 0);
    mask_output.setTo(255, output_gray > 0);
    mask_output.setTo(0,output_gray == 255);
    cv::Mat outputBuff = output.clone();
    
    outputBuff.setTo(cv::Scalar(255,255,255));
    output.copyTo(outputBuff,mask_output);
    
	img2.copyTo(outputBuff(cv::Rect(img1.cols-avg,0,img2.cols,img2.rows)),mask);
	
	//Get ghost image
	
	cv::Mat outputBuff2 = outputBuff.clone();
	output.copyTo(outputBuff,mask_output);
	
	cv::Mat outputGhost = outputBuff2*0.5 + outputBuff*0.5;
	
	//If want to show ghost change to outputGhost
	return outputGhost;
}	

cv::Mat cut_frame(cv::Mat &result, cv::Size frame, vector<Point2f> cornersL_new, vector<Point2f> cornersC_estimated,int avg_v)
{	
	cv::Mat output(frame.height,frame.width, CV_8UC3 );
	if(avg_v > 0)
		avg_v = 0;
	//std::cout << "Esq: " << cornersL_new[0].x-cornersC_estimated[0].x << " Top: " << cornersL_new[0].y-cornersC_estimated[0].y +avg_v<< std::endl;
	//std::cout << "cornersL_new.y: " << cornersL_new[0].y << "cornersC_estimated.y: " << cornersC_estimated[0].y << std::endl;
	//std::cout << "avg_v: " << avg_v<< "largura output: " << output.cols << std::endl;
	result(cv::Rect(cornersL_new[0].x-cornersC_estimated[0].x,cornersL_new[0].y-cornersC_estimated[0].y-avg_v,output.cols, output.rows)).copyTo(output(cv::Rect(0,0,output.cols,output.rows)));
	return output;
}

int main() 
{

	//READ IMAGES
	std::cout << "-- Reading Images --" << std::endl;
	cv::Mat imgL, imgR, imgL_warp, imgR_warp, imgCL, imgCR,imgR_calib, imgL_calib;
	
	imgL_calib = cv::imread("./VirtualL/VirtualL2.jpg");
	imgCL= cv::imread("./VirtualC/VirtualC2.jpg");
	imgCR = cv::imread("./VirtualC/VirtualC.jpg");
	imgR_calib = cv::imread("./VirtualR/VirtualR.jpg");
	
	//GET HOMOGRAPHY MATRIX AND SIZE
	std::cout << "-- Getting new calibration parameters --" << std::endl;
	cv::Mat Ht_L, Ht_R;
	cv::Size warpedL_size, warpedR_size;	
	int warpedL_height, warpedL_width , warpedR_height, warpedR_width, avg_v, avg_h;
	
	//find chessboards
	std::cout << "-- Searching for chessboard --" << std::endl;
	vector<Point2f> cornersL, cornersCR, cornersCL, cornersR, cornersL_new, cornersR_new, cornersR_calib,cornersL_calib;
	vector<Point2f> cornersC_estimated;
	bool found1 = cv::findChessboardCorners(imgL_calib, patternSize, cornersL_calib);
	bool found2 = cv::findChessboardCorners(imgR_calib, patternSize, cornersR_calib);
	bool found3 = cv::findChessboardCorners(imgCL, patternSize, cornersCL);
	bool found4 = cv::findChessboardCorners(imgCR, patternSize, cornersCR);
	
	//estimate homography
	std::cout << "-- Estimate homography --" << std::endl;
	float ki,k;	
	cv::Size frame;
	frame.width = imgL_calib.cols;
	frame.height = imgL_calib.rows;
	cv::namedWindow("matches",cv::WINDOW_NORMAL);
	cv::resizeWindow("matches",960,536);
	cv::Mat result, output;	
	
	for( ki = 0; ki < 90; ki+=2.5)
	{
		if(ki<45)
		{
			imgL = imgL_calib.clone();
			imgR = imgCL.clone();
			cornersL = cornersL_calib;
			cornersR = cornersCL;
			k = 1 - ki/45;
		}else if(ki>=45) 
		{
			imgL = imgCR.clone();
			imgR = imgR_calib.clone();
			cornersL = cornersCR;
			cornersR = cornersR_calib;
			k = 1 - (ki-45)/45;
		}
		
		estimateTransform(imgL, imgR, cornersC_estimated, cornersL, cornersR, cornersL_new, cornersR_new, Ht_L, Ht_R, warpedL_size, warpedR_size, k);
		//Warp the prespective to get the value of the horizontal and vertical displacement
		cv::warpPerspective(imgL, imgL_warp, Ht_L, warpedL_size);
		cv::warpPerspective(imgR, imgR_warp, Ht_R, warpedR_size);
		//Warp Point corners
				
		avg_v = vertically_allign_calib(imgL_warp, imgR_warp, cornersL_new, cornersR_new);
		avg_h = horizontal_stitching_calib(imgL_warp, imgR_warp, cornersL_new, cornersR_new);
		
		vertically_allign_apply(imgL_warp, imgR_warp, avg_v);
		
		result = horizontal_stitching_apply(imgL_warp,imgR_warp,avg_h);
	
		output = cut_frame(result, frame, cornersL_new, cornersC_estimated,avg_v); 
		std::cout << "angle: " << ki << std::endl;
		cv::imshow("matches",output);
		cv::waitKey();
		
	}		
}
