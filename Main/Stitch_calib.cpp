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
void estimateTransform(cv::Mat imgL,cv::Mat imgR, vector<Point2f> &cornersL, vector<Point2f> &cornersR, cv::Mat &Ht_L, cv::Mat &Ht_R, cv::Size &warpedL_size, cv::Size &warpedR_size, float k)
{ 
	
	vector<Point2f> cornersC_estimated = cornersL;
	//Estimate by assuming linear the corners of C
	
	diffVectors(cornersC_estimated, cornersR);
	multVector(cornersC_estimated,k);
	sumVectors(cornersC_estimated,cornersR);
	warpedL_size = getTransform(imgL, cornersL, cornersC_estimated, Ht_L);
	warpedR_size = getTransform(imgR, cornersR, cornersC_estimated, Ht_R);
}

int horizontal_stitching_calib(cv::Mat &img1, cv::Mat &img2)
{
	vector<Point2f> corners1, corners2;
	
	bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
	bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);
	
	if(found1 * found2 == 0)
		std::cout << "ERROR: CHESS NOT FOUND!! IMGL: " << found1 << " IMGR: " << found2 << std::endl;
	else
		std::cout << "Chess found" << std::endl;
		
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += img1.cols - corners1[i].x + corners2[i].x;
	}
	
	int avg = round(sum/corners1.size());
	return avg;
}

int vertically_allign_calib(cv::Mat &img1, cv::Mat &img2)
{
	vector<Point2f> corners1, corners2;
	
	bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
	bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);
	
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += corners1[i].y - corners2[i].y;
	}
	int avg = round(sum/corners1.size());

	return avg;
}
int main() 
{

	//READ IMAGES
	std::cout << "-- Reading Images --" << std::endl;
	cv::Mat imgL, imgR, imgL_warp, imgR_warp;
	
	imgL = cv::imread("../Images/squarefloor2/L1_squarefloor2_chess1.jpg");
	imgR = cv::imread("../Images/squarefloor2/R1_squarefloor2_chess1.jpg");
	
	//GET HOMOGRAPHY MATRIX AND SIZE
	std::cout << "-- Getting new calibration parameters --" << std::endl;
	cv::Mat Ht_L, Ht_R;
	cv::Size warpedL_size, warpedR_size;	
	int warpedL_height, warpedL_width , warpedR_height, warpedR_width, avg_v, avg_h;;
	
	//find chessboards
	std::cout << "-- Searching for chessboard --" << std::endl;
	vector<Point2f> cornersL, cornersC, cornersR;
	bool found1 = cv::findChessboardCorners(imgL, patternSize, cornersL);
	bool found2 = cv::findChessboardCorners(imgR, patternSize, cornersR);
	
	//estimate homography
	std::cout << "-- Estimate homography --" << std::endl;
	float k;	
	std::cout << "Position? (0º-45º)" << std::endl; // Type a number and press enter
	cin >> k;
			
	k = k/45;
	estimateTransform(imgL, imgR, cornersL, cornersR, Ht_L, Ht_R, warpedL_size, warpedR_size, k);
	
	//Warp the prespective to get the value of the horizontal and vertical displacement
	cv::warpPerspective(imgL, imgL_warp, Ht_L, warpedL_size);
	cv::warpPerspective(imgR, imgR_warp, Ht_R, warpedR_size);
			
	avg_v = vertically_allign_calib(imgL_warp, imgR_warp);
	avg_h = horizontal_stitching_calib(imgL_warp, imgR_warp);
	
	//SAVE NEW HOMOGRAPHY MATRIX AND PARAMETERS
	std::cout << "-- Saving new parameters --" << std::endl;
	std::ostringstream name;
    string val;
    std::cout << "New parameters file name?" << std::endl;
    std::cin >> val;
    name << "./Homography_params_" << val << ".xml";
    
	warpedL_height = warpedL_size.height;
	warpedL_width = warpedL_size.width;
	warpedR_height = warpedR_size.height;
	warpedR_width = warpedR_size.width;
	
	cv::FileStorage cv_file = cv::FileStorage(name.str(), cv::FileStorage::WRITE);
	cv_file.write("Ht_L",Ht_L);
	cv_file.write("Ht_R",Ht_R);
	cv_file.write("warpedL_height",warpedL_height);
	cv_file.write("warpedL_width",warpedL_width);
	cv_file.write("warpedR_height",warpedR_height);
	cv_file.write("warpedR_width",warpedR_width);
	cv_file.write("avg_v",avg_v);
	cv_file.write("avg_h",avg_h); 
	
	//SHOW RESULTS
	std::cout << "Ht_L:" << std::endl;
	std::cout << Ht_L << std::endl;
	std::cout << "Ht_R:" << std::endl;
	std::cout << Ht_R << std::endl;
}

