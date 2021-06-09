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

#include <fstream>
using namespace std;
using namespace cv;
using namespace cv::detail;

 void DivideVector(vector<Point2f> &v){
     for(int i=0; i<v.size(); i++)
     {
		v[i].x = v[i].x/2 ;
		v[i].y = v[i].y/2 ;
	 }
	 
		
  }
 void SumVectors(vector<Point2f> &a, vector<Point2f> &b){
	 
     for(int i=0; i<a.size(); i++)
     {
		a[i].y = a[i].y + b[i].y;
		a[i].x = a[i].x + b[i].x;
	 }
 }
  
int main()
{
//Variables	
cv::Size patternSize(6,9);
cv::Mat imgL, imgC, imgR;
vector<Point2f> corners1, corners2,corners3;

cv::namedWindow("imgL",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL",960,536);
cv::namedWindow("imgC",cv::WINDOW_NORMAL);
cv::resizeWindow("imgC",960,536);
cv::namedWindow("imgR",cv::WINDOW_NORMAL);
cv::resizeWindow("imgR",960,536);

//Read Images
imgL = cv::imread("./L1.jpg");
imgR = cv::imread("./R1.jpg");
imgC = cv::imread("./C1.jpg");

cv::imshow("imgL",imgL);
cv::imshow("imgC",imgC);
cv::imshow("imgR",imgR);

//Find Chessboards
bool found1 = cv::findChessboardCorners(imgL, patternSize, corners1);
bool found2 = cv::findChessboardCorners(imgC, patternSize, corners2);
bool found3 = cv::findChessboardCorners(imgR, patternSize, corners3);

//Find Homographies
cv::Mat H_LC = cv::findHomography(corners1, corners2);
cv::Mat H_RC = cv::findHomography(corners3, corners2);
cv::Mat H_CL = cv::findHomography(corners2, corners1);
cv::Mat H_CR = cv::findHomography(corners2, corners3);

//Save Corners To CSV File

std::ofstream myfile;
myfile.open ("Corners.csv");
myfile << "Xl,Yl,Xc,Yc,Xr,Yr,\n";

for(int i=0; i<corners1.size(); i++)
{
	
	myfile << corners1[i].x << " , " << corners1[i].y << "," << corners2[i].x << "," << corners2[i].y << "," << corners3[i].x << "," << corners3[i].y << ",\n";
	
}
myfile.close();


//Show matrices
std::cout << "H_LC:\n" << H_LC << std::endl;
std::cout << "H_CR:\n" << H_CR << std::endl;
std::cout << "H_RC:\n" << H_RC << std::endl;
std::cout << "H_CR:\n" << H_CR << std::endl;

//CHECK THE DIFERENCE BETWEEN DIRECT AND THE INVERSES
std::cout << "Inverse relation to the direct LC,CL" << std::endl;
cv::Mat H_LC_inv = H_LC.inv();

//Warp Prespectives
cv::Mat imgL_C, imgL_C_L;
cv::warpPerspective(imgL, imgL_C, H_LC, imgL.size());
cv::warpPerspective(imgL_C, imgL_C_L, H_LC_inv, imgL_C.size());

//Create windows
cv::namedWindow("imgL_C",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C",960,536);
cv::namedWindow("imgL_C_L",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C_L",960,536);

//Show images
cv::imshow("imgL_C",imgL_C);
cv::imshow("imgL_C_L",imgL_C_L);

//Show matrices
std::cout << "H_LC:\n" << H_LC << std::endl;
std::cout << "H_CL:\n" << H_CL << std::endl;
std::cout << "H_LC_inv:\n" << H_LC_inv << std::endl;

cv::waitKey();

cv::destroyWindow("imgL_C");
cv::destroyWindow("imgL_C_L");

//CHECK THE DIFERENCE BETWEEN DIRECT AND THE INVERSES
std::cout << "Relation between transformation LC and CR" << std::endl;

//Warp Prespectives
cv::Mat imgC_R, imgC_R_false;
cv::warpPerspective(imgC, imgC_R, H_CR, imgC.size());
cv::warpPerspective(imgC, imgC_R_false, H_LC, imgC.size());

//Create windows
cv::namedWindow("imgC_R",cv::WINDOW_NORMAL);
cv::resizeWindow("imgC_R",960,536);
cv::namedWindow("imgC_R_false",cv::WINDOW_NORMAL);
cv::resizeWindow("imgC_R_false",960,536);

//Show images
cv::imshow("imgC_R",imgC_R);
cv::imshow("imgC_R_false",imgC_R_false);

//Show matrices
std::cout << "H_LC:\n" << H_LC << std::endl;
std::cout << "H_CR:\n" << H_CR << std::endl;

cv::waitKey();

cv::destroyWindow("imgC_R");
cv::destroyWindow("imgC_R_false");

//CHECK THE DIFERENCE BETWEEN DIRECT AND THE INVERSES
std::cout << "Relation between transformation LC+CR and LR" << std::endl;

cv::Mat H_LR_combine = H_CR * H_LC;

//Warp Prespectives
cv::Mat imgL_C_R, imgL_R_combine;
cv::warpPerspective(imgL, imgL_C, H_LC, imgL.size());
cv::warpPerspective(imgL_C, imgL_C_R, H_CR, imgL_C.size());
cv::warpPerspective(imgL, imgL_R_combine, H_LR_combine, imgL.size());

//Create windows
cv::namedWindow("imgL_C_R",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C_R",960,536);
cv::namedWindow("imgL_R_combine",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_R_combine",960,536);

//Show images
cv::imshow("imgL_C_R",imgL_C_R);
cv::imshow("imgL_R_combine",imgL_R_combine);

//Show matrices
std::cout << "H_LR_combine:\n" << H_LR_combine << std::endl;

cv::waitKey();

cv::destroyWindow("imgL_C_R");
cv::destroyWindow("imgL_R_combine");

//CHECK THE DIFERENCE BETWEEN DIRECT AND THE INVERSES
std::cout << "Relation between half transformations" << std::endl;

cv::Mat H_LC2 = H_LC/2;
cv::Mat H_CL2 = H_CL/2;

//Warp Prespectives
cv::Mat imgL_C2, imgC_L2;
cv::warpPerspective(imgL, imgL_C2, H_LC2, imgL.size());
cv::warpPerspective(imgC, imgC_L2, H_CL2, imgL_C.size());


//Create windows
cv::namedWindow("imgL_C2",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C2",960,536);
cv::namedWindow("imgC_L2",cv::WINDOW_NORMAL);
cv::resizeWindow("imgC_L2",960,536);

//Show images
cv::imshow("imgL_C2",imgL_C2);
cv::imshow("imgC_L2",imgC_L2);

//Show matrices
std::cout << "H_LC2:\n" << H_LC2 << std::endl;
std::cout << "H_CL2:\n" << H_CL2 << std::endl;

cv::waitKey();

cv::destroyWindow("imgL_C2");
cv::destroyWindow("imgC_L2");

//CHECK THE DIFERENCE BETWEEN DIRECT AND THE INVERSES
std::cout << "Relation between half transformations LR RL" << std::endl;

cv::Mat H_LR = cv::findHomography(corners1, corners3);
cv::Mat H_RL = cv::findHomography(corners3, corners1);

cv::Mat H_LR2 = H_LR/2;
cv::Mat H_RL2 = H_RL/2;

//Warp Prespectives
cv::Mat imgL_R2, imgR_L2;
cv::warpPerspective(imgL, imgL_R2, H_LR2, imgL.size());
cv::warpPerspective(imgR, imgR_L2, H_RL2, imgR.size());


//Create windows
cv::namedWindow("imgL_R2",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_R2",960,536);
cv::namedWindow("imgR_L2",cv::WINDOW_NORMAL);
cv::resizeWindow("imgR_L2",960,536);

//Show images
cv::imshow("imgL_R2",imgL_R2);
cv::imshow("imgR_L2",imgR_L2);

//Show matrices
std::cout << "H_LR2:\n" << H_LR2 << std::endl;
std::cout << "H_RL2:\n" << H_RL2 << std::endl;

cv::waitKey();

cv::destroyWindow("imgL_R2");
cv::destroyWindow("imgR_L2");

//*************** ESTIMATE C CORNERS **********************
std::cout << "Estimate C corners" << std::endl;

vector<Point2f> corners2_estimated = corners1;
SumVectors(corners2_estimated, corners3);
std::cout << "somei" << std::endl;
DivideVector(corners2_estimated);
std::cout << "dividi" << std::endl;

std::cout << corners2_estimated << std::endl;

cv::Mat H_LC_estimated = cv::findHomography(corners1, corners2_estimated);
cv::Mat H_RC_estimated = cv::findHomography(corners3, corners2_estimated);

//Warp Prespectives
cv::Mat imgL_C_estimated, imgR_C_estimated, imgR_C;
cv::warpPerspective(imgL, imgL_C_estimated, H_LC_estimated, imgL.size());
cv::warpPerspective(imgR, imgR_C_estimated, H_RC_estimated, imgR.size());
cv::warpPerspective(imgR, imgR_C, H_RC, imgR.size());

//Create windows
cv::namedWindow("imgL_C_estimated",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C_estimated",960,536);
cv::namedWindow("imgR_C_estimated",cv::WINDOW_NORMAL);
cv::resizeWindow("imgR_C_estimated",960,536);
cv::namedWindow("imgL_C",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C",960,536);
cv::namedWindow("imgR_C",cv::WINDOW_NORMAL);
cv::resizeWindow("imgR_C",960,536);

//Show images
cv::imshow("imgL_C_estimated",imgL_C_estimated);
cv::imshow("imgR_C_estimated",imgR_C_estimated);
cv::imshow("imgL_C",imgL_C);
cv::imshow("imgR_C",imgR_C);
//Show matrices
std::cout << "H_LC:\n" << H_LC << std::endl;
std::cout << "H_LC_estimated:\n" << H_LC_estimated << std::endl;
std::cout << "H_RC:\n" << H_RC << std::endl;
std::cout << "H_RC_estimated:\n" << H_RC_estimated << std::endl;
cv::waitKey();

cv::destroyWindow("imgL_C_estimated");
cv::destroyWindow("imgR_C_estimated");
cv::destroyWindow("imgL_C");
cv::destroyWindow("imgR_C");
}
