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

cv::Mat H_LR = H_CR * H_LC;

//Warp Prespectives
cv::Mat imgL_C_R, imgL_R;
cv::warpPerspective(imgL, imgL_C, H_LC, imgL.size());
cv::warpPerspective(imgL_C, imgL_C_R, H_CR, imgL_C.size());
cv::warpPerspective(imgL, imgL_R, H_LR, imgL.size());

//Create windows
cv::namedWindow("imgL_C_R",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_C_R",960,536);
cv::namedWindow("imgL_R",cv::WINDOW_NORMAL);
cv::resizeWindow("imgL_R",960,536);

//Show images
cv::imshow("imgL_C_R",imgL_C_R);
cv::imshow("imgL_R",imgL_R);

//Show matrices
std::cout << "H_LR:\n" << H_LR << std::endl;

cv::waitKey();

cv::destroyWindow("imgL_C_R");
cv::destroyWindow("imgL_R");

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
cv::imshow("imgL_C2",imgL_C_R);
cv::imshow("imgC_L2",imgL_R);

//Show matrices
std::cout << "H_LC2:\n" << H_LC2 << std::endl;
std::cout << "H_CL2:\n" << H_CL2 << std::endl;

cv::waitKey();

cv::destroyWindow("imgL_C2");
cv::destroyWindow("imgC_L2");
}
