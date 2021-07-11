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

vector<Point2f> NewPoint;
	
void onMouse(int event, int x, int y, int flags, void *param)
{
	cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
	switch (event){
	case cv::EVENT_LBUTTONDOWN:
		cout << "at(" << x << "," << y << ")pixs value is:" << static_cast<int>
			(im->at<uchar>(cv::Point(x, y))) << endl;
		NewPoint.push_back(Point2f(x, y));
		break;
	}

}
static void drawEpipolarLines(cv::Mat F, cv::Mat& img1, cv::Mat& img2, vector<Point2f> points1, vector<Point2f> points2)
{
  //Cria uma imagem output com o dobro do tamanho de img1 e cria duas ROIs
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  
  //Cola imagem da esq e da direita no output
  img1.copyTo(outImg(rect1));
  img2.copyTo(outImg(rect2));
  
  std::vector<cv::Vec3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0); //Random number generator
  
  for(size_t i=0; i<points1.size(); i++)
  {
     
    //Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, LINE_AA);
 
    //cv::line(outImg(rect1),
      //cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      //cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      //color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, LINE_AA);
  }
  
  cv::namedWindow("matches",cv::WINDOW_NORMAL);
  cv::resizeWindow("matches",960,536);
  cv::setMouseCallback("matches", onMouse, reinterpret_cast<void *>(&outImg));
  cv::imshow("matches", outImg);
  cv::waitKey(0);
}

static void drawOneLine (cv::Mat F, cv::Mat& img1, cv::Mat& img2, vector<Point2f> points1)
{
  //Cria uma imagem output com o dobro do tamanho de img1 e cria duas ROIs
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  
  //Cola imagem da esq e da direita no output
  img1.copyTo(outImg(rect1));
  img2.copyTo(outImg(rect2));
  
  std::vector<cv::Vec3f> epilines1;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
 
  cv::RNG rng(0); //Random number generator
  
  for(size_t i=0; i<points1.size(); i++)
  {
     
    //Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, LINE_AA);
 
  }
  
  cv::namedWindow("matches2",cv::WINDOW_NORMAL);
  cv::resizeWindow("matches2",960,536);
  cv::imshow("matches2", outImg);
  cv::waitKey(0);
}
int main(){
	
	std::cout << "-- Reading Images --" << std::endl;
	cv::Mat imgL, imgR;
	
	imgL = cv::imread("../Images/squarefloor2/L1_squarefloor2_chess1.jpg");
	imgR = cv::imread("../Images/squarefloor2/R1_squarefloor2_chess1.jpg");
	
	vector<Point2f> cornersL, cornersR;
	
	bool found1 = cv::findChessboardCorners(imgL, patternSize, cornersL);
	bool found2 = cv::findChessboardCorners(imgR, patternSize, cornersR);
	
	cv::Mat F = cv::findFundamentalMat(cornersL, cornersR);
	

	float data[9] = {1301.126792457533, 0, 816.1724470301982, 0, 1309.844817775022, 621.6956484968867, 0, 0, 1 };
	cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32F, data);
	cv::Mat R, t, E;
	E = findEssentialMat(cornersL, cornersR, cameraMatrix); //tem mais parametros
	recoverPose(E, cornersL, cornersR, cameraMatrix, R, t);
	
	std::cout << "Fundamental Matrix: " << F << std::endl;
	std::cout << "Essencial Matrix: " << E << std::endl;
	std::cout << "R Matrix: " << R << std::endl;
	std::cout << "t Matrix: " << t << std::endl;
	std::cout << "Intrinsics Matrix: " << cameraMatrix << std::endl;
	
	drawEpipolarLines(F,imgL,imgR,cornersL,cornersR);
	
	//Draw one line
	
	
	drawOneLine(F,imgL,imgR,NewPoint);
}
