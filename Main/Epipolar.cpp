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
  cv::setMouseCallback("matches2", onMouse, reinterpret_cast<void *>(&outImg));
  cv::imshow("matches2", outImg);
  cv::waitKey(0);
}
int main(){
	bool import;
	cv::Mat F, R, t, E, cameraMatrix;
	vector<Point2f> totalcornersL, totalcornersR;
	std::cout << "Import Calibration? 1-yes 0-no " << std::endl; // Type a number and press enter
	cin >> import;
		
	if(import)
	{
	//Read pre-calibrated saved parameters
		cv::FileStorage fs("./stereoparamsEpipolar.xml", cv::FileStorage::READ);

		fs["Fmat"] >> F;
		fs["Essencial"] >> E;
		fs["Rot"] >> R;
		fs["Trns"] >> t; 
		fs["MintL"] >> cameraMatrix; 
		
	}else
	{
	
		std::cout << "-- Reading Images --" << std::endl;
		// Extracting path of individual image stored in a given directory
		std::vector<cv::String> imagesL, imagesR;
		
		// Path of the folder containing checkerboard images
		std::string pathR = "./Depth_Disp/L/*.jpg"; //!!!!!!!!!!!!!!!!!!!!
		std::string pathL = "./Depth_Disp/R/*.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!

		cv::glob(pathL, imagesL);
		cv::glob(pathR, imagesR);

		cv::Mat frameL, frameR, grayL, grayR;
		// vector to store the pixel coordinates of detected checker board corners 
		vector<Point2f> cornersL, cornersR;
		bool successL, successR;

		// Looping over all the images in the directory
		for(int i{0}; i<imagesR.size(); i++)
		{
		  
			std::cout << imagesL[i] << std::endl;
			std::cout << imagesR[i] << std::endl;
		  
			frameL = cv::imread(imagesL[i]);
			cv::cvtColor(frameL,grayL,cv::COLOR_BGR2GRAY);
	  
			frameR = cv::imread(imagesR[i]);
			cv::cvtColor(frameR,grayR,cv::COLOR_BGR2GRAY);

			// Finding checker board corners
			// If desired number of corners are found in the image then success = true  
			successL = cv::findChessboardCorners(grayL, patternSize, cornersL);
			successR = cv::findChessboardCorners(grayR, patternSize, cornersR);
		  
			/*
			* If desired number of corner are detected,
			* we refine the pixel coordinates and display 
			* them on the images of checker board
			*/
			if((successL) && (successR))
			{
				cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

				// refining pixel coordinates for given 2d points.
				cv::cornerSubPix(grayL,cornersL,cv::Size(11,11), cv::Size(-1,-1),criteria);
				cv::cornerSubPix(grayR,cornersR,cv::Size(11,11), cv::Size(-1,-1),criteria);

				for(int i = 0 ; i< cornersL.size(); i++)
				{
					totalcornersL.push_back(cornersL[i]);
					totalcornersR.push_back(cornersR[i]);
				}
			}
		
		}
		F = cv::findFundamentalMat(totalcornersL, totalcornersR);
	
		float data[9] = {1301.126792457533, 0, 816.1724470301982, 0, 1309.844817775022, 621.6956484968867, 0, 0, 1 };
		cameraMatrix = cv::Mat(3, 3, CV_32F, data);
		E = findEssentialMat(cornersL, cornersR, cameraMatrix); //tem mais parametros
		recoverPose(E, cornersL, cornersR, cameraMatrix, R, t);
		
		cv::FileStorage cv_file = cv::FileStorage("./stereoparamsEpipolar.xml", cv::FileStorage::WRITE);
		cv_file.write("Fmat",F);
		cv_file.write("Essencial",E); 
		cv_file.write("Rot",R);
		cv_file.write("Trns",t);
		cv_file.write("MintL",cameraMatrix);
		
	}
	
	//Manual import of matrix R and t
	/*
	float  dataR[9] = { 0.7071 , 0.5000  , -0.5000 ,   -0.5000  ,  0.8536 ,   0.1464,    0.5000,    0.1464,    0.8536};
	R =  cv::Mat(3, 3, CV_32F, dataR);
	float  datat[9] = { -0.0742, -0.0525, -0.0525};
	t =  cv::Mat(3, 1, CV_32F, datat);
	*/
		
	std::cout << "Fundamental Matrix: " << F << std::endl;
	std::cout << "Essencial Matrix: " << E << std::endl;
	std::cout << "R Matrix: " << R << std::endl;
	std::cout << "t Matrix: " << t << std::endl;
	std::cout << "Intrinsics Matrix: " << cameraMatrix << std::endl;
	
	//Draw one line
	//drawEpipolarLines(F,frameL,frameR,totalcornersL,totalcornersR);
	cv::Mat frameL = cv::imread("./Depth_Disp/L/L50.jpg");
	cv::Mat frameR = cv::imread("./Depth_Disp/R/R50.jpg");
	
	cv::namedWindow("matches",cv::WINDOW_NORMAL);
	cv::resizeWindow("matches",960,536);
	cv::setMouseCallback("matches", onMouse, reinterpret_cast<void *>(&frameL));
	cv::imshow("matches", frameL);
	cv::waitKey(0);
  
	drawOneLine(F,frameL,frameR,NewPoint);
}
