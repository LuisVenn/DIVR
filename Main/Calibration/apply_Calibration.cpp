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

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

int main()
{
	//Check what camera to calibrate
    std::ostringstream name;
    string val;
    std::cout << "What camera to calib? 1 2 or 3?" << std::endl;
    std::cin >> val;
  
	// Extracting path of individual image stored in a given directory
	std::vector<cv::String> images;
	// Path of the folder containing checkerboard images
	name << "./calib" << val << "/*.jpg"; 
	std::string path = name.str();  

	cv::glob(path, images);
	
	cv::namedWindow("Image",cv::WINDOW_NORMAL);
	cv::resizeWindow("Image",960,536);
	
	cv::Mat cameraMatrix,distCoeffs;
	std::ostringstream name2;
	name2 << "./calib" << val << "/CamParams_0" << val << ".xml";
	cv::FileStorage fs(name2.str(), cv::FileStorage::READ);
	
    fs["cameraMatrix"] >> cameraMatrix;
	fs["distCoeffs"] >> distCoeffs;
	

	cv::Mat frame, frameUndistorted;
	// Looping over all the images in the directory
    for(int i{0}; i<images.size(); i++)
    {
		std::ostringstream name3;
		frame = cv::imread(images[i]);
		undistort(frame, frameUndistorted, cameraMatrix, distCoeffs);
		std::ostringstream name;
		
		name3 << "./calib" << val << "/Calibrated_imgs/"<< i << ".jpg"; //!!!!!!!!!!!!!!!!!!
		cv::imwrite(name3.str(), frameUndistorted);
		
		cv::imshow("Image",frameUndistorted);
		
		cv::waitKey(0);
		name.clear();
	  
	}

    cv::destroyAllWindows();
    return 0;
}
