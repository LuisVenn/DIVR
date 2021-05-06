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
	//cv::namedWindow("Image",cv::WINDOW_NORMAL);
	//cv::resizeWindow("Image",960,536);
	//cv::namedWindow("Image2",cv::WINDOW_NORMAL);
	//cv::resizeWindow("Image2",960,536);
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j,i,0));
  }


  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  std::string path = "./calib_2cam_03_2/*.jpg";  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  cv::glob(path, images);

  cv::Mat frame, gray,frameUndistorted;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success;

  // Looping over all the images in the directory
  for(int i{0}; i<images.size(); i++)
  {
    frame = cv::imread(images[i]);
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
      std::cout << "A procurar corners\n";
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
	  std::cout << "Acabei de procurar corners\n";
    /* 
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checker board
    */
    if(success)
    {
      cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER +
                cv::TermCriteria::EPS, 30, 0.001);
      
      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
      
      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
      
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }

    //cv::imshow("Image",frame);
    //cv::waitKey(0);
    std::cout << "Next Frame \n";
  }

  

  cv::Mat cameraMatrix,distCoeffs,R,T;

  /*
   * Performing camera calibration by 
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the 
   * detected corners (imgpoints)
  */

  cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);

  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;
 
  
  int a;
  for(int i{0}; i<images.size(); i++)
  {
	  frame = cv::imread(images[i]);
	  undistort(frame, frameUndistorted, cameraMatrix, distCoeffs);
	  a = i + 1;
	  std::ostringstream name;
	  name << "./calib_2cam_03_2/Ind_Calibrated/" << a << ".jpg"; //!!!!!!!!!!!!!!!!!!
	  cv::imwrite(name.str(), frameUndistorted);
	  //cv::imshow("Image",frameUndistorted);
	  //cv::imshow("Image2",frame);
	  //cv::waitKey(0);
	  name.clear();
	  
  }
  
  // Export Camera Matrix to File
	int n_camera = 03; //!!!!!!!!!!!

	ofstream myfile;
	myfile.open("CamParams_03_2.txt"); //!!!!!!!!!!!!11
	
	myfile << "Camera Matrix nº" << n_camera << '\n' ;
	myfile << cameraMatrix << '\n';
	myfile << "DistCoeffs \n" ;
	myfile << distCoeffs << '\n';
	myfile << "R \n" ;
	myfile << R << '\n';
	myfile << "T \n" ;
	myfile << T;
	
	myfile.close();
	
  cv::destroyAllWindows();
  return 0;
}
