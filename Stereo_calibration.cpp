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


// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

int main()
{
  
// Creating vector to store vectors of 3D points for each checkerboard image
std::vector<std::vector<cv::Point3f> > objpoints;

// Creating vector to store vectors of 2D points for each checkerboard image
std::vector<std::vector<cv::Point2f> > imgpointsL, imgpointsR;

// Defining the world coordinates for 3D points
std::vector<cv::Point3f> objp;
for(int i{0}; i<CHECKERBOARD[1]; i++)
{
  for(int j{0}; j<CHECKERBOARD[0]; j++)
    objp.push_back(cv::Point3f(j,i,0));
}

// Extracting path of individual image stored in a given directory
std::vector<cv::String> imagesL, imagesR;
// Path of the folder containing checkerboard images
std::string pathL = "./Disparity_map/Depth_map/Vertical_Curve/StereoCalib_L_2/*.jpg"; //!!!!!!!!!!!!!!!!!!!!
std::string pathR = "./Disparity_map/Depth_map/Vertical_Curve/StereoCalib_R_2/*.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!

cv::glob(pathL, imagesL);
cv::glob(pathR, imagesR);

cv::Mat frameL, frameR, grayL, grayR,croppedL,croppedR;
// vector to store the pixel coordinates of detected checker board corners 
std::vector<cv::Point2f> corner_ptsL, corner_ptsR;
bool successL, successR;

// Looping over all the images in the directory
cv::Mat frameL_border,frameR_border;
for(int i{0}; i<imagesL.size(); i++)
{
  cv::namedWindow("ImageL",cv::WINDOW_NORMAL);
  cv::resizeWindow("ImageL",960,536);
  cv::namedWindow("ImageR",cv::WINDOW_NORMAL);
  cv::resizeWindow("ImageR",960,536);
  
  frameL = cv::imread(imagesL[i]);
  //cv::cvtColor(frameL,grayL,cv::COLOR_BGR2GRAY);

  frameR = cv::imread(imagesR[i]);
  //cv::cvtColor(frameR,grayR,cv::COLOR_BGR2GRAY);
  //int x = frameL.cols/3;
  //cv::Range cols(x*2, frameL.cols);
  //cv::Range rows(0, frameL.rows);
  //cv::Mat croppedL = frameL(rows, cols);
  
  //cv::Range cols2(0, x);
  //croppedL = frameL(rows, cols);
  //croppedR = frameR(rows, cols2);
  //Initialize arguments for the filter
  
  int top, bottom, left, right; 
  int borderType = cv::BORDER_CONSTANT;
  top = (int) (0.05*frameL.rows); bottom = top;
  left = (int) (0.05*frameL.cols); right = left;
  int no = 0;
  cv::Scalar value(255,255,255);
  
//Create Border
  copyMakeBorder( frameL, frameL_border, top, bottom, left, right, borderType, value );
  copyMakeBorder( frameR, frameR_border, top, bottom, left, left, borderType, value );

  cv::cvtColor(frameL_border,grayL,cv::COLOR_BGR2GRAY);
  cv::cvtColor(frameR_border,grayR,cv::COLOR_BGR2GRAY);
  // Finding checker board corners
  // If desired number of corners are found in the image then success = true  
  successL = cv::findChessboardCorners(
    grayL,
    cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]),
    corner_ptsL);
    // cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

  successR = cv::findChessboardCorners(
    grayR,
    cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]),
    corner_ptsR);
    // cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
  /*
    * If desired number of corner are detected,
    * we refine the pixel coordinates and display 
    * them on the images of checker board
  */
  if((successL) && (successR))
  {
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

    // refining pixel coordinates for given 2d points.
    cv::cornerSubPix(grayL,corner_ptsL,cv::Size(11,11), cv::Size(-1,-1),criteria);
    cv::cornerSubPix(grayR,corner_ptsR,cv::Size(11,11), cv::Size(-1,-1),criteria);

    // Displaying the detected corner points on the checker board
    cv::drawChessboardCorners(frameL_border, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL,successL);
    cv::drawChessboardCorners(frameR_border, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR,successR);

    objpoints.push_back(objp);
    imgpointsL.push_back(corner_ptsL);
    imgpointsR.push_back(corner_ptsR);
    std::cout << "Sucess" << std::endl;
  }

  cv::imshow("ImageL",frameL_border);
  cv::imshow("ImageR",frameR_border);
  //cv::waitKey(0);
  cv::destroyAllWindows();
}

cv::destroyAllWindows();

cv::Mat mtxL,distL,R_L,T_L;
cv::Mat mtxR,distR,R_R,T_R;
cv::Mat Rot, Trns, Emat, Fmat;
cv::Mat new_mtxL, new_mtxR;

// Calibrating left camera
cv::calibrateCamera(objpoints,
                    imgpointsL,
                    grayL.size(),
                    mtxL,
                    distL,
                    R_L,
                    T_L);

new_mtxL = cv::getOptimalNewCameraMatrix(mtxL,
                              distL,
                              grayL.size(),
                              1,
                              grayL.size(),
                              0);

// Calibrating right camera
cv::calibrateCamera(objpoints,
                    imgpointsR,
                    grayR.size(),
                    mtxR,
                    distR,
                    R_R,
                    T_R);

new_mtxR = cv::getOptimalNewCameraMatrix(mtxR,
                              distR,
                              grayR.size(),
                              1,
                              grayR.size(),
                              0);
                              
//Stereo calibrarion intrisic defined


int flag = 0;
flag |= cv::CALIB_FIX_INTRINSIC;

// This step is performed to transformation between the two cameras and calculate Essential and 
// Fundamenatl matrix
cv::stereoCalibrate(objpoints,
                    imgpointsL,
                    imgpointsR,
                    new_mtxL,
                    distL,
                    new_mtxR,
                    distR,
                    grayR.size(),
                    Rot,
                    Trns,
                    Emat,
                    Fmat,
                    flag,
                    cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 1e-6));
                    
//STEREO RETIFICATION

cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;

// Once we know the transformation between the two cameras we can perform 
// stereo rectification
cv::stereoRectify(new_mtxL,
                  distL,
                  new_mtxR,
                  distR,
                  grayR.size(),
                  Rot,
                  Trns,
                  rect_l,
                  rect_r,
                  proj_mat_l,
                  proj_mat_r,
                  Q,
                  1);

//COMPUTE MAP FOR STEREO RETIFICATION

cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

cv::initUndistortRectifyMap(new_mtxL,
                            distL,
                            rect_l,
                            proj_mat_l,
                            grayR.size(),
                            CV_16SC2,
                            Left_Stereo_Map1,
                            Left_Stereo_Map2);

cv::initUndistortRectifyMap(new_mtxR,
                            distR,
                            rect_r,
                            proj_mat_r,
                            grayR.size(),
                            CV_16SC2,
                            Right_Stereo_Map1,
                            Right_Stereo_Map2);

cv::FileStorage cv_file = cv::FileStorage("./Disparity_map/Depth_map/Vertical_Curve/verticalcurveborder_cam_stereo_params_3.xml", cv::FileStorage::WRITE);
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map1);
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map2);
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map1);
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map2); 

//Apply retifications
  cv::namedWindow("Left image before rectification",cv::WINDOW_NORMAL);
  cv::resizeWindow("Left image before rectification",960,536);
  cv::namedWindow("Right image before rectification",cv::WINDOW_NORMAL);
  cv::resizeWindow("Right image before rectification",960,536);
cv::imshow("Left image before rectification",frameL_border);
cv::imshow("Right image before rectification",frameR_border);

cv::Mat Left_nice, Right_nice;

// Apply the calculated maps for rectification and undistortion 
cv::remap(frameL_border,
          Left_nice,
          Left_Stereo_Map1,
          Left_Stereo_Map2,
          cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT,
          0);

cv::remap(frameR_border,
          Right_nice,
          Right_Stereo_Map1,
          Right_Stereo_Map2,
          cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT,
          0);

  cv::namedWindow("Left image after rectification",cv::WINDOW_NORMAL);
  cv::resizeWindow("Left image after rectification",960,536);
  cv::namedWindow("Right image after rectification",cv::WINDOW_NORMAL);
  cv::resizeWindow("Right image after rectification",960,536);
cv::imshow("Left image after rectification",Left_nice);
cv::imshow("Right image after rectification",Right_nice);

cv::waitKey(0);

//cv::Mat Left_nice_split[3], Right_nice_split[3];

//std::vector<cv::Mat> Anaglyph_channels;

//cv::split(Left_nice, Left_nice_split);
//cv::split(Right_nice, Right_nice_split);

//Anaglyph_channels.push_back(Right_nice_split[0]);
//Anaglyph_channels.push_back(Right_nice_split[1]);
//Anaglyph_channels.push_back(Left_nice_split[2]);

//cv::Mat Anaglyph_img;

//cv::merge(Anaglyph_channels, Anaglyph_img);

//cv::imshow("Anaglyph image", Anaglyph_img);
//cv::waitKey(0);
return 0;
}
