#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"

//GOLBAL VARIABLES


 
//FUNCTIONS

//Color map and color bar scale
void color_map(cv::Mat& input /*CV_32FC1*/, cv::Mat& dest, int color_map){

  int num_bar_w=100;
  int color_bar_w=30;
  int vline=10;

  cv::Mat win_mat(cv::Size(input.cols+num_bar_w+num_bar_w+vline, input.rows), CV_8UC3, cv::Scalar(255,255,255));

  //Input image to
  double Min, Max;
  cv::minMaxLoc(input, &Min, &Max);
  int max_int=ceil(Max);

  std::cout<<" Min "<< Min<<" Max "<< Max<<std::endl;

  input.convertTo(input,CV_8UC3,255.0/(Max-Min),-255.0*Min/(Max-Min));
  input.convertTo(input, CV_8UC3);

  cv::Mat M;
  cv::applyColorMap(input, M, color_map);

  M.copyTo(win_mat(cv::Rect(  0, 0, input.cols, input.rows)));

  //Scale
  cv::Mat num_window(cv::Size(num_bar_w, input.rows), CV_8UC3, cv::Scalar(255,255,255));
  for(int i=0; i<=8; i++){
      int j=i*input.rows/8;
      float value = (Max/8)*i;
      cv::putText(num_window, std::to_string(value), cv::Point(5, num_window.rows-j-5),cv::FONT_HERSHEY_SIMPLEX, 0.6 , cv::Scalar(0,0,0), 1 , 2 , false);
  }

  //color bar
  cv::Mat color_bar(cv::Size(color_bar_w, input.rows), CV_8UC3, cv::Scalar(255,255,255));
  cv::Mat cb;
  for(int i=0; i<color_bar.rows; i++){
    for(int j=0; j<color_bar_w; j++){
      int v=255-255*i/color_bar.rows;
      color_bar.at<cv::Vec3b>(i,j)=cv::Vec3b(v,v,v);
    }
  }

  color_bar.convertTo(color_bar, CV_8UC3);
  cv::applyColorMap(color_bar, cb, color_map);
  num_window.copyTo(win_mat(cv::Rect(input.cols+vline+color_bar_w, 0, num_bar_w, input.rows)));
  cb.copyTo(win_mat(cv::Rect(input.cols+vline, 0, color_bar_w, input.rows)));
  dest=win_mat.clone();
}

int main()
{
  // Initialize windows size
  cv::namedWindow("Image1",cv::WINDOW_NORMAL);
  cv::resizeWindow("Image1",960,536);
  cv::namedWindow("Image2",cv::WINDOW_NORMAL);
  cv::resizeWindow("Image2",960,536);
  cv::namedWindow("Out",cv::WINDOW_NORMAL);
  cv::resizeWindow("Out",960,536);
  
  // Initialize variables to store the maps for stereo rectification
  cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
  cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

  // Reading the mapping values for stereo image rectification
  cv::FileStorage cv_file2 = cv::FileStorage("./Disparity_map/Set_2/cam_stereo_params_2.xml", cv::FileStorage::READ); //!!!!!!
  cv_file2["Left_Stereo_Map_x"] >> Left_Stereo_Map1;
  cv_file2["Left_Stereo_Map_y"] >> Left_Stereo_Map2;
  cv_file2["Right_Stereo_Map_x"] >> Right_Stereo_Map1;
  cv_file2["Right_Stereo_Map_y"] >> Right_Stereo_Map2;
  cv_file2.release();

  // initialize values for StereoSGBM parameters
  int numDisparities, blockSize, minDisparity, speckleRange, speckleWindowSize, disp12MaxDiff, preFilterType, preFilterSize, preFilterCap, textureThreshold,uniquenessRatio;
  cv::Mat sol(2, 1, CV_32F);

  // Reading the disparity parameters
  cv::FileStorage cv_file3 = cv::FileStorage("./Disparity_map/Depth_map/depth_estimation_params_cpp.xml", cv::FileStorage::READ); //!!!!!!
  cv_file3["numDisparities"] >> numDisparities;
  cv_file3["preFilterType"] >> preFilterType;
  cv_file3["preFilterSize"] >> preFilterSize;
  cv_file3["preFilterCap"] >> preFilterCap;
  cv_file3["textureThreshold"] >> textureThreshold;
  cv_file3["uniquenessRatio"] >> uniquenessRatio;
  cv_file3["speckleRange"] >> speckleRange;
  cv_file3["speckleWindowSize"] >> speckleWindowSize;
  cv_file3["disp12MaxDiff"] >> disp12MaxDiff;
  cv_file3["minDisparity"] >> minDisparity;
  cv_file3["blockSize"] >> blockSize;
  cv_file3["sol"] >> sol;
  cv_file3.release();
  std::cout << "numDisparities:" << numDisparities << "\n";
  std::cout << "blocksize:" << blockSize<< "\n";
  // Creating an object of StereoSGBM algorithm
  cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
  
  //Set stereo parameters from file
  
  stereo->setNumDisparities(numDisparities);
  stereo->setBlockSize(blockSize*2+5);
  stereo->setPreFilterType(preFilterType);
  stereo->setPreFilterSize(preFilterSize*2+5);
  stereo->setPreFilterCap(preFilterCap);
  stereo->setTextureThreshold(textureThreshold);
  stereo->setUniquenessRatio(uniquenessRatio);
  stereo->setSpeckleRange(speckleRange);
  stereo->setSpeckleWindowSize(speckleWindowSize*2);
  stereo->setDisp12MaxDiff(disp12MaxDiff);
  stereo->setMinDisparity(minDisparity);


  //Set values for depth estimation
  float m,b;
  
  m = sol.at<float>(0,0);
  b = sol.at<float>(1,0);
  
  //Read images
  cv::Mat imgL;
  cv::Mat imgR;
  cv::Mat imgL_gray;
  cv::Mat imgR_gray;
  cv::Mat disp, disparity;
  
  imgL = cv::imread("./Disparity_map/Depth_map/Depth_map_01/100.jpg");
  std::cout << "deu L\n";
  cv::imshow("Image1",imgL);

  imgR = cv::imread("./Disparity_map/Depth_map/Depth_map_03/100.jpg");
  std::cout << "Deu R\n";
  cv::imshow("Image2",imgR);
  
  // Converting images to grayscale
  cv::cvtColor(imgL, imgL_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(imgR, imgR_gray, cv::COLOR_BGR2GRAY);

  // Initialize matrix for rectified stereo images
  cv::Mat Left_nice, Right_nice;
  // Applying stereo image rectification on the left image
  cv::remap(imgL_gray,
            Left_nice,
            Left_Stereo_Map1,
            Left_Stereo_Map2,
            cv::INTER_LANCZOS4,
            cv::BORDER_CONSTANT,
            0);

  // Applying stereo image rectification on the right image
  cv::remap(imgR_gray,
            Right_nice,
            Right_Stereo_Map1,
            Right_Stereo_Map2,
            cv::INTER_LANCZOS4,
            cv::BORDER_CONSTANT,
            0);
   
 // Calculating disparith using the StereoBM algorithm
 stereo->compute(Left_nice,Right_nice,disp);
 disp.convertTo(disparity,CV_32F, 1.0);
 
 // Scaling down the disparity values and normalizing them 
 disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

 // Displaying the colour disparity map with scale
   
 cv::Mat out;
 color_map(disparity, out, cv::COLORMAP_JET);
 cv::imshow("Out",out);
 cv::waitKey(0);
 return 0;

}
