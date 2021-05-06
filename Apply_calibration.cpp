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




int main()
{
	//READ XML FILE
	
    
    cv::FileStorage fs("improved_params2_cpp.xml", cv::FileStorage::READ);
    
	cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
	cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;

	fs["Left_Stereo_Map_x"] >> Left_Stereo_Map1;
	fs["Left_Stereo_Map_y"] >> Left_Stereo_Map2;
	fs["Right_Stereo_Map_x"] >> Right_Stereo_Map1;
	fs["Right_Stereo_Map_y"] >> Right_Stereo_Map2; 
	
	fs.release();
	
	/////////////////////////////7
cv::Mat frameL,frameR;
  
  frameL = cv::imread("./Disparity_map/left_eye3.jpg");
  //cv::resize(imgL,imgL,cv::Size(600,600));
  frameR = cv::imread("./Disparity_map/right_eye3.jpg");
  //cv::resize(imgR,imgR,cv::Size(600,600));
  
cv::imshow("Left image before rectification",frameL);
cv::imshow("Right image before rectification",frameR);


cv::Mat Left_nice, Right_nice;

// Apply the calculated maps for rectification and undistortion 
cv::remap(frameL,
          Left_nice,
          Left_Stereo_Map1,
          Left_Stereo_Map2,
          cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT,
          0);

cv::remap(frameR,
          Right_nice,
          Right_Stereo_Map1,
          Right_Stereo_Map2,
          cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT,
          0);


cv::imshow("Left image after rectification",Left_nice);
cv::imshow("Right image after rectification",Right_nice);
cv::imwrite("/home/luis/Desktop/DIVR/Disparity_map/left_eye_nice.jpg", Left_nice);
cv::imwrite("/home/luis/Desktop/DIVR/Disparity_map/right_eye_nice.jpg", Right_nice);
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
cv::waitKey(0);

}
