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
	cv::Mat Left_nice, Right_nice;
	Left_nice = cv::imread("./Disparity_map/Set_2/calib_2cam_01_2/Test/left_eye_nice.jpg");
	Right_nice = cv::imread("./Disparity_map/Set_2/calib_2cam_03_2/Test/right_eye_nice.jpg");
cv::Mat Left_nice_split[3], Right_nice_split[3];

std::vector<cv::Mat> Anaglyph_channels;

cv::split(Left_nice, Left_nice_split);
cv::split(Right_nice, Right_nice_split);

Anaglyph_channels.push_back(Right_nice_split[0]);
Anaglyph_channels.push_back(Right_nice_split[1]);
Anaglyph_channels.push_back(Left_nice_split[2]);

cv::Mat Anaglyph_img;

cv::merge(Anaglyph_channels, Anaglyph_img);
cv::imwrite("/home/luis/Desktop/DIVR/Disparity_map/Anaglyph_example.jpg", Anaglyph_img);
cv::imshow("Anaglyph image", Anaglyph_img);
cv::waitKey(0);
}
