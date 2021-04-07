#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(){

  cv::VideoCapture cap("Cam01_7abr2.h264"); 
  cv::VideoCapture cap2("Cam02_7abr2.h264"); 
  cv::VideoCapture cap3("Cam03_7abr2.h264");  
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  Mat frame,frame2,frame3;
 
	while(frame.empty())
	{
		cap >> frame;
	}
	while(frame2.empty())
	{
		cap2 >> frame2;
	}
	while(frame3.empty())
	{
		cap3 >> frame3;
	}
	
      
	imshow( "Frame", frame );
	waitKey();
	imshow( "Frame", frame2 );
	waitKey();
	imshow( "Frame", frame3 );
	waitKey();
	
	cv::imwrite("/home/luis/Desktop/DIVR/1st_frame_01.jpg", frame);
	cv::imwrite("/home/luis/Desktop/DIVR/1st_frame_02.jpg", frame2);
    cv::imwrite("/home/luis/Desktop/DIVR/1st_frame_03.jpg", frame3);
 
  // When everything done, release the video capture object
  cap.release();
  cap2.release();
  cap3.release();
  destroyAllWindows();
	
  return 0;
}
