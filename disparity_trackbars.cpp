#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgcodecs.hpp"

void color_map(cv::Mat& input /*CV_32FC1*/, cv::Mat& dest, int color_map){

  int num_bar_w=30;
  int color_bar_w=10;
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
  for(int i=0; i<=7; i++){
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

// initialize values for StereoSGBM parameters
int numDisparities = 13;
int blockSize = 7;
int preFilterType = 1;
int preFilterSize = 1;
int preFilterCap = 31;
int minDisparity = 0;
int textureThreshold = 10;
int uniquenessRatio = 15;
int speckleRange = 0;
int speckleWindowSize = 0;
int disp12MaxDiff = -1;
int dispType = CV_16S;


// Creating an object of StereoSGBM algorithm
cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();

cv::Mat imgL;
cv::Mat imgR;
cv::Mat imgL_gray;
cv::Mat imgR_gray;
 cv::Mat disp, disparity;
  
// Calc matrix M
std::vector<float> z_vec;
std::vector<cv::Point2f> coeff_vec;

// These parameters can vary according to the setup
// Keeping the target object at max_dist we store disparity values
// after every sample_delta distance.
//int max_dist = 300; // max distance to keep the target object (in cm)
//int min_dist = 50; // Minimum distance the stereo setup can measure (in cm)
int sample_delta = 50; // Distance between two sampling points (in cm)
int Z = 50;

// Defining callback functions for mouse events
void mouseEvent(int evt, int x, int y, int flags, void* param) {                    
    float depth_val;

    if (evt == cv::EVENT_LBUTTONDOWN) {
      depth_val  = disparity.at<float>(y,x);

      if (depth_val > 0)
      {
        z_vec.push_back(Z);
        coeff_vec.push_back(cv::Point2f(1.0f/(float)depth_val, 1.0f));
        std::cout << "Registei um ponto com Z =" << Z << "com disparidade =" << depth_val << "\n";  
        
      }
    }  
}

// Defining callback functions for the trackbars to update parameter values

static void on_trackbar1( int, void* )
{
  stereo->setNumDisparities(numDisparities*16);
  numDisparities = numDisparities*16;
}

static void on_trackbar2( int, void* )
{
  stereo->setBlockSize(blockSize*2+5);
  blockSize = blockSize*2+5;
}

static void on_trackbar3( int, void* )
{
  stereo->setPreFilterType(preFilterType);
}

static void on_trackbar4( int, void* )
{
  stereo->setPreFilterSize(preFilterSize*2+5);
  preFilterSize = preFilterSize*2+5;
}

static void on_trackbar5( int, void* )
{
  stereo->setPreFilterCap(preFilterCap);
}

static void on_trackbar6( int, void* )
{
  stereo->setTextureThreshold(textureThreshold);
}

static void on_trackbar7( int, void* )
{
  stereo->setUniquenessRatio(uniquenessRatio);
}

static void on_trackbar8( int, void* )
{
  stereo->setSpeckleRange(speckleRange);
}

static void on_trackbar9( int, void* )
{
  stereo->setSpeckleWindowSize(speckleWindowSize*2);
  speckleWindowSize = speckleWindowSize*2;
}

static void on_trackbar10( int, void* )
{
  stereo->setDisp12MaxDiff(disp12MaxDiff);
}

static void on_trackbar11( int, void* )
{
  stereo->setMinDisparity(minDisparity);
}


int main()
{
  // Initialize windows size
  cv::namedWindow("Image",cv::WINDOW_NORMAL);
  cv::resizeWindow("Image",960,536);
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

  // Check for left and right camera IDs
  // These values can change depending on the system
  //int CamL_id{2}; // Camera ID for left camera
  //int CamR_id{0}; // Camera ID for right camera

  //cv::VideoCapture camL(CamL_id), camR(CamR_id);

  //// Check if left camera is attached
  //if (!camL.isOpened())
  //{
    //std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    //return -1;
  //}

  //// Check if right camera is attached
  //if (!camL.isOpened())
  //{
    //std::cout << "Could not open camera with index : " << CamL_id << std::endl;
    //return -1;
  //}

  // Creating a named window to be linked to the trackbars
  cv::namedWindow("disparity",cv::WINDOW_NORMAL);
  cv::resizeWindow("disparity",600,600);

  // Creating trackbars to dynamically update the StereoBM parameters
  cv::createTrackbar("numDisparities", "disparity", &numDisparities, 30, on_trackbar1);
  cv::createTrackbar("blockSize", "disparity", &blockSize, 50, on_trackbar2);
  cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
  cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
  cv::createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
  cv::createTrackbar("textureThreshold", "disparity", &textureThreshold, 100, on_trackbar6);
  cv::createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar7);
  cv::createTrackbar("speckleRange", "disparity", &speckleRange, 100, on_trackbar8);
  cv::createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 25, on_trackbar9);
  cv::createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar10);
  cv::createTrackbar("minDisparity", "disparity", &minDisparity, 25, on_trackbar11);


  cv::Mat frameL,frameR;
  
  imgL = cv::imread("./Disparity_map/Set_2/calib_2cam_01_2/Test/1_left_eye.jpg"); //!!!!!!!!!!!!!!!!!!!!!!!
  //imgL = cv::imread("./Disparity_map/Depth_map/Depth_map_01/100.jpg");
  std::cout << "deu L\n";
  cv::imshow("Image1",imgL);
  imgR = cv::imread("./Disparity_map/Set_2/calib_2cam_03_2/Test/1_right_eye.jpg"); //!!!!!!!!!!!!!!!!!!!!!!!!
  //imgR = cv::imread("./Disparity_map/Depth_map/Depth_map_03/100.jpg");
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
   
    //cv::imwrite("./Disparity_map/Set_2/calib_2cam_01_2/Test/1_left_eye_stereo_nice.jpg", Left_nice);
    //cv::imwrite("./Disparity_map/Set_2/calib_2cam_03_2/Test/1_right_eye_stereo_nice.jpg", Right_nice);
 

 while(true)
  { 

    // Calculating disparith using the StereoBM algorithm
    stereo->compute(Left_nice,Right_nice,disp);

    // NOTE: Code returns a 16bit signed single channel image,
		// CV_16S containing a disparity map scaled by 16. Hence it 
    // is essential to convert it to CV_32F and scale it down 16 times.

    // Converting disparity values to CV_32F from CV_16S
 
    disp.convertTo(disparity,CV_32F, 1.0);

    // Scaling down the disparity values and normalizing them 
    std::cout << "numDisparities:" << numDisparities << "\n";
    std::cout << "minDIsparity:" << minDisparity << "\n";
    double Min2, Max2;
	cv::minMaxLoc(disparity, &Min2, &Max2);
    std::cout << "Max pre norm:" << Max2 << "\n";
    std::cout << "Min pre norm:" << Min2 << "\n";
    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);
    cv::minMaxLoc(disparity, &Min2, &Max2);
	std::cout << "Max pos norm:" << Max2 << "\n";
    std::cout << "Min pos norm:" << Min2 << "\n";

    // Displaying the disparity map
    cv::imshow("disparity",disparity);
    cv::imshow("Image",disparity);
    cv::Mat out;
	color_map(disparity, out, cv::COLORMAP_JET);
	cv::imshow("Out",out);
	std::cout << "A espera\n";
    // Close window using esc key
    if (cv::waitKey(0) == 102) cv::imwrite("./Disparity_map/Set_2/disparity_map.jpg", disparity);
    if (cv::waitKey(0) == 27) break;
    
  }
  
  //Read depth calibration files
  
  std::string folderL = "./Disparity_map/Depth_map/Depth_map_01/";
  std::string folderR = "./Disparity_map/Depth_map/Depth_map_03/";
  std::string suffix = ".jpg";
  
  // Inicialization of disparity window
  cv::destroyAllWindows();
  cv::namedWindow("disparity",cv::WINDOW_NORMAL);
  cv::resizeWindow("disparity",600,600);
  cv::setMouseCallback("disparity", mouseEvent, NULL);
  
  for(Z; Z<=250; Z+=50)
  {
	std::cout << "Z value:" << Z << "\n";
	std::stringstream ss;
	ss << Z;
    
	std::string number = ss.str();
    std::string nameL = folderL + number + suffix;
    std::string nameR = folderR + number + suffix;
    
	imgL = cv::imread(nameL);
	cv::imshow("Image1",imgL);
	imgR = cv::imread(nameR);
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

    // NOTE: Code returns a 16bit signed single channel image,
		// CV_16S containing a disparity map scaled by 16. Hence it 
    // is essential to convert it to CV_32F and scale it down 16 times.

    // Converting disparity values to CV_32F from CV_16S
 
    disp.convertTo(disparity,CV_32F, 1.0);

    // Scaling down the disparity values and normalizing them 

    disparity = (disparity/16.0f - (float)minDisparity)/((float)numDisparities);

    // Displaying the disparity map
    cv::imshow("disparity",disparity);
    cv::waitKey(0);
	std::cout << "um ciclo \n";
    
  }
  std::cout << "aqui nao 0 \n";
  std::cout << "z_vec size: " << z_vec.size()<< "\n";
  std::cout << "z_vec size: " << z_vec.data() << "\n";
  for(int i{0}; i < z_vec.size();i++)
  {
	  std::cout << "z_vec[" << i << "] = " << z_vec[i] << "\n";
	  std::cout << "coeff_vec[" << i << "] = " << coeff_vec[i] << "\n";
  }
  cv::Mat Z_mat(z_vec.size(), 1, CV_32F, z_vec.data());
  cv::Mat coeff(z_vec.size(), 2, CV_32F, coeff_vec.data());

  cv::Mat sol(2, 1, CV_32F);
  float M;
  float B;
	std::cout << "aqui nao\n";
  // Solving for M using least square fitting with QR decomposition method 
  cv::solve(coeff, Z_mat, sol, cv::DECOMP_QR);
	std::cout << "aqui nao 2\n";
  M = sol.at<float>(0,0);
  B = sol.at<float>(1,0);
	std::cout << "aqui nao 3\n";
	std::cout << "M : " << M <<"\n";
	std::cout << "B : " << B;
   //Storing the updated value of M along with the stereo parameters
  cv::FileStorage cv_file3 = cv::FileStorage("./Disparity_map/Depth_map/depth_estimation_params_cpp.xml", cv::FileStorage::WRITE);
  cv_file3.write("numDisparities",numDisparities);
  cv_file3.write("blockSize",blockSize);
  cv_file3.write("preFilterType",preFilterType);
  cv_file3.write("preFilterSize",preFilterSize);
  cv_file3.write("preFilterCap",preFilterCap);
  cv_file3.write("textureThreshold",textureThreshold);
  cv_file3.write("uniquenessRatio",uniquenessRatio);
  cv_file3.write("speckleRange",speckleRange);
  cv_file3.write("speckleWindowSize",speckleWindowSize);
  cv_file3.write("disp12MaxDiff",disp12MaxDiff);
  cv_file3.write("minDisparity",minDisparity);
  cv_file3.write("sol",sol);
  cv_file3.release();
  return 0;
}
