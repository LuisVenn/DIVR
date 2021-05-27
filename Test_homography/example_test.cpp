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
using namespace cv;
using namespace cv::detail;

cv::Size patternSize(6,9);
cv::Mat img1, img2, img1_crop, img2_crop, img1_crop_warp,img_matches, img2_vert_allig;

void left2rightplane(cv::Mat &img1, cv::Mat &img2, cv::Mat &img1_warp)
{
vector<Point2f> corners1, corners2;

bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);

//Get the homography matrix
cv::Mat H = cv::findHomography(corners1, corners2);
cv::Mat H_inv = H.inv();

std::cout << "H:\n" << H << std::endl;

//Find mask size for warping
vector<Point2f> original_corners, mask_corners;
original_corners.push_back(cv::Point2f(0,0));
original_corners.push_back(cv::Point2f(img1.cols-1,0));
original_corners.push_back(cv::Point2f(img1.cols-1,img1.rows-1));
original_corners.push_back(cv::Point2f(0,img1.rows-1));

cv::perspectiveTransform(original_corners, mask_corners, H);

cv::Size size_mask_max(0,0),size_mask_min(0,0), size_mask(0,0);

for(int i=0; i<mask_corners.size(); i++)
{
	//std::cout << "x:" << mask_corners[i].x << "y:" << mask_corners[i].y << std::endl;
	size_mask_min.height= min((int)mask_corners[i].y,(int)size_mask_min.height);
	size_mask_max.height= max((int)mask_corners[i].y,(int)size_mask_max.height);
	
	size_mask_min.width= min((int)mask_corners[i].x,(int)size_mask_min.width);
	size_mask_max.width= max((int)mask_corners[i].x,(int)size_mask_max.width);
}

size_mask.height = size_mask_max.height - size_mask_min.height;
size_mask.width = size_mask_max.width - size_mask_min.width;


cv::Mat T = (Mat_<double>(3,3) << 1, 0, -1*size_mask_min.width , 0, 1, -1* size_mask_min.height, 0, 0, 1);
cv:Mat Ht = T * H; 

//std::cout << "original_corners: " << original_corners << std::endl;
//std::cout << "Altura: " << size_mask.height << " Largura: " << size_mask.width << std::endl;    
//std::cout << "Minimo_height: " << size_mask_min.height << " Maximo_height: " << size_mask_max.height << std::endl;
//std::cout << "Minimo_width: " << size_mask_min.width << " Maximo_width: " << size_mask_max.width << std::endl;

//Warp image with mask size

cv::warpPerspective(img1, img1_warp, Ht, size_mask);

}

void saveimage(cv::Mat &img1_crop_warp)
{
	std::ostringstream name;
    int val;
    std::cout << "Val?" << std::endl;
    std::cin >> val;
    name << "/home/luis/Desktop/DIVR/Disparity_map/Depth_map/Vertical_Curve/StereoCalib_L_warp/" << val << ".jpg";
    cv::imwrite(name.str(), img1_crop_warp);
}

void match_features(cv::Mat &img1, cv::Mat &img2, cv::Mat &output)
{
	//Features dectection and matching
	Ptr<ORB> detector = ORB::create(200);
	std::vector<KeyPoint> keypoints1,keypoints2;
	Mat descriptors1,descriptors2;

	detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
	detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

	//Matching FEatures
	BFMatcher matcher(NORM_L2);
	//matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	std::vector<vector<DMatch> > matches;

	matcher.knnMatch(descriptors1, descriptors2, matches,2);

	std::vector<DMatch> match1;
	std::vector<DMatch> match2;

	for(int i=0; i<matches.size(); i++)
	{
		match1.push_back(matches[i][0]);
		match2.push_back(matches[i][1]);
	}

	cv::drawMatches(img1, keypoints1, img2, keypoints2, match1, output);
}

void vertically_allign(cv::Mat &img1, cv::Mat &img2, cv::Mat &output)
{
	vector<Point2f> corners1, corners2;
	
	bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
	bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);
	
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += corners1[i].y - corners2[i].y;
	}
	int avg = round(sum/corners1.size());
	
	//Create buffer
	int bottom = avg;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    cv::Scalar value(255,255,255);
  
	//Create Border
	copyMakeBorder( img2, output, no, bottom, no, no, borderType, value );
	
	//Translation
	cv::Mat T = (Mat_<double>(3,3) << 1, 0, 0 , 0, 1, avg, 0, 0, 1);
	cv::warpPerspective(output, output, T, output.size());
	  
}

cv::Mat horizontal_stitching(cv::Mat &img1, cv::Mat &img2)
{
	vector<Point2f> corners1, corners2;
	
	bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
	bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);
	
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += img1.cols - corners1[i].x + corners2[i].x;
		//std::cout << "1x: " << corners1[i].x << "2x: " << corners2[i].x << std::endl;
		//std::cout << "diffx: " << corners1[i].x - corners2[i].x << std::endl;
	}
	int avg = round(sum/corners1.size());
	std::cout << avg << std::endl;
	//int output_width = img1.cols + img2.cols - avg + 100; 
	//cv::Size output_size(img1.rows, output_width);
	cv::Mat output = img1.clone();
	
	//Create buffer
	int right = img2.cols - avg;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    cv::Scalar value(255,255,255);
  
	//Create Border
	copyMakeBorder( img1, output, no, no, no, right, borderType, value );
	
	img2.copyTo(output(cv::Rect(img1.cols-avg,0,img2.cols,img2.rows)));
	
	
	return output;
}

cv::Mat horizontal_line_stitching(cv::Mat &img1, cv::Mat &img2)
{
	vector<Point2f> corners1, corners2;
	
	bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
	bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);
	
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += img1.cols - corners1[i].x + corners2[i].x;
		//std::cout << "1x: " << corners1[i].x << "2x: " << corners2[i].x << std::endl;
		//std::cout << "diffx: " << corners1[i].x - corners2[i].x << std::endl;
	}
	int avg = round(sum/corners1.size());
	std::cout << avg << std::endl;
	//int output_width = img1.cols + img2.cols - avg + 100; 
	//cv::Size output_size(img1.rows, output_width);
	cv::Mat output = img1.clone();
	
	//Create buffer
	int right = img2.cols - avg;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    cv::Scalar value(255,255,255);
  
	//Create Border
	copyMakeBorder( img1, output, no, no, no, right, borderType, value );
	int gap = 30;
	int line = round((img2.rows/gap));
	int remainder = round((img2.rows % gap)); 	
	
	for(int i=0; i<line-1; i++)
	{
		//if(i == line-1)
		//{
			//gap = remainder;
		//}
		avg	+= 2;
		//avg = rand() % 50 + 150;
		img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(output(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)));
	}
	
	
	return output;
}

void draw_grid(cv::Mat &mat)
{
	// assume that mat.type=CV_8UC3

int dist=mat.size().height/30;

int height=mat.size().height;
int width= mat.size().width;
for(int i=0;i<height;i+=dist)
  cv::line(mat,Point(0,i),Point(width,i),cv::Scalar(255,255,255));

}
///////////////////////////////////////////////////////////////////////////////////////

int main()
{
    //Image sizes
	cv::namedWindow("Left image before rectification",cv::WINDOW_NORMAL);
	cv::resizeWindow("Left image before rectification",960,536);
	cv::namedWindow("Right image before rectification",cv::WINDOW_NORMAL);
	cv::resizeWindow("Right image before rectification",960,536);

	cv::namedWindow("Left image after rectification",cv::WINDOW_NORMAL);
	cv::resizeWindow("Left image after rectification",960,536);
	cv::namedWindow("Right image after rectification",cv::WINDOW_NORMAL);
	cv::resizeWindow("Right image after rectification",960,536);

	std::vector<cv::String> imagesL, imagesR;

	// Path of the folder containing checkerboard images
	std::string pathL = "/home/luis/Desktop/DIVR/Disparity_map/Depth_map/Vertical_Curve/StereoCalib_L/*.jpg"; //!!!!!!!!!!!!!!!!!!!!
	std::string pathR = "/home/luis/Desktop/DIVR/Disparity_map/Depth_map/Vertical_Curve/StereoCalib_R/*.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!

	cv::glob(pathL, imagesL);
	cv::glob(pathR, imagesR);

	for(int i{0}; i<imagesL.size(); i++)
	{
		std::cout << "novo par" << std::endl;
		std::cout << imagesL[i] << std::endl;
		std::cout << imagesR[i] << std::endl;
		img1 = cv::imread(imagesL[i]);
		img2 = cv::imread(imagesR[i]);
	   
		//Crop images for calibration
		int x = img1.cols/3;
		cv::Range cols(x*2, img1.cols);
		cv::Range rows(0, img1.rows);
		  
		cv::Range cols2(0, x);
		img1_crop = img1(rows, cols);
		img2_crop = img2(rows, cols2);
		
		//img1_crop = img1.clone();
		//img2_crop = img2.clone();

		left2rightplane(img1_crop,img2_crop, img1_crop_warp);
        std::cout << "feito" << std::endl;
        
        vertically_allign(img1_crop_warp, img2_crop, img2_vert_allig);
        
        match_features(img1_crop_warp, img2_vert_allig, img_matches);
        
        draw_grid(img_matches);
        
        //cv::Mat img_horz = horizontal_stitching(img1_crop_warp, img2_vert_allig);
        cv::Mat img_horz = horizontal_line_stitching(img1_crop_warp, img2_vert_allig);
        
		cv::imshow("Left image before rectification",img1_crop);
		cv::imshow("Right image before rectification",img2_crop);
		cv::imshow("Right image after rectification", img_horz);
		cv::imshow("Left image after rectification",img_matches);
		cv::waitKey();
	}



////With sample images
//cv::Mat img1_border_warp, img1_border_warp_features,img1_border_features;
////Read sample images
//img1 = cv::imread("sampleL.jpg");
//img2 = cv::imread("sampleR.jpg");

////img1_border = img1.clone();
////img2_border = img2.clone();

////Crop Image

//img1_border = img1(rows, cols);
//img2_border = img2(rows, cols2);

//////Creates Border
////copyMakeBorder( img1, img1_border, no, no, no, right, borderType, value );
////copyMakeBorder( img2, img2_border, no, no, left, no, borderType, value );

////Warp borded sample images with H matrix 
//cv::warpPerspective(img1_border, img1_border_warp, Ht, size_mask);

////Shows border image warped
//cv::imshow("Left image after rectification",img1_border_warp);
//cv::imshow("Right image after rectification",img2_border);
//cv::waitKey();

////Features dectection and matching
//Ptr<ORB> detector = ORB::create(200);
//std::vector<KeyPoint> keypoints1,keypoints2;
//Mat descriptors1,descriptors2;

//detector->detectAndCompute( img1_border_warp, noArray(), keypoints1, descriptors1 );
//detector->detectAndCompute( img2_border, noArray(), keypoints2, descriptors2 );

////Matching FEatures
//BFMatcher matcher(NORM_L2);
////matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
//std::vector<vector<DMatch> > matches;

//matcher.knnMatch(descriptors1, descriptors2, matches,2);

//std::vector<DMatch> match1;
//std::vector<DMatch> match2;

//for(int i=0; i<matches.size(); i++)
//{
    //match1.push_back(matches[i][0]);
    //match2.push_back(matches[i][1]);
//}

//cv::Mat img_matches1, img_matches2;
//cv::drawMatches(img1_border_warp, keypoints1, img2_border, keypoints2, match1, img_matches1);
//cv::drawMatches(img1_border_warp, keypoints1, img2_border, keypoints2, match2, img_matches2);

//cv::drawKeypoints(img1_border_warp, keypoints1, img1_border_warp_features,255);
	
//cv::imshow("Left image after rectification",img1_border_warp);
//cv::imshow("Right image after rectification",img1_border_warp_features);
//cv::waitKey();

////Get features coordinates
//vector<Point2f> coords;
//for(size_t i = 0; i < keypoints1.size(); ++i)
//{
	//coords.push_back(keypoints1[i].pt);
//}

////std::cout << "Coordenadas:" << coords << std::endl;

////Transform features to orinial images
//vector<Point2f> original_coords;
//cv::perspectiveTransform(coords, original_coords, H_inv);

////std::cout << "Coordenadas novas:" << original_coords << std::endl;
//img1_border_features = img1_border.clone();
////Plot features on original image
//for(size_t i = 0; i < original_coords.size(); ++i)
//{
	//cv::circle(img1_border_features, original_coords[i], 10, (255,0,0),5);
//}


////cv::drawKeypoints(img1, keypoints_real, img1,255);

//cv::imshow("Left image after rectification",img1_border_features);
//cv::imshow("Right image after rectification",img1_border_warp_features);
//cv::waitKey();

//cv::imshow("Left image after rectification",img1_border_warp);
//cv::imshow("Right image after rectification",img2_border);
//cv::waitKey();

//std::cout<< "Matches" << std::endl;
//cv::imshow("Left image after rectification",img_matches1);
//cv::imshow("Right image after rectification",img_matches2);
//cv::waitKey();


return 0;
}
