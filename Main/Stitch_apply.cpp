
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

#include <string>

#include <fstream>
using namespace std;
using namespace cv;
using namespace cv::detail;

cv::Size patternSize(6,9);

struct blob {
  int top;
  int bot;
  int d;
};

struct Matches_coordinates {
  vector<Point2f> L;
  vector<Point2f> R;
};

bool organiza (DMatch p1, DMatch p2) 
{ 
	return p1.distance < p2.distance; 
}

bool organizablob (blob p1, blob p2) 
{ 
	return p1.top < p2.top; 
}

void vertically_allign_apply(cv::Mat &img1, cv::Mat &img2, int avg)
{
//Create buffer
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    cv::Scalar value(255,255,255);
  
	//Create Border
	if (avg > 0)
		copyMakeBorder( img2, img2, avg, no, no, no, borderType, value );
	else if (avg < 0)
		copyMakeBorder( img1, img1, -avg, no, no, no, borderType, value );
}

//Get mask of new objects on image, used to feature detection and object size estimation for line stitching
void getMask(cv::Mat frame1, cv::Mat frame2, cv::Mat &mask)
{
	cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(frame2, frame2, cv::COLOR_BGR2GRAY);
	
	//create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    
    pBackSub = createBackgroundSubtractorMOG2(500,200,true);
    //pBackSub = createBackgroundSubtractorKNN();
    
    //update the background model
    cv::Mat fgMask;
    
    pBackSub->apply(frame1, fgMask);
    pBackSub->apply(frame2, fgMask);
    
    //show the current frame and the fg masks
    cv::Mat diffrence = frame1 - frame2;
 
    mask = cv::Mat::zeros(diffrence.size(), CV_8U);
    
    mask.setTo(255, diffrence > 25);

    //Morphologic operations
    cv::Mat element = getStructuringElement(MORPH_RECT,Size(20,20),Point(9,9));
    cv::morphologyEx(mask,mask,MORPH_OPEN,element);
    cv::morphologyEx(mask,mask,MORPH_CLOSE,element);
    cv::morphologyEx(mask,mask,MORPH_DILATE,element);
    
}

void getMatches(cv::Mat img1, cv::Mat img2, cv::Mat mask1, cv::Mat mask2, Matches_coordinates &matches_coords)
{
	//Features dectection and matching
	Ptr<ORB> detector = cv::ORB::create();
	std::vector<KeyPoint> keypoints1,keypoints2;
	Mat descriptors1,descriptors2;

	detector->detectAndCompute( img1, mask1, keypoints1, descriptors1 );
	detector->detectAndCompute( img2, mask2, keypoints2, descriptors2 );
	
	//Matching Features
	BFMatcher matcher(NORM_HAMMING);
	
	std::vector<vector<DMatch> > matches;

	matcher.knnMatch(descriptors1, descriptors2, matches,2);

	std::vector<DMatch> match1;
	std::vector<DMatch> match2;

    std::vector<DMatch> sorted_matches;
    
    //get the best match
    for (size_t i = 0; i < matches.size(); i++)
    {
		 sorted_matches.push_back(matches[i][0]);
    }
	
	sort(sorted_matches.begin(),sorted_matches.end(),organiza);
	
   	std::cout << "--------------------" << std::endl;
   	std::cout << "----Good matches----" << std::endl;
   	std::cout << "--------------------" << std::endl;
   	
   	for (size_t i = 0; i < sorted_matches.size(); i++)
    {
		if (sorted_matches[i].distance < 45)
		{
			matches_coords.L.push_back(keypoints1[sorted_matches[i].queryIdx].pt);
			matches_coords.R.push_back(keypoints2[sorted_matches[i].trainIdx].pt);
			std::cout << "Keypoint 1: " << keypoints1[sorted_matches[i].queryIdx].pt << std::endl;
			std::cout << "Keypoint 2: " << keypoints2[sorted_matches[i].trainIdx].pt << std::endl;
			std::cout << "Distance: " << sorted_matches[i].distance << std::endl;
			std::cout << "--------------------" << std::endl;
		}
    }
}

void getBlobs(cv::Mat mask1, cv::Mat mask2, Matches_coordinates matches_coords, vector<blob> &blobs)
{
   //In wich blobs are this features
   
	cv::Mat stat,centroid,mask3; //para que preciso da mask3??
	int nLabels = connectedComponentsWithStats(mask1, mask3, stat,centroid,8, CV_16U);
	int w = mask1.cols;
	blob blob_buff;
	vector<bool> block(nLabels,1);
	
	for (size_t imatch = 0; imatch < matches_coords.L.size(); imatch++)
    {
		cv::Point pt1 = matches_coords.L[imatch];
		cv::Point pt2 = matches_coords.R[imatch];

		for (int i=1;i<nLabels;i++)
		{
			if(block[i] && (pt1.x >= stat.at<int>(i,CC_STAT_LEFT)) && (pt1.x <= (stat.at<int>(i,CC_STAT_LEFT) + stat.at<int>(i,CC_STAT_WIDTH)) && (pt1.y >= stat.at<int>(i,CC_STAT_TOP)) && (pt1.y <= (stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT)))))
			{
				blob_buff.top = stat.at<int>(i,CC_STAT_TOP);
				blob_buff.bot = stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT);
				blob_buff.d = ((w - pt1.x) + pt2.x);
				blobs.push_back(blob_buff);
				std::cout << i << " blob top: " << blob_buff.top << std::endl;
				std::cout << i << " blob bot: " << blob_buff.bot << std::endl;
				std::cout << i << " d: " << blob_buff.d << std::endl;
				block[i] = 0;
				
				break;
			}	 
		}	
	}
	sort(blobs.begin(),blobs.end(),organizablob);
	std::cout << "number of good features found in blobs: " << blobs.size() << std::endl;
}

cv::Mat horizontal_line_stitching_apply(cv::Mat &img1, cv::Mat &img2, int avg, cv::Mat mask1, cv::Mat mask2, Matches_coordinates matches_coords, vector<blob> blobs)
{
	cv::Mat output = img1.clone();
	
    int  dMin = avg + 1;
    if (blobs.size() > 0)
    {
		auto it = std::min_element(blobs.begin(), blobs.end(), [](const blob& a,const blob& b) { return a.d < b.d; });
		dMin = it[0].d; 
		std::cout << "dmin: " << dMin << std::endl;
    }
  
	//Create buffer
	int right;
	if (dMin > avg)
		right = img2.cols - avg;
	else
		right = img2.cols - dMin;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    int buff = 0;
    cv::Scalar value(255,255,255);
	if(img2.rows >img1.rows) buff = img2.rows-img1.rows;
	
	//Create Border
	copyMakeBorder( img1, output, no, buff, no, right, borderType, value );
	
	//Create Mask to clean black part
	cv::Mat mask = cv::Mat::zeros(img2.size(), CV_8U);
	cv::Mat mask_output = cv::Mat::zeros(output.size(), CV_8U);
	
	cv::Mat img2_gray, output_gray;
	
	cv::cvtColor( img2, img2_gray, cv::COLOR_RGB2GRAY );
	cv::cvtColor( output, output_gray, cv::COLOR_RGB2GRAY);
	
	mask.setTo(255, img2_gray > 0);
    mask_output.setTo(255, output_gray > 0);
    mask_output.setTo(0,output_gray == 255);
    cv::Mat outputBuff = output.clone();
    
    outputBuff.setTo(cv::Scalar(255,255,255));
    output.copyTo(outputBuff,mask_output);
    
	int gap = 1;
	int line = round((img2.rows/gap));
	std::cout << line << std::endl;
	int remainder = round((img2.rows % gap)); 	

	int top = 0,bot,d;
	int i = 0;
	
	if(blobs.size() == 0)
	{
		for(i; i<img2.rows; i++)
		{
			img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
		}
	}else{	
		
		for(int i2=0;i2 < blobs.size();i2++)
		{	
			if(i < blobs[i2].top )
			{
				for(i; i<blobs[i2].top; i++)
				{
					img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					
				}	
			
				for(i; i<blobs[i2].bot; i++)
				{
			
					img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-blobs[i2].d,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					
				}
					
			}
			if(i2 == blobs.size()-1)
			{
				for(i; i<img2.rows; i++)
				{
					img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
				}
			}
			
		}
	}
	return outputBuff;
}	

int main() 
{
	//READ IMAGES
	cv::Mat imgL, imgR, imgR_warp, imgL_warp;
	cv::Mat imgL_bg, imgR_bg, imgR_bg_warp, imgL_bg_warp;
	
	//frames
	imgL = cv::imread("../Images/squarefloor2/L1_squarefloor2_box2.jpg");
	imgR = cv::imread("../Images/squarefloor2/R1_squarefloor2_box2.jpg");
	//background
	imgL_bg = cv::imread("../Images/squarefloor2/L1_squarefloor2_bgbox2.jpg");
	imgR_bg = cv::imread("../Images/squarefloor2/R1_squarefloor2_bgbox2.jpg");
	
	//READ PRE-CALIBRATED PARAMETERS
	cv::Mat Ht_L, Ht_R;
	cv::Size warpedL_size, warpedR_size;	
	int warpedL_height, warpedL_width , warpedR_height, warpedR_width, avg_v, avg_h;;
	
	cv::FileStorage fs("./Homography_params_squarefloor2.xml", cv::FileStorage::READ);

	fs["Ht_L"] >> Ht_L;
	fs["Ht_R"] >> Ht_R;
	fs["warpedL_height"] >> warpedL_height;
	fs["warpedL_width"] >> warpedL_width; 
	fs["warpedR_height"] >> warpedR_height;
	fs["warpedR_width"] >> warpedR_width; 
	fs["avg_v"] >> avg_v;
	fs["avg_h"] >> avg_h; 
	
	warpedL_size.height = warpedL_height;
    warpedL_size.width  = warpedL_width ;
    warpedR_size.height = warpedR_height;
    warpedR_size.width  = warpedR_width ;
	
	//WARP FRAMES AND BACKGROUND
		
	cv::warpPerspective(imgL, imgL_warp, Ht_L, warpedL_size);
	cv::warpPerspective(imgR, imgR_warp, Ht_R, warpedR_size);
	cv::warpPerspective(imgL_bg, imgL_bg_warp, Ht_L, warpedL_size);
	cv::warpPerspective(imgR_bg, imgR_bg_warp, Ht_R, warpedR_size);
	
	//Allign features vertically
    vertically_allign_apply(imgL_warp, imgR_warp, avg_v);
    vertically_allign_apply(imgL_bg_warp, imgR_bg_warp, avg_v);
	
	//GET MASKS
	cv::Mat maskL_warp, maskR_warp;
			
	getMask(imgL_bg_warp, imgL_warp, maskL_warp);
	getMask(imgR_bg_warp, imgR_warp, maskR_warp);
	
	//DETECT FEATURES
	Matches_coordinates matches_coords;
			
	getMatches(imgL_warp, imgR_warp, maskL_warp, maskR_warp, matches_coords);
	
	//DIVIDE BLOBS
	vector<blob> blobs;
	getBlobs(maskL_warp, maskR_warp, matches_coords, blobs);
	
	//LINE STITCHING
	cv::Mat result;
	
	result = horizontal_line_stitching_apply(imgL_warp, imgR_warp, avg_h, maskL_warp, maskR_warp, matches_coords, blobs);
	
	//SHOW RESULT
	cv::namedWindow("Result",cv::WINDOW_NORMAL);
	cv::resizeWindow("Result",960,536);
	cv::imshow("Result", result);
	cv::waitKey();
			
}
