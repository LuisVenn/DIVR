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

//Estruturas
struct blob {
  int top;
  int bot;
  int d;
};

struct Matches_coordinates {
  vector<Point2f> L;
  vector<Point2f> R;
};

//Vector algebra functions
bool organiza (DMatch p1, DMatch p2) 
{ 
	return p1.distance < p2.distance; 
}

bool organizablob (blob p1, blob p2) 
{ 
	return p1.top < p2.top; 
}

void multVector(vector<Point2f> &v, float k){
     for(int i=0; i<v.size(); i++)
     {
		v[i].x = v[i].x*k ;
		v[i].y = v[i].y*k ;
	 }
  }
  
 void diffVectors(vector<Point2f> &a, vector<Point2f> &b){
	 
     for(int i=0; i<a.size(); i++)
     {
		a[i].y = a[i].y - b[i].y;
		a[i].x = a[i].x - b[i].x;
	 }
 }
 
 void sumVectors(vector<Point2f> &a, vector<Point2f> &b){
	 
     for(int i=0; i<a.size(); i++)
     {
		a[i].y = a[i].y + b[i].y;
		a[i].x = a[i].x + b[i].x;
	 }
 }
 
//Function to estimate the homography matrix and return size
cv::Size getTransform(cv::Mat &img1, vector<Point2f> &corners1, vector<Point2f> &corners2, cv::Mat &Ht)
{

	//Get the homography matrix
	cv::Mat H = cv::findHomography(corners1, corners2);

	//Find mask size for warping
	vector<Point2f> original_corners, mask_corners;
	original_corners.push_back(cv::Point2f(0,0));
	original_corners.push_back(cv::Point2f(img1.cols-1,0));
	original_corners.push_back(cv::Point2f(img1.cols-1,img1.rows-1));
	original_corners.push_back(cv::Point2f(0,img1.rows-1));

	cv::perspectiveTransform(original_corners, mask_corners, H);

	cv::Size size_mask_max(-1000,-1000),size_mask_min(1000,1000), size_mask(0,0);

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
	Ht = T * H; 

	return size_mask;
}

void estimateTransform(cv::Mat imgL,cv::Mat imgR, vector<Point2f> &cornersL, vector<Point2f> &cornersR, cv::Mat &Ht_L, cv::Mat &Ht_R, cv::Size &warpedL_size, cv::Size &warpedR_size, float k)
{
	
	vector<Point2f> cornersC_estimated = cornersL;
	//Estimate by assuming linear the corners of C
	
	diffVectors(cornersC_estimated, cornersR);
	multVector(cornersC_estimated,k);
	sumVectors(cornersC_estimated,cornersR);
	std::cout << cornersC_estimated << std::endl;
	warpedL_size = getTransform(imgL, cornersL, cornersC_estimated, Ht_L);
	warpedR_size = getTransform(imgR, cornersR, cornersC_estimated, Ht_R);
}

void saveimage(cv::Mat &img1)
{
	std::ostringstream name;
    string val;
    cv::namedWindow("Save image",cv::WINDOW_NORMAL);
	cv::resizeWindow("Save image",960,536);
    cv::imshow("Save image",img1);
    std::cout << "Image name?" << std::endl;
    std::cin >> val;
    name << "/home/luis/Desktop/DIVR/Images/" << val << ".jpg";
    cv::imwrite(name.str(), img1);
}

void match_features2(cv::Mat &img1, cv::Mat &img2, cv::Mat &output)
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
	
	//get the best match
	std::vector<DMatch> good_matches, sorted_matches;
	
    for (size_t i = 0; i < matches.size(); i++)
    {
		 sorted_matches.push_back(matches[i][0]);
    }
	
	sort(sorted_matches.begin(),sorted_matches.end(),organiza);
   	
   	for (size_t i = 0; i < sorted_matches.size(); i++)
    {
		if (sorted_matches[i].distance < 45)
		{
			good_matches.push_back(sorted_matches[i]);
		}
    }
    
	cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output);
}
void getMatches(cv::Mat img1, cv::Mat img2, cv::Mat mask1, cv::Mat mask2, Matches_coordinates &matches_coords)
{
	//Features dectection and matching
	Ptr<ORB> detector = cv::ORB::create();
	std::vector<KeyPoint> keypoints1,keypoints2;
	Mat descriptors1,descriptors2;

	detector->detectAndCompute( img1, mask1, keypoints1, descriptors1 );
	detector->detectAndCompute( img2, mask2, keypoints2, descriptors2 );
	
	//cv::namedWindow("Left mask",cv::WINDOW_NORMAL);
	//cv::resizeWindow("Left mask",960,536);
	//cv::imshow("Left mask", mask1);
	//cv::namedWindow("Right mask",cv::WINDOW_NORMAL);
	//cv::resizeWindow("Right mask",960,536);
	//cv::imshow("Right mask", mask2);
	//cv::waitKey();
	
	//Matching Features
	BFMatcher matcher(NORM_HAMMING);
	//matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	std::vector<vector<DMatch> > matches;

	matcher.knnMatch(descriptors1, descriptors2, matches,2);

	std::vector<DMatch> match1;
	std::vector<DMatch> match2;
	int i2 = 0;
	
	//-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> sorted_matches;
    std::vector<DMatch> query;
    std::vector<DMatch> train;
    
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
			//good_matches.push_back(sorted_matches[i]);
			matches_coords.L.push_back(keypoints1[sorted_matches[i].queryIdx].pt);
			matches_coords.R.push_back(keypoints2[sorted_matches[i].trainIdx].pt);
			std::cout << "Keypoint 1: " << keypoints1[sorted_matches[i].queryIdx].pt << std::endl;
			std::cout << "Keypoint 2: " << keypoints2[sorted_matches[i].trainIdx].pt << std::endl;
			std::cout << "Distance: " << sorted_matches[i].distance << std::endl;
			std::cout << "--------------------" << std::endl;
		}
    }
    //cv::Mat imgmatches;
    //cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, imgmatches);
    //cv::imshow("Features matching",imgmatches);
	//cv::waitKey();
}

void getLines(cv::Mat img1, cv::Mat img2, cv::Mat mask1, cv::Mat mask2, Matches_coordinates matches_coords, vector<blob> &blobs)
{
   //In wich blobs are this features
   
	cv::Mat stat,centroid,mask3; //para que preciso da mask3??
	int nLabels = connectedComponentsWithStats(mask1, mask3, stat,centroid,8, CV_16U);
	int w = img1.cols;
	blob blob_buff;
	
	
	for (size_t imatch = 0; imatch < matches_coords.L.size(); imatch++)
    {
		cv::Point pt1 = matches_coords.L[imatch];
		cv::Point pt2 = matches_coords.R[imatch];
		std::cout << "------------------------------------" << std::endl;
		std::cout << "Keypoint 1: " << pt1 << std::endl;
		std::cout << "Keypoint 2: " << pt2 << std::endl;
		for (int i=1;i<nLabels;i++)
		{
			std::cout << "objeto " << i << " limites x: " << stat.at<int>(i,CC_STAT_LEFT) << "-" << (stat.at<int>(i,CC_STAT_LEFT) + stat.at<int>(i,CC_STAT_WIDTH)) << "Check: " << ((pt1.x >= stat.at<int>(i,CC_STAT_LEFT)) && (pt1.x <= (stat.at<int>(i,CC_STAT_LEFT) + stat.at<int>(i,CC_STAT_WIDTH)))) << std::endl;
			std::cout << "objeto " << i << " limites y: " << stat.at<int>(i,CC_STAT_TOP) << "-" <<  (stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT)) << "Check: " << ((pt1.y >= stat.at<int>(i,CC_STAT_TOP)) && (pt1.y <= (stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT)))) << std::endl;
			
			if((pt1.x >= stat.at<int>(i,CC_STAT_LEFT)) && (pt1.x <= (stat.at<int>(i,CC_STAT_LEFT) + stat.at<int>(i,CC_STAT_WIDTH)) && (pt1.y >= stat.at<int>(i,CC_STAT_TOP)) && (pt1.y <= (stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT)))))
			{
				blob_buff.top = stat.at<int>(i,CC_STAT_TOP);
				blob_buff.bot = stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT);
				blob_buff.d = ((w - pt1.x) + pt2.x);
				blobs.push_back(blob_buff);
				std::cout << i << " blob top: " << blob_buff.top << std::endl;
				std::cout << i << " blob bot: " << blob_buff.bot << std::endl;
				std::cout << i << " d: " << blob_buff.d << std::endl;
				
				break;
			}	 
		}	
	}
	sort(blobs.begin(),blobs.end(),organizablob);
	std::cout << "number of good features found in blobs: " << blobs.size() << std::endl;
}

int vertically_allign_calib(cv::Mat &img1, cv::Mat &img2)
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
		
	////Translation
	//cv::Mat T = (Mat_<double>(3,3) << 1, 0, 0 , 0, 1, avg, 0, 0, 1);
	//cv::warpPerspective(output, output, T, output.size());
	return avg;
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

int horizontal_stitching_calib(cv::Mat &img1, cv::Mat &img2)
{
	vector<Point2f> corners1, corners2;
	
	bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
	bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);
	
	float sum=0;
	
	for(int i=0; i<corners1.size(); i++)
	{
		sum += img1.cols - corners1[i].x + corners2[i].x;
	}
	
	int avg = round(sum/corners1.size());
	return avg;
}

cv::Mat horizontal_stitching_apply(cv::Mat &img1, cv::Mat &img2,int avg)
{
	cv::Mat output = img1.clone();
	
	//Create buffer
	int right = img2.cols - avg;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    int buff = 0;
    cv::Scalar value(255,255,255);
	if(img2.rows >img1.rows) buff = img2.rows-img1.rows;
	
	//Create Border

	copyMakeBorder( img1, output, no, buff, no, right, borderType, value );
	
	//Create Mask
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
    
	img2.copyTo(outputBuff(cv::Rect(img1.cols-avg,0,img2.cols,img2.rows)),mask);
	
	//Get ghost image
	
	cv::Mat outputBuff2 = outputBuff.clone();
	output.copyTo(outputBuff,mask_output);
	
	cv::Mat outputGhost = outputBuff2*0.5 + outputBuff*0.5;
	
	//If want to show ghost change to outputGhost
	return outputGhost;
}	
	
cv::Mat horizontal_line_stitching_apply2(cv::Mat &img1, cv::Mat &img2,int avg, cv::Mat mask1, cv::Mat mask2, Matches_coordinates matches_coords)
{
	cv::Mat output = img1.clone();
	vector<blob> blobs; 
	
	//Match features and get the areas to line stitch
    getLines(img1, img2, mask1, mask2, matches_coords, blobs );
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
	
	//Create Mask
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
			std::cout << "i: " << i << std::endl;
			std::cout << "blob " << i2 << " top :" << blobs[i2].top << std::endl;
			std::cout << "blob " << i2 << " bot :" << blobs[i2].bot << std::endl;
			std::cout << "blob " << i2 << " d :" << blobs[i2].d << std::endl;
			
			if((i2>0 && blobs[i2-1].bot > blobs[i2].top)) 
			{
				if(i2 == blobs.size()-1)
				{
					for(i; i<img2.rows; i++)
					{
						img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					}
				break;
				}else continue;			
			}	
			
			if(i < blobs[i2].top )
			{
				for(i; i<blobs[i2].top; i++)
				{
					img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					
				}	
			}
			
			if(i == blobs[i2].top)
			{
				for(i; i<blobs[i2].bot; i++)
				{
			
					img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-blobs[i2].d,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					
				}	
			}
		}
	}
	return outputBuff;
}	
	
cv::Mat horizontal_line_stitching_apply(cv::Mat &img1, cv::Mat &img2,int avg)
{
cv::Mat output = img1.clone();
	
	//Create buffer
	int right = img2.cols - avg;
    int borderType = cv::BORDER_CONSTANT;
    int no = 0;
    int buff = 0;
    cv::Scalar value(255,255,255);
	if(img2.rows >img1.rows) buff = img2.rows-img1.rows;
	
	//Create Border

	copyMakeBorder( img1, output, no, buff, no, right, borderType, value );
	
	//Create Mask
	cv::Mat mask = cv::Mat::zeros(img2.size(), CV_8U);
	cv::Mat mask_output = cv::Mat::zeros(output.size(), CV_8U);
	cv::Mat img2_gray, output_gray;
	cv::cvtColor( img2, img2_gray, cv::COLOR_RGB2GRAY );
	cv::cvtColor( output, output_gray, cv::COLOR_RGB2GRAY);
	
	mask.setTo(255, img2_gray > 0);
    mask_output.setTo(255, output_gray > 0);
    
    cv::Mat outputBuff = output.clone();
    
    outputBuff.setTo(cv::Scalar(255,255,255));
    output.copyTo(outputBuff,mask_output);
	
	int gap = 1;
	int line = round((img2.rows/gap));
	int remainder = round((img2.rows % gap)); 	
	
	for(int i=0; i<line-1; i++)
	{
		
		avg	+= 0.1;
		img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
	}	
	
	return outputBuff;
}		

//Draw horizontal grid on images for vertical allignment analyses
void draw_grid(cv::Mat &mat)
{
	// assume that mat.type=CV_8UC3

	int dist=mat.size().height/30;

	int height=mat.size().height;
	int width= mat.size().width;
	for(int i=0;i<height;i+=dist)
	cv::line(mat,Point(0,i),Point(width,i),cv::Scalar(255,255,255));

}

//Get mask of new objects on image, used to feature detection and object size estimation for line stitching
void getMask(cv::Mat frame1, cv::Mat frame2, cv::Mat &fgMask)
{
	//cv::cvtColor(frame1, frame1, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(frame2, frame2, cv::COLOR_BGR2GRAY);
	
	//create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    
    pBackSub = createBackgroundSubtractorMOG2(500,50,true);
    //pBackSub = createBackgroundSubtractorKNN();
    
    //update the background model
    
    pBackSub->apply(frame1, fgMask, 0);
    pBackSub->apply(frame2, fgMask, 0);
    
    //show the current frame and the fg masks
    //cv::Mat diffrence = frame1 - frame2;
 
    //mask = cv::Mat::zeros(diffrence.size(), CV_8U);
    
    //mask.setTo(255, diffrence > 25);
	
    //Morphologic operations
    // Blur the foreground mask to reduce the effect of noise and false positives
	cv::blur(fgMask, fgMask, cv::Size(15, 15), cv::Point(-1, -1));
	// Remove the shadow parts and the noise
	cv::threshold(fgMask, fgMask, 128, 255, cv::THRESH_BINARY);
    cv::Mat element = getStructuringElement(MORPH_RECT,Size(20,20),Point(9,9));
    cv::morphologyEx(fgMask,fgMask,MORPH_OPEN,element);
    cv::morphologyEx(fgMask,fgMask,MORPH_CLOSE,element);
    cv::morphologyEx(fgMask,fgMask,MORPH_DILATE,element);
    
}

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////MAIN///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

int main()
{
    //Image sizes
	cv::namedWindow("Left image warped",cv::WINDOW_NORMAL);
	cv::resizeWindow("Left image warped",960,536);
	cv::namedWindow("Right image warped",cv::WINDOW_NORMAL);
	cv::resizeWindow("Right image warped",960,536);
	cv::namedWindow("Features matching",cv::WINDOW_NORMAL);
	cv::resizeWindow("Features matching",960,536);
	cv::namedWindow("Stitching result",cv::WINDOW_NORMAL);
	cv::resizeWindow("Stitching result",960,536);
	cv::namedWindow("Right image",cv::WINDOW_NORMAL);
	cv::resizeWindow("Right image",960,536);
	cv::namedWindow("Left image",cv::WINDOW_NORMAL);
	cv::resizeWindow("Left image",960,536);
	cv::namedWindow("Center image",cv::WINDOW_NORMAL);
	cv::resizeWindow("Center image",960,536);
	
	std::vector<cv::String> imagesL, imagesR, imagesC;
	cv::Mat imgL, imgR, imgC, imgR_warp, imgL_warp,img_matches;
	
	bool calibrationImg;
	std::cout << "Images to use? 1-chess 0-sample " << std::endl; // Type a number and press enter
	cin >> calibrationImg;
	
	std::string pathL, pathR, pathC;
	
	if(calibrationImg)
	{
		pathL = "../Images/squarefloor2/L1_squarefloor2_chess1.jpg"; //!!!!!!!!!!!!!!!!!!!!
		pathR = "../Images/squarefloor2/R1_squarefloor2_chess1.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!
		pathC = "../Images/squarefloor2/R1_squarefloor2_chess1.jpg";
	}else
	{
		pathL = "../Images/squarefloor2/L1_squarefloor2_box2.jpg"; //!!!!!!!!!!!!!!!!!!!!
		pathR = "../Images/squarefloor2/R1_squarefloor2_box2.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!
		pathC = "../Images/squarefloor2/R1_squarefloor2_chess1.jpg";
	}
	
	cv::glob(pathL, imagesL);
	cv::glob(pathR, imagesR);
	cv::glob(pathC, imagesC);
	
	
	//std::cout << "novo par" << std::endl;
	std::cout << imagesL[0] << std::endl;
	std::cout << imagesR[0] << std::endl;
	std::cout << imagesC[0] << std::endl;
		
	imgL = cv::imread(imagesL[0]);
	imgR = cv::imread(imagesR[0]);
	imgC = cv::imread(imagesC[0]);
	
	cv::imshow("Left image",imgL);
	cv::imshow("Right image",imgR);
	cv::imshow("Center image", imgC);
	
	//Get homography matrix and mask size
	cv::Mat Ht_L, Ht_R;
	cv::Size warpedL_size, warpedR_size;	
	int warpedL_height, warpedL_width , warpedR_height, warpedR_width, avg_v, avg_h;;
	bool newCalib = 0;
	bool estimate, save;
	
	//Ask for new calibration or saved one
	if(calibrationImg)
	{
		std::cout << "New calib? 1-yes 0-no " << std::endl; // Type a number and press enter
		cin >> newCalib;
	}
	
	//Make new calibration if needed
	if(newCalib)
	{
		vector<Point2f> cornersL, cornersC, cornersR;
		bool found1 = cv::findChessboardCorners(imgL, patternSize, cornersL);
		bool found2 = cv::findChessboardCorners(imgC, patternSize, cornersC);
		bool found3 = cv::findChessboardCorners(imgR, patternSize, cornersR);
		
		//Ask if the calibrations is to be estimated or direct
		std::cout << "Type of calibration? 1-estimated 0-to center image " << std::endl; // Type a number and press enter
		cin >> estimate;
		if(!estimate)
		{
			//Get the direct transform
			warpedL_size = getTransform(imgL,cornersL,cornersC,Ht_L); //returns mask and the matrix H
			warpedR_size = getTransform(imgR,cornersR,cornersC,Ht_R);
		}else
		{
			//estimate the transform for a 0 to 45 degrees
			float k;
			
			std::cout << "Position? (0º-45º)" << std::endl; // Type a number and press enter
			cin >> k;
			
			k = k/45;
			estimateTransform(imgL, imgR, cornersL, cornersR, Ht_L, Ht_R, warpedL_size, warpedR_size, k);
			std::cout << Ht_L << std::endl;
		}
		
		//Warp the prespective to get the value of the horizontal and vertical displacement
		cv::warpPerspective(imgL, imgL_warp, Ht_L, warpedL_size);
		cv::warpPerspective(imgR, imgR_warp, Ht_R, warpedR_size);
			
		avg_v = vertically_allign_calib(imgL_warp, imgR_warp);
		avg_h = horizontal_stitching_calib(imgL_warp, imgR_warp);
		
		//Ask if want to save the new calibration parameters
		std::cout << "Save Calibration? 1-yes 0-no " << std::endl; // Type a number and press enter
		cin >> save;
		
		if(save)
		{
			//Save new calibration parameters
			warpedL_height = warpedL_size.height;
			warpedL_width = warpedL_size.width;
			warpedR_height = warpedR_size.height;
			warpedR_width = warpedR_size.width;
				
			cv::FileStorage cv_file = cv::FileStorage("./Homography_params_squarefloor2.xml", cv::FileStorage::WRITE);
			cv_file.write("Ht_L",Ht_L);
			cv_file.write("Ht_R",Ht_R);
			cv_file.write("warpedL_height",warpedL_height);
			cv_file.write("warpedL_width",warpedL_width);
			cv_file.write("warpedR_height",warpedR_height);
			cv_file.write("warpedR_width",warpedR_width);
			cv_file.write("avg_v",avg_v);
			cv_file.write("avg_h",avg_h); 
		}
	}else
	{
		//Read pre-calibrated saved parameters
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
	}
		
    for(int i{0}; i<imagesL.size(); i++)
	{
		//Read images
		imgL = cv::imread(imagesL[i]);
		imgR = cv::imread(imagesR[i]);
		
		//FOR ANALYSIS PURPOSES
        //warp images to the same plane 
        cv::warpPerspective(imgL, imgL_warp, Ht_L, warpedL_size);
        cv::warpPerspective(imgR, imgR_warp, Ht_R, warpedR_size);
        
        //Allign features vertically
        vertically_allign_apply(imgL_warp, imgR_warp, avg_v);
        
        //Match features for analyses 
        match_features2(imgL_warp, imgR_warp, img_matches);
        
        //Draw horizontal evaluation grid
        draw_grid(img_matches);
        
        //Stich images
        //Needs two frames to calculate the background subtraction
        bool lines;
        std::cout << "Type of stitching? (only for sample) 0- Direct 1- Line " << std::endl; // Type a number and press enter
		cin >> lines;
		
		cv::Mat img_horz;
		
        if(lines)
		{
			//get background images for subtraction
			cv::Mat framechessL = cv::imread("../Images/squarefloor2/L1_squarefloor2_bgbox2.jpg");
			cv::Mat framechessR = cv::imread("../Images/squarefloor2/R1_squarefloor2_bgbox2.jpg");
			
			//GET MASKS AND FEATURES WITH ORIGINAL IMAGES 
			cv::Mat maskL, maskR;
			
			getMask(framechessL,imgL,maskL);
			getMask(framechessR,imgR,maskR);
			
			Matches_coordinates matches_coords;
			
			getMatches(imgL, imgR, maskL, maskR, matches_coords); 
			
			Matches_coordinates matches_coords_warped;
			cv::Point2f coordsL_warped_buff, coordsR_warped_buff;
		
			//WARP IMAGES, FEATURES AND MASKS
			//images
			
			//masks
			cv::Mat maskL_warp;
			cv::Mat maskR_warp;
		
			cv::warpPerspective(maskL, maskL_warp, Ht_L, warpedL_size);
			cv::warpPerspective(maskR, maskR_warp, Ht_R, warpedR_size);
			cv::imshow("Stitching result", maskL_warp);
			cv::imshow("Features matching",maskR_warp);
			cv::waitKey();
			
			//features coordinates
			cv::perspectiveTransform(matches_coords.L, matches_coords_warped.L, Ht_L);
			cv::perspectiveTransform(matches_coords.R, matches_coords_warped.R, Ht_R);
			
			//Allign features vertically
			vertically_allign_apply(maskL_warp, maskR_warp, avg_v);
			
			img_horz = horizontal_line_stitching_apply2(imgL_warp, imgR_warp, avg_h, maskL_warp, maskR_warp, matches_coords_warped);
			//img_horz = horizontal_line_stitching_apply(imgL_warp, imgR_warp, avg_h);
		}else
		{
			img_horz = horizontal_stitching_apply(imgL_warp, imgR_warp, avg_h);
		}
			
			
        //saveimage(imgL_warp);
		//saveimage(imgR_warp);
        
		cv::imshow("Left image warped",imgL_warp);
		cv::imshow("Right image warped",imgR_warp);
		cv::imshow("Stitching result", img_horz);
		cv::imshow("Features matching",img_matches);
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
