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
cv::Mat imgL, imgR, imgC, imgL_crop, imgR_crop,imgC_crop,imgR_warp, imgL_warp,img_matches, imgR_vert_allig;

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
cv:Mat Ht = T * H; 

//std::cout << "original_corners: " << original_corners << std::endl;
//std::cout << "Altura: " << size_mask.height << " Largura: " << size_mask.width << std::endl;    
//std::cout << "Minimo_height: " << size_mask_min.height << " Maximo_height: " << size_mask_max.height << std::endl;
//std::cout << "Minimo_width: " << size_mask_min.width << " Maximo_width: " << size_mask_max.width << std::endl;

//Warp image with mask size

cv::warpPerspective(img1, img1_warp, Ht, size_mask);

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

void estimateTransform(cv::Mat imgL,cv::Mat imgR, vector<Point2f> &cornersL, vector<Point2f> &cornersR, cv::Mat &Ht_L, cv::Mat &Ht_R, cv::Size &mask_L, cv::Size &mask_R, float k)
{
	
	vector<Point2f> cornersC_estimated = cornersL;

	//Estimate by assuming linear the corners of C
	
	diffVectors(cornersC_estimated, cornersR);
	multVector(cornersC_estimated,k);
	sumVectors(cornersC_estimated,cornersR);
	std::cout << cornersC_estimated << std::endl;
	mask_L = getTransform(imgL, cornersL, cornersC_estimated, Ht_L);
	mask_R = getTransform(imgR, cornersR, cornersC_estimated, Ht_R);

}

void right2leftplane(cv::Mat &img2, cv::Mat &img1, cv::Mat &img2_warp)
{
vector<Point2f> corners1, corners2;

bool found1 = cv::findChessboardCorners(img1, patternSize, corners1);
bool found2 = cv::findChessboardCorners(img2, patternSize, corners2);

//Get the homography matrix
cv::Mat H = cv::findHomography(corners2, corners1);
cv::Mat H_inv = H.inv();

std::cout << "H:\n" << H << std::endl;

//Find mask size for warping
vector<Point2f> original_corners, mask_corners;
original_corners.push_back(cv::Point2f(0,0));
original_corners.push_back(cv::Point2f(img2.cols-1,0));
original_corners.push_back(cv::Point2f(img2.cols-1,img2.rows-1));
original_corners.push_back(cv::Point2f(0,img2.rows-1));

cv::perspectiveTransform(original_corners, mask_corners, H);

cv::Size size_mask_max(-1000,-1000),size_mask_min(1000,1000), size_mask(0,0);

for(int i=0; i<mask_corners.size(); i++)
{
	std::cout << "x:" << mask_corners[i].x << "y:" << mask_corners[i].y << std::endl;
	size_mask_min.height= min((int)mask_corners[i].y,(int)size_mask_min.height);
	size_mask_max.height= max((int)mask_corners[i].y,(int)size_mask_max.height);
	
	size_mask_min.width= min((int)mask_corners[i].x,(int)size_mask_min.width);
	size_mask_max.width= max((int)mask_corners[i].x,(int)size_mask_max.width);
}

size_mask.height = size_mask_max.height - size_mask_min.height;
size_mask.width = size_mask_max.width - size_mask_min.width;

std::cout << size_mask_min.width << std::endl;

cv::Mat T = (Mat_<double>(3,3) << 1, 0, -1*size_mask_min.width , 0, 1, -1* size_mask_min.height, 0, 0, 1);
cv:Mat Ht = T * H; 

//std::cout << "original_corners: " << original_corners << std::endl;
//std::cout << "Altura: " << size_mask.height << " Largura: " << size_mask.width << std::endl;    
//std::cout << "Minimo_height: " << size_mask_min.height << " Maximo_height: " << size_mask_max.height << std::endl;
//std::cout << "Minimo_width: " << size_mask_min.width << " Maximo_width: " << size_mask_max.width << std::endl;

//Warp image with mask size

cv::warpPerspective(img2, img2_warp, Ht, size_mask);

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

	std::vector<DMatch> match1;
	std::vector<DMatch> match2;
	int i2 = 0;
	
	match1.push_back(matches[0][0]);
	for(int i=1; i<matches.size(); i++)
	{
		if( matches[i][0].distance < match1[0].distance ){
			match1[0]=matches[i][0];
			
			
		}
		
	}
	
	cv::drawMatches(img1, keypoints1, img2, keypoints2, match1, output);
}

void getLines(cv::Mat img1, cv::Mat img2, cv::Mat mask1, cv::Mat mask2, int &top, int &bot, int &d)
{
	//Features dectection and matching
	Ptr<ORB> detector = cv::ORB::create();
	std::vector<KeyPoint> keypoints1,keypoints2;
	Mat descriptors1,descriptors2;

	detector->detectAndCompute( img1, mask1, keypoints1, descriptors1 );
	detector->detectAndCompute( img2, mask2, keypoints2, descriptors2 );

	//Matching FEatures
	BFMatcher matcher(NORM_HAMMING);
	//matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
	std::vector<vector<DMatch> > matches;

	matcher.knnMatch(descriptors1, descriptors2, matches,2);

	std::vector<DMatch> match1;
	std::vector<DMatch> match2;
	int i2 = 0;
	
	//-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    std::vector<DMatch> query;
    std::vector<DMatch> train;
    
    good_matches.push_back(matches[0][0]);
    //get the best match
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < good_matches[0].distance)
        {
            good_matches[0] = matches[i][0];
        }
    }
  
   	std::cout << "Keypoint 1: " << keypoints1[good_matches[0].queryIdx].pt << std::endl;
	std::cout << "Keypoint 2: " << keypoints2[good_matches[0].trainIdx].pt << std::endl;
   
   //In wich blobs are this features
   
	cv::Mat stat,centroid,mask3; //para que preciso da mask3??
	int nLabels = connectedComponentsWithStats(mask1, mask3, stat,centroid,8, CV_16U);
	int w = img1.cols;
	
	for (int i=1;i<nLabels;i++)
	{
		cv::Point pt1 = keypoints1[good_matches[0].queryIdx].pt;
		cv::Point pt2 = keypoints2[good_matches[0].trainIdx].pt;
		std::cout << "img1 x: " << pt1.y << std::endl;
		std::cout << "img2 x: " << pt2.y << std::endl;
		std::cout << " top: " << stat.at<int>(i,CC_STAT_TOP) << std::endl;
		std::cout << " bot: " <<  (stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT)) << std::endl;
		std::cout << i << std::endl;
		if((pt1.y > stat.at<int>(i,CC_STAT_TOP)) && (pt1.y < (stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT))))
		{
			std::cout << i << std::endl;
			top = stat.at<int>(i,CC_STAT_TOP);
			bot = stat.at<int>(i,CC_STAT_TOP) + stat.at<int>(i,CC_STAT_HEIGHT);
			d = ((w - pt1.x) + pt2.x)
			;
			std::cout << i << " blob top: " << top << std::endl;
			std::cout << i << " blob bot: " << bot << std::endl;
			std::cout << i << " d: " << d << std::endl;
			std::cout << "Keypoint 1: " << keypoints1[good_matches[0].queryIdx].pt << std::endl;
			std::cout << "Keypoint 2: " << keypoints2[good_matches[0].trainIdx].pt << std::endl;
			break;
		}
	     
	}
	std::cout << "ja sai" << std::endl;
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


    // color all masked pixel red:
	
	img2.copyTo(outputBuff(cv::Rect(img1.cols-avg,0,img2.cols,img2.rows)),mask);
	
	//Get ghost image
	
	cv::Mat outputBuff2 = outputBuff.clone();
	output.copyTo(outputBuff,mask_output);
	
	cv::Mat outputGhost = outputBuff2*0.5 + outputBuff*0.5;
	
	
	
	//If want to show ghost change to outputGhost
	return outputGhost;
}	
	
cv::Mat horizontal_line_stitching_apply2(cv::Mat &img1, cv::Mat &img2,int avg, cv::Mat mask1, cv::Mat mask2)
{
	cv::Mat output = img1.clone();
	int top, bot , d;
	//Match features and get the areas to line stitch
    getLines(img1, img2, mask1, mask2, top, bot , d );
  
	//Create buffer
	int right;
	if (d > avg)
		right = img2.cols - avg;
	else
		right = img2.cols - d;
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
	
	std::cout << "top:  "<< top << std::endl;
	std::cout << "bot:  "<< bot << std::endl;
	std::cout << "avg:  "<< avg << std::endl;
	std::cout << "d:  "<< d << std::endl;
	for(int i=0; i<top; i++)
	{
		
		img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
	}	
	
	std::cout << "aqui" << std::endl;
	for(int i=top; i<bot; i++)
	{
		img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-d,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
	}	
	std::cout << "ou aqui" << std::endl;
	for(int i=bot; i<img2.rows; i++)
	{
		
		img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
	}	
	std::cout << "afinal nao" << std::endl;
	
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
	
	// Path of the folder containing checkerboard images
	//std::string pathL = "./ImagesL/*.jpg"; //!!!!!!!!!!!!!!!!!!!!
	//std::string pathR = "./ImagesR/*.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!
	//std::string pathC = "./ImagesC/*.jpg";
	bool calibrationImg;
	std::cout << "Images to use? 1-chess 0-sample " << std::endl; // Type a number and press enter
	cin >> calibrationImg;
	
	std::string pathL, pathR, pathC;
	
	if(calibrationImg)
	{
		pathL = "../Images/L1_squarefloor.jpg"; //!!!!!!!!!!!!!!!!!!!!
		pathR = "../Images/R1_squarefloor.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!
		pathC = "../Images/C1_squarefloor.jpg";
	}else
	{
		pathL = "../Images/L4_squarefloor.jpg"; //!!!!!!!!!!!!!!!!!!!!
		pathR = "../Images/R4_squarefloor.jpg"; //!!!!!!!!!!!!!!!!!!!!!!!
		pathC = "../Images/C4_squarefloor.jpg";
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
	   
	////Crop images for calibration
	//int x = imgL.cols/3;
	//cv::Range cols(x*2, imgL.cols);
	//cv::Range rows(0, imgL.rows);
		  
	//cv::Range cols2(0, x);
	//imgL_crop = imgL(rows, cols);
	//imgR_crop = imgR(rows, cols2);
		
	//imgL_crop = imgL.clone();
	//imgR_crop = imgR.clone();
	//imgC_crop = imgC.clone();
		
	//left2rightplane(imgL_crop,imgC_crop, imgL_crop_warp);
	//right2leftplane(imgR_crop, imgC_crop, imgR_crop_warp);
		
	//Get homography matrix and mask size
	cv::Mat Ht_L, Ht_R;
	cv::Size mask_L, mask_R;	
	int mask_L_height, mask_L_width , mask_R_height, mask_R_width, avg_v, avg_h;;
	bool newCalib = 0;
	bool estimate, save;
	
	if(calibrationImg)
	{
		std::cout << "New calib? 1-yes 0-no " << std::endl; // Type a number and press enter
		cin >> newCalib;
	}
	
	if(newCalib)
	{
		vector<Point2f> cornersL, cornersC, cornersR;
		bool found1 = cv::findChessboardCorners(imgL, patternSize, cornersL);
		bool found2 = cv::findChessboardCorners(imgC, patternSize, cornersC);
		bool found3 = cv::findChessboardCorners(imgR, patternSize, cornersR);
		
		std::cout << "Type of calibration? 1-estimated 0-to center image " << std::endl; // Type a number and press enter
		cin >> estimate;
		if(!estimate)
		{
			/////////////GET DIRECT TRANSFORM////////////
			mask_L = getTransform(imgL,cornersL,cornersC,Ht_L); //returns mask and the matrix H
			mask_R = getTransform(imgR,cornersR,cornersC,Ht_R);
		}else
		{
			////////////ESTIMATE TRANSFORM//////////
			float k;
			std::cout << "Position? (0-1)" << std::endl; // Type a number and press enter
			cin >> k;
			
			estimateTransform(imgL, imgR, cornersL, cornersR, Ht_L, Ht_R, mask_L, mask_R, k);
			std::cout << Ht_L << std::endl;
		}
		//Warp
		cv::warpPerspective(imgL, imgL_warp, Ht_L, mask_L);
		cv::warpPerspective(imgR, imgR_warp, Ht_R, mask_R);
			
		avg_v = vertically_allign_calib(imgL_warp, imgR_warp);
		avg_h = horizontal_stitching_calib(imgL_warp, imgR_warp);
		
		std::cout << "Save transformation? 1-yes 0-no " << std::endl; // Type a number and press enter
		cin >> save;
		if(save)
		{
			mask_L_height = mask_L.height;
			mask_L_width = mask_L.width;
			mask_R_height = mask_R.height;
			mask_R_width = mask_R.width;
				
			cv::FileStorage cv_file = cv::FileStorage("./Homography_params_squarefloor.xml", cv::FileStorage::WRITE);
			cv_file.write("Ht_L",Ht_L);
			cv_file.write("Ht_R",Ht_R);
			cv_file.write("mask_L_height",mask_L_height);
			cv_file.write("mask_L_width",mask_L_width);
			cv_file.write("mask_R_height",mask_R_height);
			cv_file.write("mask_R_width",mask_R_width);
			cv_file.write("avg_v",avg_v);
			cv_file.write("avg_h",avg_h); 
		}
	}else
	{
		cv::FileStorage fs("./Homography_params_squarefloor.xml", cv::FileStorage::READ);

		fs["Ht_L"] >> Ht_L;
		fs["Ht_R"] >> Ht_R;
		fs["mask_L_height"] >> mask_L_height;
		fs["mask_L_width"] >> mask_L_width; 
		fs["mask_R_height"] >> mask_R_height;
		fs["mask_R_width"] >> mask_R_width; 
		fs["avg_v"] >> avg_v;
		fs["avg_h"] >> avg_h; 
		
		mask_L.height = mask_L_height;
        mask_L.width  = mask_L_width ;
        mask_R.height = mask_R_height;
        mask_R.width  = mask_R_width ;
	}
		
    for(int i{0}; i<imagesL.size(); i++)
	{
		imgL = cv::imread(imagesL[i]);
		imgR = cv::imread(imagesR[i]);
		
        //warp images to the same plane
        cv::warpPerspective(imgL, imgL_warp, Ht_L, mask_L);
        cv::warpPerspective(imgR, imgR_warp, Ht_R, mask_R);
        
        //Allign features vertically
        vertically_allign_apply(imgL_warp, imgR_warp, avg_v);
        
        //Match features for analyses 
        match_features2(imgL_warp, imgR_warp, img_matches);
        
        //Draw horizontal evaluation grid
        draw_grid(img_matches);
        
        //Stich images
        
        //Needs two frames to calculate the background subtraction
        bool stitch;
        std::cout << "Type of stitching? (only for sample) 0- Direct 1- Line " << std::endl; // Type a number and press enter
		cin >> stitch;
		
		cv::Mat img_horz;
        if(stitch)
		{
			cv::Mat framechessL = cv::imread("../Images/L1_squarefloor.jpg");
			cv::Mat framechessR = cv::imread("../Images/R1_squarefloor.jpg");
			
			cv::Mat framechessL_warp;
			cv::Mat framechessR_warp;
			//warp images to the same plane
			cv::warpPerspective(framechessL, framechessL_warp, Ht_L, mask_L);
			cv::warpPerspective(framechessR, framechessR_warp, Ht_R, mask_R);
			
			//Allign features vertically
			vertically_allign_apply(framechessL_warp, framechessR_warp, avg_v);
			
			cv::Mat mask, mask2;
	
			getMask(framechessL_warp,imgL_warp,mask);
			getMask(framechessR_warp,imgR_warp,mask2);
			
			img_horz = horizontal_line_stitching_apply2(imgL_warp, imgR_warp, avg_h, mask, mask2);
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
