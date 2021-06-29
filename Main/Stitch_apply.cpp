
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
  int height;
  int leftL;
  int widthL;
  int leftR;
  int widthR;
  int dtotal;
  int nfeatures;
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

void drawSquares(cv::Mat &mask1, vector<blob> blobs)
{
   //In wich blobs are this features
	
	for (int i=0;i<blobs.size();i++)
	{
		// Top Left Corner
		Point p1(blobs[i].leftL, blobs[i].top);
	  
		// Bottom Right Corner
		Point p2(blobs[i].leftL+blobs[i].widthL, blobs[i].top+blobs[i].height);
	  
		int thickness = 2;
	  
		// Drawing the Rectangle
		cv::rectangle(mask1, p1, p2, Scalar(255, 0, 0), thickness, LINE_8);	
		
	}	
		cv::namedWindow("Result",cv::WINDOW_NORMAL);
	cv::resizeWindow("Result",960,536);
	cv::imshow("Result", mask1);
	cv::waitKey();
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

    std::vector<DMatch> sorted_matches,good_matches;
    
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
			good_matches.push_back(sorted_matches[i]);
			matches_coords.L.push_back(keypoints1[sorted_matches[i].queryIdx].pt);
			matches_coords.R.push_back(keypoints2[sorted_matches[i].trainIdx].pt);
			std::cout << "Keypoint 1: " << keypoints1[sorted_matches[i].queryIdx].pt << std::endl;
			std::cout << "Keypoint 2: " << keypoints2[sorted_matches[i].trainIdx].pt << std::endl;
			std::cout << "Distance: " << sorted_matches[i].distance << std::endl;
			std::cout << "--------------------" << std::endl;
		}
    }
    
    cv::Mat imgmatches;
    cv::drawMatches(mask1, keypoints1, mask2, keypoints2, good_matches, imgmatches);
    cv::namedWindow("matches",cv::WINDOW_NORMAL);
	cv::resizeWindow("matches",960,536);
    cv::imshow("matches",imgmatches);
	cv::waitKey();
}

vector<blob> getBlobs(cv::Mat mask1, cv::Mat mask2, int avg, Matches_coordinates matches_coords)
{
   //In wich blobs are this features
	
	cv::Mat statL,statR,centroid,mask3; //para que preciso da mask3??
	int nLabelsL = connectedComponentsWithStats(mask1, mask3, statL,centroid,8, CV_16U);
	int nLabelsR = connectedComponentsWithStats(mask2, mask3, statR,centroid,8, CV_16U);
	int w = mask1.cols;
	blob blob_buff;
	int maxLabels = nLabelsL;
	if(nLabelsR>nLabelsL) maxLabels = nLabelsR;
	vector<blob> blobs(maxLabels);
	
	for(int i=0;i<2;i++)
	{
		std::cout << "blob " << i << " : Left: " << statR.at<int>(i,CC_STAT_LEFT) << "Width: " << statR.at<int>(i,CC_STAT_WIDTH) << std::endl;  
	} 
	for (size_t imatch = 0; imatch < matches_coords.L.size(); imatch++)
    {
		cv::Point pt1 = matches_coords.L[imatch];
		cv::Point pt2 = matches_coords.R[imatch];

		for (int i=1;i<nLabelsL;i++) //the first one is the background
		{
			if((pt1.x >= statL.at<int>(i,CC_STAT_LEFT)) && (pt1.x <= (statL.at<int>(i,CC_STAT_LEFT) + statL.at<int>(i,CC_STAT_WIDTH)) && (pt1.y >= statL.at<int>(i,CC_STAT_TOP)) && (pt1.y <= (statL.at<int>(i,CC_STAT_TOP) + statL.at<int>(i,CC_STAT_HEIGHT)))))
			{
				//std::cout << "pertence a esquerda: "<< i << std::endl;
			
				for(int i2=1; i2<nLabelsR; i2++)
				{
					if((pt2.x >= statR.at<int>(i2,CC_STAT_LEFT)) && (pt2.x <= (statR.at<int>(i2,CC_STAT_LEFT) + statR.at<int>(i2,CC_STAT_WIDTH)) && (pt2.y >= statR.at<int>(i2,CC_STAT_TOP)) && (pt2.y <= (statR.at<int>(i2,CC_STAT_TOP) + statR.at<int>(i2,CC_STAT_HEIGHT)))))
					{
						//std::cout << "pertence a direita: " << i2 << std::endl;	
						if(!blobs[i].top)
						{
							//std::cout << "criou" << std::endl;
							blobs[i].top = statL.at<int>(i,CC_STAT_TOP);
							blobs[i].height = statL.at<int>(i,CC_STAT_HEIGHT);
							blobs[i].leftL = statL.at<int>(i,CC_STAT_LEFT);
							blobs[i].widthL = statL.at<int>(i,CC_STAT_WIDTH);
							blobs[i].leftR = statR.at<int>(i2,CC_STAT_LEFT);
							blobs[i].widthR = statR.at<int>(i2,CC_STAT_WIDTH);
					
						}
						blobs[i].dtotal += ((w - pt1.x) + pt2.x);
						blobs[i].nfeatures += 1;
						break;
					}
				}
			}
		}
	}
	//Get average distance and remove blobs with no features
	for (int i=0;i<blobs.size();i++) //the first one is the background?
	{
		//std::cout << "Blob nº "<< i << " Top: " << blobs[i].top << " Height: " << blobs[i].height  << " dtotal: " << blobs[i].dtotal << " nfeatures: " << blobs[i].nfeatures << " d: " << blobs[i].d << std::endl;
		if(blobs[i].nfeatures>0) 
		{
			blobs[i].d = (avg-(blobs[i].dtotal/blobs[i].nfeatures))/2;
		}else{
			blobs.erase(blobs.begin()+i);
			i--;
		}
	}
	//std::cout << "No features blobs removed" << std::endl;
	for (int i=0;i<blobs.size();i++) 
	{
		std::cout << "Blob nº "<< i << " Top: " << blobs[i].top << " Height: " << blobs[i].height  << " dtotal: " << blobs[i].dtotal << " nfeatures: " << blobs[i].nfeatures << " d: " << blobs[i].d << std::endl;
		std::cout << " LeftL: " << blobs[i].leftL <<" WidthL: " << blobs[i].widthL << " LeftR: " << blobs[i].leftR <<" WidthR: " << blobs[i].widthR << std::endl;
		std::cout << avg<< std::endl;
	}
	sort(blobs.begin(),blobs.end(),organizablob);
	std::cout << "number of blobs with features: " << blobs.size() << std::endl;
	return blobs;
}

//cv::Mat horizontal_line_stitching_apply(cv::Mat &img1, cv::Mat &img2, int avg, cv::Mat mask1, cv::Mat mask2, Matches_coordinates matches_coords, vector<blob> blobs)
//{
	//cv::Mat output = img1.clone();
	
    //int  dMin = avg + 1;
    //if (blobs.size() > 0)
    //{
		//auto it = std::min_element(blobs.begin(), blobs.end(), [](const blob& a,const blob& b) { return a.d < b.d; });
		//dMin = it[0].d; 
		//std::cout << "dmin: " << dMin << std::endl;
    //}
  
	////Create buffer
	//int right;
	//if (dMin > avg)
		//right = img2.cols - avg;
	//else
		//right = img2.cols - dMin;
    //int borderType = cv::BORDER_CONSTANT;
    //int no = 0;
    //int buff = 0;
    //cv::Scalar value(255,255,255);
	//if(img2.rows >img1.rows) buff = img2.rows-img1.rows;
	
	////Create Border
	//copyMakeBorder( img1, output, no, buff, no, right, borderType, value );
	
	////Create Mask to clean black part
	//cv::Mat mask = cv::Mat::zeros(img2.size(), CV_8U);
	//cv::Mat mask_output = cv::Mat::zeros(output.size(), CV_8U);
	
	//cv::Mat img2_gray, output_gray;
	
	//cv::cvtColor( img2, img2_gray, cv::COLOR_RGB2GRAY );
	//cv::cvtColor( output, output_gray, cv::COLOR_RGB2GRAY);
	
	//mask.setTo(255, img2_gray > 0);
    //mask_output.setTo(255, output_gray > 0);
    //mask_output.setTo(0,output_gray == 255);
    //cv::Mat outputBuff = output.clone();
    
    //outputBuff.setTo(cv::Scalar(255,255,255));
    //output.copyTo(outputBuff,mask_output);
    
	//int gap = 1;
	//int line = round((img2.rows/gap));
	//std::cout << line << std::endl;
	//int remainder = round((img2.rows % gap)); 	

	//int top = 0,bot,d;
	//int i = 0;
	
	//if(blobs.size() == 0)
	//{
		//for(i; i<img2.rows; i++)
		//{
			//img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
		//}
	//}else{	
		
		//for(int i2=0;i2 < blobs.size();i2++)
		//{	
			//if(i < blobs[i2].top )
			//{
				//for(i; i<blobs[i2].top; i++)
				//{
					//img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					
				//}	
			
				//for(i; i<blobs[i2].bot; i++)
				//{
			
					//img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-blobs[i2].d,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
					
				//}
					
			//}
			//if(i2 == blobs.size()-1)
			//{
				//for(i; i<img2.rows; i++)
				//{
					//img2(cv::Rect(0,i*gap,img2.cols,gap)).copyTo(outputBuff(cv::Rect(img1.cols-avg,i*gap,img2.cols,gap)),mask(cv::Rect(0,i*gap,img2.cols,gap)));
				//}
			//}
			
		//}
	//}
	//return outputBuff;
//}	

cv::Mat horizontal_blob_stitching_apply(cv::Mat &img1, cv::Mat &img2, cv::Mat bkg1, cv::Mat bkg2, int avg, vector<blob> blobs)
{
	//Create Mask to clean black part
	cv::Mat mask1 = cv::Mat::zeros(img1.size(), CV_8U);
	cv::Mat mask2 = cv::Mat::zeros(img2.size(), CV_8U);
	
	cv::Mat img1_gray, img2_gray, output_gray;
	
	cv::cvtColor( img1, img1_gray, cv::COLOR_RGB2GRAY );
	cv::cvtColor( img2, img2_gray, cv::COLOR_RGB2GRAY );
	
	mask1.setTo(255, img1_gray > 0);
	mask2.setTo(255, img2_gray > 0);
    
    int buff=0;
    if(img2.rows > img1.rows) buff = img2.rows-img1.rows;
    cv::Mat outputBuff(img1.rows+buff,img1.cols+img2.cols-avg,CV_8UC3);
    
    outputBuff.setTo(cv::Scalar(255,255,255));
    
	//use the bckg to hide moved blobs
	cv::Mat img2_patched = img2.clone();
	cv::Mat img1_patched = img1.clone();	
	
	//Make patches with the background on the area of the blob
	for(int i = 0; i<blobs.size();i++)
	{
		bkg1(cv::Rect(blobs[i].leftL, blobs[i].top, blobs[i].widthL, blobs[i].height)).copyTo(img1_patched(cv::Rect(blobs[i].leftL, blobs[i].top, blobs[i].widthL, blobs[i].height)),mask1(cv::Rect(blobs[i].leftL, blobs[i].top, blobs[i].widthL, blobs[i].height)));
		bkg2(cv::Rect(blobs[i].leftR, blobs[i].top, blobs[i].widthR, blobs[i].height)).copyTo(img2_patched(cv::Rect(blobs[i].leftR, blobs[i].top, blobs[i].widthR, blobs[i].height)),mask2(cv::Rect(blobs[i].leftR, blobs[i].top, blobs[i].widthR, blobs[i].height)));
	}
	
	//drawSquares(img1,blobs);
	
	//Stitch the patched images to create the panorama
	img1_patched(cv::Rect(0,0,img1_patched.cols,img1_patched.rows)).copyTo(outputBuff(cv::Rect(0,0,img1_patched.cols,img1_patched.rows)),mask1(cv::Rect(0,0,img1_patched.cols,img1_patched.rows)));
	img2_patched(cv::Rect(0,0,img2_patched.cols,img2_patched.rows)).copyTo(outputBuff(cv::Rect(img1.cols-avg,0,img2_patched.cols,img2_patched.rows)),mask2(cv::Rect(0,0,img2_patched.cols,img2_patched.rows)));
	
	//Stamp the blob on the "virtual camera" position"
	
	cv::Mat outputBuff2 ;
	outputBuff2 = Mat::zeros(outputBuff.rows, outputBuff.cols, CV_8UC3);
	cv::Mat outputBuff1 = outputBuff2.clone();
	int width;
	cv::namedWindow("Result",cv::WINDOW_NORMAL);
	cv::resizeWindow("Result",960,536);
	cv::Mat img1buff,img2buff;
	for(int i = 0; i<blobs.size();i++)
	{
		width = blobs[i].widthL;
		if(blobs[i].widthR > width) width = blobs[i].widthR;
		
		int midPoint = img1.cols-avg+blobs[i].leftR+blobs[i].d + width/2;
		double ratio = (midPoint-(img1.cols-avg))/(double)avg;
		
		//std::cout << "avg: " << avg << " ratio: " <<  ratio << " midpoint: "<< midPoint << " (midPoint-(img1.cols-avg)): " << (midPoint-(img1.cols-avg))<< " (midPoint-(img1.cols-avg))/avg: " << (midPoint-(img1.cols-avg))/avg << std::endl;
		img1buff = img1 * (1-ratio);
		img2buff = img2 * ratio;
		
		img1buff(cv::Rect(blobs[i].leftL, blobs[i].top, width, blobs[i].height)).copyTo(outputBuff1(cv::Rect(img1.cols-avg+blobs[i].leftR+blobs[i].d,blobs[i].top, width,blobs[i].height)),mask1(cv::Rect(blobs[i].leftL, blobs[i].top, width, blobs[i].height)));
		img2buff(cv::Rect(blobs[i].leftR, blobs[i].top, width, blobs[i].height)).copyTo(outputBuff2(cv::Rect(img1.cols-avg+blobs[i].leftR+blobs[i].d,blobs[i].top, width,blobs[i].height)),mask2(cv::Rect(blobs[i].leftR, blobs[i].top, width, blobs[i].height)));
		outputBuff(cv::Rect(img1.cols-avg+blobs[i].leftR+blobs[i].d,blobs[i].top, width,blobs[i].height)) = 0;
		outputBuff += outputBuff2 + outputBuff1;
		
		outputBuff1 = 0;
		outputBuff2 = 0;
		img1buff = 0;
		img2buff = 0;
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
	
	vector<blob> blobs = getBlobs(maskL_warp, maskR_warp, avg_h, matches_coords);
	
	//LINE STITCHING
	cv::Mat result;
	
	//result = horizontal_line_stitching_apply(imgL_warp, imgR_warp, avg_h, maskL_warp, maskR_warp, matches_coords, blobs);
	result = horizontal_blob_stitching_apply(imgL_warp, imgR_warp, imgL_bg_warp, imgR_bg_warp, avg_h, blobs);
	//SHOW RESULT
	cv::namedWindow("Result",cv::WINDOW_NORMAL);
	cv::resizeWindow("Result",960,536);
	cv::imshow("Result", result);
	cv::waitKey();
			
}
