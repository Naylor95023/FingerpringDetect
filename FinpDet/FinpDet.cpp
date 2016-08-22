#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

// Function for thinning any given binary image within the range of 0-255. 
void _thinningIteration(Mat& im, int iter)
{
	Mat marker = Mat::zeros(im.size(), CV_8UC1);
	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}
	im &= ~marker;
}
void _thinning(Mat& im)
{
	// Enforce the range tob e in between 0 - 255
	im /= 255;

	Mat prev = Mat::zeros(im.size(), CV_8UC1);
	Mat diff;
	do {
		_thinningIteration(im, 0);
		_thinningIteration(im, 1);
		absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (countNonZero(diff) > 0);

	im *= 255;
}
//Use thinned date calculature its harris and take a threshold value to filt the features.
Mat _detector(Mat input, int showIf, string filename)
{
	// Now lets detect the strong minutiae using Haris corner detection
	Mat harris_corners, harris_normalised;
	harris_corners = Mat::zeros(input.size(), CV_32FC1);
	cornerHarris(input, harris_corners, 2, 3, 0.04, BORDER_DEFAULT);
	normalize(harris_corners, harris_normalised, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// Select the strongest corners that you want
	int threshold_harris = 125;
	vector<KeyPoint> keypoints;

	// Make a color clone for visualisation purposes
	Mat rescaled;
	convertScaleAbs(harris_normalised, rescaled);
	Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
	Mat in[] = { rescaled, rescaled, rescaled };
	int from_to[] = { 0, 0, 1, 1, 2, 2 };
	mixChannels(in, 3, &harris_c, 1, from_to, 3);
	for (float x = 0.0; x < harris_normalised.cols; x++) {
		for (float y = 0.0; y < harris_normalised.rows; y++) {
			if ((int)harris_normalised.at<float>(y, x) > threshold_harris) {
				// Draw or store the keypoint location here, just like you decide. In our case we will store the location of the keypoint
				circle(harris_c, Point(x, y), 5, Scalar(0, 255, 0), 1);
				circle(harris_c, Point(x, y), 1, Scalar(0, 0, 255), 1);
				keypoints.push_back(KeyPoint(x, y, 1.0));
			}
		}
	}
	//get descriptors
	Mat descriptors;
	auto featureExtractor = DescriptorExtractor::create("ORB");
    featureExtractor->compute(input, keypoints, descriptors);

	if(showIf > 3)imshow(filename + "Harris-Step4", harris_c); 
	
	return descriptors;
}
//Enhance the data quality(readGray->Binary->thinned)
Mat _preTreat(string filename, int showIf)
{
	string PATH = "G:\\_dbf\\";
	// Read in an input image - directly in grayscale CV_8UC1
	Mat input = imread(PATH + filename, IMREAD_GRAYSCALE);
	// Binarize the image, through local thresholding
	Mat input_binary;
	threshold(input, input_binary, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	// Now apply the thinning algorithm
	Mat input_thinned = input_binary.clone();
	_thinning(input_thinned);

	if(showIf < 5){
		if(showIf > 0) imshow(filename + "Input---Step1", input);
		if(showIf > 1)imshow(filename + "Binary--Step2", input_binary); 
		if(showIf > 2)imshow(filename + "Thinned-Step3", input_thinned);
	}
	return _detector(input_thinned, showIf, filename);
}
//Input man1 and man2 features descriptors, compare they to get "Matching Score"
void _compareData(Mat descriptors, Mat descriptors2, string FILES)
{
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector< DMatch > matches;
    matcher->match(descriptors, descriptors2, matches);

    // Loop over matches and multiply
    float score = 0.0;
    for(int i=0; i < matches.size(); i++){
        DMatch current_match = matches[i];
        score = score + current_match.distance;
    }

	cout << endl << "Files : "<< FILES << endl <<
		"Matching score = " << score << endl;
}
//man->personID, sample->dataNumber, get the true data path
string _setPerson(int man, int sample)
{
	stringstream ss;
	ss<<"10"<<man<<"_"<<sample <<".tif";
	return ss.str(); 
}
//compare person1 to person2's all samples
void _comparePerson(int person1, int person2, int showIf)
{
	string DATA_1 = _setPerson(person1, 1);
	Mat descriptors = _preTreat(DATA_1, showIf);

	for(int i = 1; i <= 8; i++){
		string DATA_2 = _setPerson(person2, i);
		Mat descriptors2 = _preTreat(DATA_2, showIf);
		_compareData(descriptors, descriptors2, DATA_1 + " & " + DATA_2);
	}cout << "--------------------------------------"<<endl;

}

//main----------------------------------------------------------
int main()
{	
	int PERSON_1 = 1; int SAMPLE_1 = 1; int SHOW_STEP_1 = 4;
	int PERSON_2 = 1; int SAMPLE_2 = 2; int SHOW_STEP_2 = 4;
	//SHOW_STEP->0:not show anything, >4:Step4 only
	//1:Step1, 2:Step2, 3:Step3, 4:Step4, 
	/*
	//compare one Data Sample
	string DATA_1 = _setPerson(PERSON_1, SAMPLE_1);
	string DATA_2 = _setPerson(PERSON_2, SAMPLE_2);
	Mat descriptors = _preTreat(DATA_1, SHOW_STEP_1);
	Mat descriptors2 = _preTreat(DATA_2, SHOW_STEP_2);
	_compareData(descriptors, descriptors2, DATA_1 + " & " + DATA_2);
	*/
	//compare total Data Sample
	PERSON_2 = 2;
	//_comparePerson(PERSON_1, PERSON_2, 5);
	
	PERSON_2 = 3;
	//_comparePerson(PERSON_1, PERSON_2, 5);

	PERSON_2 = 4;
	//_comparePerson(PERSON_1, PERSON_2, 5);

	waitKey(0);
	cout<< endl <<"Press any key to end..."<<endl;
	cin.get();
    return 0;	
}
