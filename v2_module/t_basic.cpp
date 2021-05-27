//#define basic
#ifdef basic
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

using namespace cv;

int main() {
	// You can set the path of an image file
	Mat imgColor = imread("test.jpg", IMREAD_COLOR);
	Mat imgGray = imread("test.jpg", IMREAD_GRAYSCALE);

	printf("%d %d", imgColor.rows, imgColor.cols);
	int x = 390;
	int y = 10;
	int rVal, gVal, bVal;
	rVal = imgColor.at<Vec3b>(y, x)[2];
	gVal = imgColor.at<Vec3b>(y, x)[1];
	bVal = imgColor.at<Vec3b>(y, x)[0];
	int grayVal;
	grayVal = imgGray.at<uchar>(y, x);

	int height = imgColor.rows;
	int width = imgColor.cols;

	Mat result(height, width, CV_8UC1);		// for gray-scale(one channel, unsigned char)
	Mat colors(height, width, CV_8UC3);		// for color(three channels, unsigned char)

	// if you want to fill your Mat object with 0 or 1
	Mat img0 = Mat::zeros(height, width, CV_32FC1);		// one channel, float type
	Mat img1 = Mat::ones(height, width, CV_64FC3);		// three channels, double type

	// if you want to fill your Mat with a specific value
	Mat img2(height, width, CV_8UC1);
	img2 = Scalar(39);

	// Case 1: referencing 
	Mat img_ref = imgColor;
	// Case 2: copy
	Mat img_copy1 = imgColor.clone();
	// another way to copy
	Mat img_copy2;
	imgColor.copyTo(img_copy2);

	imshow("color", imgColor);
	//imshow("gray", imgGray);
	//imshow("img0", img0);
	//imshow("img1", img1);
	//imshow("img2", img2);
	imshow("img_ref", img_ref);
	imshow("img_copy1", img_copy1);
	imshow("img_copy2", img_copy2);
	imwrite("result.bmp", imgColor);
	waitKey(5000);
	return 0;
}

#endif