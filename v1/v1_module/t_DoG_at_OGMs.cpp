//#define OGMs
#ifdef OGMs
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<stdlib.h>

#define Pi 3.14159265359

using namespace cv;
using namespace std;

Mat src, src_gray;
int kernel_12[9] = { -1,0,1,-1,0,1,-1,0,1 };
int kernel_34[9] = { -1,-1,-1,0,0,0,1,1,1 };
int kernel_56[9] = { 1,0,-1,1,0,-1,1,0,-1 };
int kernel_70[9] = { 1,1,1,0,0,0,-1,-1,-1 };
int thresh = 160;

double gaussian_kernel[5][5];
double gaussian_kernel_k[5][5];

int main() {
	printf("Oriented Gradient Maps (OGMs)\n");

	src = imread("palm_1.png", IMREAD_COLOR); 
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	

	int scale = 3;
	int kscale;
	kscale = (int)scale * 1.6;
	if (kscale % 2 == 0)
		kscale++;


	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		//cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}
	imshow("original", src);


	// create gaussian kernel
	double r, s = (double)(2.0 * scale * scale);  // Assigning standard deviation to 1.0
	double sum = 0.0;   // Initialization of sun for normalization
	int x, y;

	for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
	{
		for (y = -2; y <= 2; y++)
		{
			r = sqrt(x*x + y * y);
			gaussian_kernel[x + 2][y + 2] = (exp(-(r*r) / s)) / (Pi * s);
			sum += gaussian_kernel[x + 2][y + 2];
		}
	}
	for (int i = 0; i < 5; ++i) // Loop to normalize the kernel
		for (int j = 0; j < 5; ++j)
			gaussian_kernel[i][j] /= sum;

	// create k gaussian kernel
	s = (double)(2.0 * kscale * kscale);
	sum = 0.0;
	for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
	{
		for (y = -2; y <= 2; y++)
		{
			r = sqrt(x*x + y * y);
			gaussian_kernel_k[x + 2][y + 2] = (exp(-(r*r) / s)) / (Pi * s);
			sum += gaussian_kernel_k[x + 2][y + 2];
		}
	}
	for (int i = 0; i < 5; ++i) // Loop to normalize the kernel
		for (int j = 0; j < 5; ++j)
			gaussian_kernel_k[i][j] /= sum;


	int height = src.rows;
	int width = src.cols;
	int I = 0;


	Mat dst = Mat::zeros(src.size(), CV_32FC1);
	Mat dst_gaussian = Mat::zeros(src.size(), CV_32FC1);
	Mat dst_kgaussian = Mat::zeros(src.size(), CV_32FC1);


	// create OGMs
	/* dir 1,3,5,7 */
	for (int j = 2; j < height-2; j++) {
		for (int i = 2; i < width-2; i++) {
			for(int m=0;m<9;m++){
				x = i - 2 + m / 3 + m % 3;
				y = j + m / 3 - m % 3;
				I+=(src_gray.at<uchar>(y, x)*kernel_56[m]);
			}
			//printf("%d\t", I);
			if (I < 0) I = 0;
			dst.at<float>(j, i) = (float)I;
			I = 0;
		}
	}
	/* dir 0,2,4,6
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			for (int m = 0; m < 9; m++) {
				x = i + m % 3 - 1;
				y = j + m / 3 - 1;
				I += (src_gray.at<uchar>(y, x)*kernel_56[m]);
			}
			if (I < 0) I = 0;
			dst.at<float>(j, i) = (float)I;
			I = 0;
		}
	}
	*/


	// normalize OGMs
	Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);


	double G = 0.0;
	// gaussian filtering
	for (int j = 2; j < height-2; j++) {
		for (int i = 2; i < width-2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)dst.at<float>(j + y,i+x)*gaussian_kernel[x + 2][y + 2]);
			dst_gaussian.at<float>(j, i) = (float)G;
			G = 0.0;
		}
	}
	// k_gaussian filtering
	for (int j = 2; j < height - 2; j++) {
		for (int i = 2; i < width - 2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)dst.at<float>(j + y, i + x)*gaussian_kernel_k[x + 2][y + 2]);
			dst_kgaussian.at<float>(j, i) = (float)G;
			G = 0.0;
		}
	}

	

	// DoG at OGMs
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			dst.at<float>(j, i) = dst_gaussian.at<float>(j, i) - dst_kgaussian.at<float>(j, i);
		}
	}


	// normalize DoG at OGMs
	Mat dog_norm, dog_norm_scaled;
	normalize(dst, dog_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dog_norm, dog_norm_scaled);


	int num = 0,temp;
	// print DoG at OGMs
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			temp = dog_norm_scaled.at<uchar>(j, i);
			//printf("%d\t", temp);
			if (temp > thresh) {
				num++;
				circle(dog_norm_scaled, Point(i, j), scale, Scalar(0, 0, 255), 2, 8, 0);
				circle(src, Point(i, j), scale, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	printf("%d", num);


	imshow("OGMs", dst_norm_scaled);
	imshow("DoG at OGMs", dog_norm_scaled);
	imshow("red circle", src);

	waitKey();
	return 0;
}

#endif