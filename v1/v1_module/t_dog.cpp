//#define DoG
#ifdef DoG
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

Mat src, src_gray, src_kgray, src_gaussian,src_kgaussian;

int main() {
	int scale = 3;
	int kscale;
	kscale = (int)scale * 1.6;
	if (kscale % 2 == 0)
		kscale++;
	src = imread("palm_1.png", IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		//cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}
	imshow("original", src);

	src_gaussian = src.clone();
	src_kgaussian = src.clone();

	GaussianBlur(src, src_gaussian, Size(scale, scale), 0, 0);
	GaussianBlur(src, src_kgaussian, Size(kscale, kscale), 0, 0);
	cvtColor(src_gaussian, src_gray, COLOR_BGR2GRAY);
	cvtColor(src_kgaussian, src_kgray, COLOR_BGR2GRAY);

	//cvtColor(src, src_gray, COLOR_BGR2GRAY);
	
	int height = src.rows;
	int width = src.cols;
	double k = 0.04;
	int num = 0;
	
	
	Mat dst = Mat::zeros(src.size(), CV_8UC1);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			dst.at<uchar>(j, i) = src_kgray.at<uchar>(j, i) - src_gray.at<uchar>(j, i);

			if (dst.at<uchar>(j, i) > 1) {
				num++;
				circle(src, Point(i, j), scale, Scalar(0, 0, 255), 2, 8, 0);
			}
			//printf("%f\t", dst.at<float>(j, i));
		}
	}

	printf("Difference of Gaussian (DoG)\n");
	printf("%d", num);

	imshow("DoG", dst);
	imshow("circle", src);
	
	waitKey();
	return 0;
}

#endif