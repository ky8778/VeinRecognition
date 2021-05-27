#define ROI
#ifdef ROI

#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

typedef struct Cpoint{
	int x;
	int y;
}cpoint;

Mat src, src_gray;
int width, height;
int bi_thresh;

const char* source0 = "ex4.jpg";
int morpho_kernel[9] = { 1,1,1,1,1,1,1,1,1 };
void erosion(Mat img);
void dilation(Mat img);
void centroid_calculation(Mat img, cpoint *p);
//void maximum_inscribed_circle(Mat img, cpoint *p);

int main() {
	src = imread(source0, IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return -1;
	}
	height = src.rows;
	width = src.cols;

	imshow("original", src);

	src_gray = src.clone();
	cvtColor(src, src_gray, COLOR_BGR2GRAY);


	// Otsu's method binarization
	Mat src_bi(height, width, CV_8UC1);
	int bi_loop = 0;
	float bi_sigma, bi_sigma_min=9999999999999;
	float bi_mean_b = 0, bi_mean_w = 0;
	float bi_variance_b = 0, bi_variance_w = 0;
	float num_b=0, num_w = 0;
	int gray_value;

	for(int bi_thresh_temp = 60;bi_thresh_temp<90;bi_thresh_temp+=2){
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				//printf("%d\t", src_gray.at<uchar>(j, i));
				gray_value = src_gray.at<uchar>(j, i);
				if (gray_value > bi_thresh_temp){
					src_bi.at<uchar>(j, i) = 255;
					bi_mean_w += (float)gray_value;
					bi_variance_w += powf(gray_value,2);
					num_w++;
				}
				else{
					src_bi.at<uchar>(j, i) = 0;
					bi_mean_b += (float)gray_value;
					bi_variance_b += powf(gray_value, 2);
					num_b++;
				}
			}
		}
		bi_mean_b /= num_b;
		bi_mean_w /= num_w;
		bi_variance_b -= powf(bi_mean_b, 2);
		bi_variance_w -= powf(bi_mean_w, 2);
		bi_sigma = (bi_variance_b * num_b +bi_variance_w*num_w)/(float)(height*width);
		if (bi_sigma_min > bi_sigma){
			bi_sigma_min = bi_sigma;
			bi_thresh = bi_thresh_temp;
		}
		//printf("%f\t", bi_sigma_min);
		bi_mean_b = 0;
		bi_mean_w = 0;
		bi_variance_b = 0;
		bi_variance_w = 0;
		num_b = 0;
		num_w = 0;
	}
	printf("%d", bi_thresh);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			//printf("%d\t", src_gray.at<uchar>(j, i));
			gray_value = src_gray.at<uchar>(j, i);
			if (gray_value > bi_thresh)
				src_bi.at<uchar>(j, i) = 255;
			else
				src_bi.at<uchar>(j, i) = 0;
		}
	}
	Mat src_bi_temp = src_bi.clone();
	imshow("src_bi", src_bi_temp);


	// morphological image processing
	// opening
	erosion(src_bi);
	dilation(src_bi);
	// closing
	dilation(src_bi);
	erosion(src_bi);
	imshow("src_bi_morpho", src_bi);


	// Palm-centroid calculation
	cpoint p;
	centroid_calculation(src_bi,&p);
	Mat src_copy = src.clone();
	cvtColor(src_bi, src_copy, COLOR_GRAY2BGR);
	
	Scalar c;
	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;
	Point pCenter;
	int radius = 1;

	pCenter.x = p.x;
	pCenter.y = p.y;
	circle(src_copy, pCenter, radius, c, 2, 8, 0); // red circle
	for (int j = -50; j < 50; j++) {
		pCenter.x = p.x+j;
		pCenter.y = p.y-50;
		circle(src_copy, pCenter, radius, c, 2, 8, 0); // red circle
		pCenter.x = p.x + j;
		pCenter.y = p.y + 50;
		circle(src_copy, pCenter, radius, c, 2, 8, 0); // red circle
		pCenter.x = p.x - 50;
		pCenter.y = p.y + j;
		circle(src_copy, pCenter, radius, c, 2, 8, 0); // red circle
		pCenter.x = p.x + 50;
		pCenter.y = p.y + j;
		circle(src_copy, pCenter, radius, c, 2, 8, 0); // red circle
	}
	imshow("centerpoint", src_copy);
	imwrite("binary1.jpg", src_bi_temp);
	imwrite("binary2.jpg", src_bi);
	imwrite("centroid.jpg", src_copy);
	//maximum_inscribed_circle(src_bi, &p);

	waitKey();
	return 0;
}

void erosion(Mat img) {
	Mat result = Mat::zeros(height,width, CV_32FC1);
	int check = 0;
	for (int j = 1; j < height-1; j++) {
		for (int i = 1; i < width-1; i++) {
			// erosion
			if (img.at<uchar>(j, i) == 255) {
				for (int m = 0; m < 9; m++)
					check += (img.at<uchar>(j + m / 3 - 1, i + m % 3 - 1)*morpho_kernel[m] / 255);
				if (check == 9)							// need to change if kernel is changed
					result.at<float>(j, i) = 255;
				else
					result.at<float>(j, i) = 0;
			}
			else
				result.at<float>(j, i) = 0;
			check = 0;
		}
	}
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			img.at<uchar>(j, i) = (int)result.at<float>(j, i);
			//printf("%d\t", result.at<uchar>(j, i));
		}
	}
}

void dilation(Mat img) {
	Mat result = Mat::zeros(height, width, CV_32FC1);
	for (int j = 1; j < height-1 ; j++) {
		for (int i = 1; i < width-1 ; i++) {
			// dilation
			if (img.at<uchar>(j, i) == 255) {
				for (int m = 0; m < 9; m++)			// need to change if kernel is changed
					result.at<float>(j + m / 3 - 1, i + m % 3 - 1) = 255;
			}
			else if (result.at<float>(j, i) != 255)
				result.at<float>(j, i) = 0;
		}
	}
	for (int j = 0; j < height; j++){
		for (int i = 0; i < width; i++){
			img.at<uchar>(j, i) = (int)result.at<float>(j, i);
		}
	}
}

void centroid_calculation(Mat img, cpoint *p) {
	int m00=0, m01=0, m10=0;
	int value;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			value = img.at<uchar>(j, i);
			m00 += value/255;
			m01 += (value*j/255);
			m10 += (value*i/255);
		}
	}
	p->x = m10 / m00;
	p->y = m01 / m00;
	
	//printf("\n%d %d %d\n", m00, m10, m01);
	//printf("x : %d, y : %d\n", p->x, p->y);
	//printf("int %d float %f\n", m01/m00,(float)m01/(float)m00);
}

/*
void maximum_inscribed_circle(Mat img, cpoint *p) {
	int distance;
	float r,r_min=9999999999;
	int result_x, result_y;
	int x, y,t=0;
	int go = 1;
	for (int j = p->y - 50; j < p->y + 50; j++) {
		for (int i = p->x - 50; i < p->x + 50; i++) {
			while (go){
				t++;
				for (int m = -t; m < t; j++) {
					if (img.at<uchar>(i + t, j + m) == 0) {
						go = 0;
						x = i + t;
						y = j + m;
						break;
					}
					if (img.at<uchar>(i + m, j + t) == 0) {
						go = 0;
						x = i + m;
						y = j + t;
						break;
					}
					if (img.at<uchar>(i - t, j + m) == 0){
						go = 0;
						x = i - t;
						y = j + m;
						break;
					}
					if (img.at<uchar>(i + m, j - t) == 0){
						go = 0;
						x = i + m;
						y = j - t;
						break;
					}
				}
			}
			distance = pow(i - x, 2) + pow(j - y, 2);
			r = (float)sqrt(distance);
			if (r < r_min){
				r_min = r;

			}
			go = 1;
			t = 0;
		}
	}

	printf("x : %d, y : %d, r : %f", result_x, result_y, (float)sqrt(r_min));
}
*/
#endif