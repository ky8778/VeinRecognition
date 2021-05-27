#define multilevel_detection
#ifdef multilevel_detection

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

Mat src, src_gray, laplaceimg;
Mat src_prev_gaussian, src_next_gaussian, src_prev_gray, src_next_gray;

int *Ix, *Iy, *Ixx, *Iyy, *Ixy;
int *laplace;
int *prev_laplace, *next_laplace, *max_laplace_point;
int scale = 3;
int prev_scale = 1;
int next_scale = 5;
int height, width;

double k = 0.04;
int kernel_x[9] = { -1,0,1,-1,0,1,-1,0,1 };
int kernel_y[9] = { -1,-1,-1,0,0,0,1,1,1 };
int kernel_12[9] = { -1,0,1,-1,0,1,-1,0,1 };
int kernel_34[9] = { -1,-1,-1,0,0,0,1,1,1 };
int kernel_56[9] = { 1,0,-1,1,0,-1,1,0,-1 };
int kernel_70[9] = { 1,1,1,0,0,0,-1,-1,-1 };
double gaussian_kernel[5][5];
double gaussian_kernel_k[5][5];

// 1 : 35, 115
// 2 : 70, 128
// 3 : 44, 133
// 4 : 109 , 121
// 5 : 60, 108
// 6 : 90, 145
int harris_thresh = 44;
int hessian_thresh = 133;
// 3-0 : 130
// 3-1 : 103
// 3-2 : 133
// 3-3 : 140
// 3-4 : 133
// 3-5 : 146
// 3-6 : 150
// 3-7 : 150
int DoG_thresh = 150;

void laplacian(Mat gray, int s, int *dst);
void norm(int *arr, int size);
void hessian(Mat dst);
void Harris_D(Mat dst);
void DoG_at_OGMs(Mat dog,Mat ogm);

const char* source0 = "l_03.jpg";
const char* source1 = "laplace_l_03.jpg";
//const char* source2= "harris_l_01.jpg";
//const char* source3= "hessian_l_01.jpg";
const char* source2 = "harris_laplace_l_03.jpg";
const char* source3 = "hessian_laplace_l_03.jpg";
const char* source4 = "DoG_at_OGMs_03_6.jpg";
const char* source5 = "OGMs_red_circle_03_6.jpg";
const char* source6 = "OGMs_03_6.jpg";
const char* source7 = "DoG_red_circle_03_6.jpg";


int main()
{
	src = imread(source0, IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		//cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}
	imshow("original", src);

	src_prev_gaussian = src.clone();
	src_next_gaussian = src.clone();

	// Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
	GaussianBlur(src, src, Size(3, 3), scale, scale, BORDER_DEFAULT);
	GaussianBlur(src, src_prev_gaussian, Size(3, 3), prev_scale, prev_scale);
	GaussianBlur(src, src_next_gaussian, Size(3, 3), next_scale, next_scale);

	height = src.rows;
	width = src.cols;

	src_gray = src.clone();
	src_prev_gray = src_prev_gaussian.clone();
	src_next_gray = src_next_gaussian.clone();

	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	cvtColor(src_prev_gaussian, src_prev_gray, COLOR_BGR2GRAY);
	cvtColor(src_next_gaussian, src_next_gray, COLOR_BGR2GRAY);

	laplaceimg = src_gray.clone();

	Ix = (int*)calloc(height*width, sizeof(int));
	Ixx = (int*)calloc(height*width, sizeof(int));
	Iy = (int*)calloc(height*width, sizeof(int));
	Iyy = (int*)calloc(height*width, sizeof(int));
	Ixy = (int*)calloc(height*width, sizeof(int));
	laplace = (int*)calloc(height*width, sizeof(int));
	prev_laplace = (int*)calloc(height*width, sizeof(int));
	next_laplace = (int*)calloc(height*width, sizeof(int));
	max_laplace_point = (int*)calloc(height*width, sizeof(int));

	laplacian(src_prev_gray, prev_scale, prev_laplace);
	laplacian(src_next_gray, next_scale, next_laplace);
	laplacian(src_gray, scale, laplace);

	norm(laplace, height*width);
	norm(prev_laplace, height*width);
	norm(next_laplace, height*width);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			laplaceimg.at<uchar>(j, i) = laplace[j*width + i];
		}
	}
	imshow("laplace", laplaceimg);

	/*
	// Declare the variables we are going to use
	Mat la_dst;
	int kernel_size = 3;
	int s = 1;
	int delta = 0;
	int ddepth = CV_16S;

	Mat abs_dst;
	Laplacian(src_gray, la_dst, ddepth, kernel_size, s, delta, BORDER_DEFAULT);

	// converting back to CV_8U
	convertScaleAbs(la_dst, abs_dst);

	//imshow("la", la_dst);
	imshow("la", abs_dst);
	*/

	int num_maxpoint = 0;

	// max laplace point
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			if (laplace[j*width + i] > prev_laplace[j*width + i]
				&& laplace[j*width + i] > next_laplace[j*width + i]) {
				max_laplace_point[j*width + i] = 1;
				num_maxpoint++;
				//printf("x : %d, y : %d\n", i, j);
			}
		}
	}
	printf("%d\n", num_maxpoint);

	Scalar c;
	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;
	Point pCenter;
	int radius = 1;
	int num = 0;


	// harris detect
	Mat src2 = src.clone();
	printf("harris\n");
	Mat harris_dst = Mat::zeros(src.size(), CV_32FC1);
	Harris_D(harris_dst);

	//int blockSize = 2;
	//int apertureSize = 3;
	//cornerHarris(src_gray, harris_dst, blockSize, apertureSize, k);

	Mat harris_norm, harris_norm_scaled;
	normalize(harris_dst, harris_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(harris_norm, harris_norm_scaled);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			//printf("%f\t", harris_dst.at<float>(j, i));
			//printf("%f\t", harris_norm.at<float>(j, i));
			//printf("%d\t", harris_norm_scaled.at<uchar>(j, i));
			//if ((int)harris_dst.at<float>(j, i) > harris_thresh || (int)harris_dst.at<float>(j,i)<(0-harris_thresh)) {
			//if (harris_norm_scaled.at<uchar>(j, i) > harris_thresh) {
			if (harris_norm_scaled.at<uchar>(j, i) > harris_thresh && max_laplace_point[j*width + i] == 1) {
				pCenter.x = i;
				pCenter.y = j;
				circle(src2, pCenter, radius, c, 2, 8, 0); // red circle
				num++;
			}
		}
	}
	printf("harris number : %d\n", num);
	imshow("harris", src2);
	//imshow("harris_map", harris_norm_scaled);
	num = 0;


	// hessian detect
	Mat src1 = src.clone();
	printf("hessian\n");
	Mat hessian_dst = Mat::zeros(src.size(), CV_32FC1);
	hessian(hessian_dst);

	Mat hessian_norm, hessian_norm_scaled;
	normalize(hessian_dst, hessian_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(hessian_norm, hessian_norm_scaled);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			//printf("%f\t", hessian_dst.at<float>(j, i));
			//printf("%f\t", hessian_norm.at<float>(j, i));
			//printf("%d\t", hessian_norm_scaled.at<uchar>(j, i));
			//if ((int)hessian_dst.at<float>(j, i) > 1500 || (int)hessian_dst.at<float>(j, i) < -1500) {
			//if (hessian_norm_scaled.at<uchar>(j, i) > hessian_thresh) {
			if (hessian_norm_scaled.at<uchar>(j, i) > hessian_thresh&& max_laplace_point[j*width + i] == 1) {
				pCenter.x = i;
				pCenter.y = j;
				circle(src1, pCenter, radius, c, 2, 8, 0); // red circle
				num++;
			}
		}
	}
	printf("hessian number : %d\n", num);
	imshow("hessian", src1);
	//imshow("hessian_map", hessian_norm_scaled);
	num = 0;


	//DoG at OGMs
	Mat src3 = src.clone();
	Mat DoG_dst = Mat::zeros(src.size(), CV_8UC1);
	Mat OGMs_dst = Mat::zeros(src.size(), CV_8UC1);
	DoG_at_OGMs(DoG_dst,OGMs_dst);
	
	Mat circle_OGMs,circle_DoG;
	cvtColor(OGMs_dst, circle_OGMs, COLOR_GRAY2BGR);
	cvtColor(DoG_dst, circle_DoG, COLOR_GRAY2BGR);

	int temp;
	// print DoG at OGMs
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			temp = DoG_dst.at<uchar>(j, i);
			//printf("%d\t", temp);
			if (temp > DoG_thresh) {
				num++;
				pCenter.x = i;
				pCenter.y = j;
				circle(circle_OGMs, pCenter, radius, c, 2, 8, 0); // red circle
				circle(circle_DoG, pCenter, radius, c, 2, 8, 0); // red circle
			}
		}
	}
	printf("DoG at OGMs number : %d\n", num);
	num = 0;
	imshow(source4, DoG_dst);
	imshow(source5, circle_OGMs);
	imshow(source6, OGMs_dst);
	imshow(source7, circle_DoG);

	imwrite(source1, laplaceimg);	//laplaceimg
	imwrite(source2, src1);			// harrisimg circle
	imwrite(source3, src2);			// hessianimg circle
	imwrite(source4, DoG_dst);		// DoGimg
	imwrite(source5, circle_OGMs);	// DoG at OGMs circle
	imwrite(source6, OGMs_dst);		// OGMsimg
	imwrite(source7, circle_DoG);

	waitKey();
	free(Ix);
	free(Iy);
	free(Ixx);
	free(Iyy);
	free(Ixy);
	free(laplace);
	free(prev_laplace);
	free(next_laplace);
	free(max_laplace_point);

	return 0;
}


void laplacian(Mat gray, int s, int *dst) {
	int Ix_val = 0, Iy_val = 0;
	int x, y;

	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			for (int m = 0; m < 9; m++) {
				x = i + m % 3 - 1;
				y = j + m / 3 - 1;
				Ix_val += (gray.at<uchar>(y, x)*kernel_x[m]);
				Iy_val += (gray.at<uchar>(y, x)*kernel_y[m]);
			}
			Ix[j*width + i] = Ix_val;
			Iy[j*width + i] = Iy_val;
			Ix_val = 0;
			Iy_val = 0;
		}
	}
	int Ixx_val = 0, Iyy_val = 0, Ixy_val = 0;
	int laplace_val;

	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			for (int m = 0; m < 9; m++) {
				x = i + m % 3 - 1;
				y = j + m / 3 - 1;
				Ixx_val += (Ix[y*width + x] * kernel_x[m]);
				Ixy_val += (Ix[y*width + x] * kernel_y[m]);
				Iyy_val += (Iy[y*width + x] * kernel_y[m]);
			}
			Ixx[j*width + i] = Ixx_val;
			Ixy[j*width + i] = Ixy_val;
			Iyy[j*width + i] = Iyy_val;
			laplace_val = abs(Ixx_val + Iyy_val)*s*s;
			dst[j*width + i] = laplace_val;
			Ixx_val = 0;
			Ixy_val = 0;
			Iyy_val = 0;
		}
	}
}
void norm(int *arr, int size) {
	int max = 0, min = 9999, term;
	for (int i = 0; i < size; i++) {
		if (min > arr[i])
			min = arr[i];
		if (max < arr[i])
			max = arr[i];
	}
	term = max - min;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			arr[j*width + i] = (arr[j*width + i] - min) * 255 / term;
			//printf("%d\t", arr[j*width+i]);
		}
	}
}
void hessian(Mat dst) {
	int det, trace;
	float har;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			det = Ixx[j*width + i] * Iyy[j*width + i] - Ixy[j*width + i] * Ixy[j*width + i];
			trace = Ixx[j*width + i] + Ixy[j*width + i];
			har = (float)det - k * (float)trace * (float)trace;
			dst.at<float>(j, i) = har;
		}
	}
}
void Harris_D(Mat dst) {
	int det, trace;
	float R;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			det = Ix[j*width + i] * Ix[j*width + i] * Iy[j*width + i] * Iy[j*width + i];
			trace = Ix[j*width + i] * Ix[j*width + i] + Iy[j*width + i] * Iy[j*width + i];
			R = (float)det - k * (float)trace * (float)trace;
			dst.at<float>(j, i) = R;
			//printf("%d\t", (int)output_img.at<float>(j,i));
		}
	}
}
void DoG_at_OGMs(Mat dog,Mat ogm) {
	printf("Oriented Gradient Maps (OGMs)\n");

	Mat ogm_tmp = Mat::zeros(src.size(), CV_32FC1);
	Mat dog_tmp = Mat::zeros(src.size(), CV_32FC1);

	int kscale;
	kscale = (int)scale * 1.6;
	if (kscale % 2 == 0)
		kscale++;

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

	int I = 0;

	//Mat dst_gaussian = Mat::zeros(src.size(), CV_32FC1);
	//Mat dst_kgaussian = Mat::zeros(src.size(), CV_32FC1);

	// create OGMs
	/* dir 1,3,5,7 
	for (int j = 2; j < height - 2; j++) {
		for (int i = 2; i < width - 2; i++) {
			for (int m = 0; m < 9; m++) {
				x = i - 2 + m / 3 + m % 3;
				y = j + m / 3 - m % 3;
				I += (src_gray.at<uchar>(y, x)*kernel_12[m]);
			}
			//printf("%d\t", I);
			if (I < 0) I = 0;
			ogm_tmp.at<float>(j, i) = (float)I;
			I = 0;
		}
	}
	*/
	/* dir 0,2,4,6 */
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			for (int m = 0; m < 9; m++) {
				x = i + m % 3 - 1;
				y = j + m / 3 - 1;
				I += (src_gray.at<uchar>(y, x)*kernel_56[m]);
			}
			if (I < 0) I = 0;
			ogm_tmp.at<float>(j, i) = (float)I;
			I = 0;
		}
	}
	

	// normalize OGMs
	Mat ogm_norm;
	normalize(ogm_tmp, ogm_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(ogm_norm, ogm);

	double G = 0.0;
	/*
	// gaussian filtering
	for (int j = 2; j < height - 2; j++) {
		for (int i = 2; i < width - 2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)ogm.at<float>(j + y, i + x)*gaussian_kernel[x + 2][y + 2]);
			dst_gaussian.at<float>(j, i) = (float)G;
			G = 0.0;
		}
	}
	// k_gaussian filtering
	for (int j = 2; j < height - 2; j++) {
		for (int i = 2; i < width - 2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)ogm.at<float>(j + y, i + x)*gaussian_kernel_k[x + 2][y + 2]);
			dst_kgaussian.at<float>(j, i) = (float)G;
			G = 0.0;
		}
	}
	// DoG at OGMs
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			dog_tmp.at<float>(j, i) = dst_gaussian.at<float>(j, i) - dst_kgaussian.at<float>(j, i);
		}
	}
	*/
	// DoG at OGMs
	for (int j = 2; j < height - 2; j++) {
		for (int i = 2; i < width - 2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)ogm.at<uchar>(j + y, i + x)*(gaussian_kernel[x+2][y+2]-gaussian_kernel_k[x + 2][y + 2]));
			dog_tmp.at<float>(j, i) = (float)G;
			G = 0.0;
		}
	}

	// normalize DoG at OGMs
	Mat dog_norm;
	normalize(dog_tmp, dog_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dog_norm, dog);
}

#endif