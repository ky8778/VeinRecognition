#define DEBUG
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

#pragma region typedefs
typedef struct Cpoint{
	int x;
	int y;
}cpoint;
#pragma endregion

#pragma region constants
//const int binarization_threshold = 
const float Pi = 3.14159265359;
const double k = 0.04;
const int morpho_kernel[9] = { 1,1,1,1,1,1,1,1,1 };
const int kernel_x[9] = { -1,0,1,-1,0,1,-1,0,1 };
const int kernel_y[9] = { -1,-1,-1,0,0,0,1,1,1 };
const int kernel_12[9] = { -1,0,1,-1,0,1,-1,0,1 };
const int kernel_34[9] = { -1,-1,-1,0,0,0,1,1,1 };
const int kernel_56[9] = { 1,0,-1,1,0,-1,1,0,-1 };
const int kernel_70[9] = { 1,1,1,0,0,0,-1,-1,-1 };
const int harris_thresh = 44;
const int hessian_thresh = 133;
const int DoG_thresh = 150;
#pragma endregion

#pragma region globals
Mat = src, src_prev_gaussian, src_next_gaussian;
Mat = src_gray, src_bi, src_prev_gray, src_next_gray, src_laplace;
int width, height, bi_thresh;
int *Ix, *Iy, *Ixx, *Iyy, *Ixy;
int *laplace, *prev_laplace, *next_laplace, *max_laplace_point;
int scale = 3, prev_scale = 1, next_scale = 5;
double gaussian_kernel[5][5];
double gaussian_kernel_k[5][5];
#pragma endregion

#pragma region functions
void Initialize(){
	src = imread("ex.jpg", IMREAD_COLOR);
	if(src.empty()){
		cout << "Could not open or find source image.\n" << endl;
		return -1;
	}
	height = src.rows;
	width = src.cols;

	src_prev_gaussian = Mat(height, width, CV_8UC3);
	src_next_gaussian = Mat(height, width, CV_8UC3);
	src_gray = Mat(height, width, CV_8UC1);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	src_prev_gray = Mat(height, width, CV_8UC1);
	src_next_gray = Mat(height, width, CV_8UC1);
	src_bi = Mat(height, width, CV_8UC1);
	src_laplace = Mat(height, width, CV_8UC1);

	Ix = (int*)calloc(height*width, sizeof(int));
	Ixx = (int*)calloc(height*width, sizeof(int));
	Iy = (int*)calloc(height*width, sizeof(int));
	Iyy = (int*)calloc(height*width, sizeof(int));
	Ixy = (int*)calloc(height*width, sizeof(int));
	laplace = (int*)calloc(height*width, sizeof(int));
	prev_laplace = (int*)calloc(height*width, sizeof(int));
	next_laplace = (int*)calloc(height*width, sizeof(int));
	max_laplace_point = (int*)calloc(height*width, sizeof(int));
	CreateGaussianKernel();
}

void OtsuBinarization(Mat img){
	float bi_sigma, bi_sigma_min = 9999999999999;
	float bi_mean_b = 0, bi_mean_w = 0;
	float bi_variance_b = 0, bi_variance_w = 0;
	float num_b=0, num_w = 0;
	int gray_value;

	for(int bi_thresh_temp = 60; bi_thresh_temp < 90; bi_thresh_temp += 2){
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
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
		bi_mean_b = 0;
		bi_mean_w = 0;
		bi_variance_b = 0;
		bi_variance_w = 0;
		num_b = 0;
		num_w = 0;
	}

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
}

void erosion(Mat img) {
	Mat result = Mat::zeros(height, width, CV_32FC1);
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

void MakeMorphologicalImage(Mat img){
	// opening
	erosion(src_bi);
	dilation(src_bi);
	// closing
	dilation(src_bi);
	erosion(src_bi);
}

void CalcPalmCentroid(Mat img, cpoint *p) {
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
}

int MaximumInscribedCircle(Mat img, cpoint p) {
	float radius = 0., radius_min = 99999999999.;
	int x, y, t=0;
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
			radius = (float)sqrt(pow(i - x, 2) + pow(j - y, 2););
			if (radius < radius_min){
				radius_min = radius;
			}
			go = 1;
			t = 0;
		}
	}
	return radius_min;
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
		}
	}
}

void GetLaplacian(Mat img){
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
}

void GaussianBlurring(Mat img){
	GaussianBlur(src, src, Size(3, 3), scale, scale, BORDER_DEFAULT);
	GaussianBlur(src, src_prev_gaussian, Size(3, 3), prev_scale, prev_scale);
	GaussianBlur(src, src_next_gaussian, Size(3, 3), next_scale, next_scale);
	cvtColor(src_prev_gaussian, src_prev_gray, COLOR_BGR2GRAY);
	cvtColor(src_next_gaussian, src_next_gray, COLOR_BGR2GRAY);
}

void GetHarris(Mat dst) {
	int det, trace;
	float R;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			det = Ix[j*width + i] * Ix[j*width + i] * Iy[j*width + i] * Iy[j*width + i];
			trace = Ix[j*width + i] * Ix[j*width + i] + Iy[j*width + i] * Iy[j*width + i];
			R = (float)det - k * (float)trace * (float)trace;
			dst.at<float>(j, i) = R;
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

void CreateGaussianKernel(){
	int _scale = 3;
	int kscale;
	kscale = int(scale * 1.6);
	if (kscale % 2 == 0) kscale++;

	// create gaussian kernel
	double r, s = (double)(2.0 * _scale * _scale);  // Assigning standard deviation to 1.0
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
}

void MakeOGMs(Mat input, Mat ogm){
	// create OGMs
	int I = 0;
	/* dir 1,3,5,7 */
	for (int j = 2; j < height-2; j++) {
		for (int i = 2; i < width-2; i++) {
			for(int m=0;m<9;m++){
				x = i - 2 + m / 3 + m % 3;
				y = j + m / 3 - m % 3;
				I+=(input.at<uchar>(y, x)*kernel_56[m]);	// dir 5
			}
			//printf("%d\t", I);
			if (I < 0) I = 0;
			ogm.at<float>(j, i) = (float)I;
			I = 0;
		}
	}
	/* dir 0,2,4,6
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			for (int m = 0; m < 9; m++) {
				x = i + m % 3 - 1;
				y = j + m / 3 - 1;
				I += (input.at<uchar>(y, x)*kernel_56[m]); // dir 6
			}
			if (I < 0) I = 0;
			ogm.at<float>(j, i) = (float)I;
			I = 0;
		}
	}
	*/

	// normalize OGMs
	normalize(ogm, ogm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(ogm, ogm);
}

void DoGAtOGMs(Mat ogm, Mat dst) {
	printf("Oriented Gradient Maps (OGMs)\n");
	
	// DoG at OGMs
	double G = 0.0;
	for (int j = 2; j < height - 2; j++) {
		for (int i = 2; i < width - 2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)ogm.at<uchar>(j + y, i + x)*(gaussian_kernel[x+2][y+2]-gaussian_kernel_k[x + 2][y + 2]));
			dst.at<float>(j, i) = (float)G;
			G = 0.0;
		}
	}

	// normalize DoG at OGMs
	normalize(dog, dog, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dog, dog);
}

void MultilevelDetection(){
	Point pCenter;
	int radius = 1;

	#pragma region Harris Detect
	// harris detect
	Mat harris_dst = Mat::zeros(src.size(), CV_32FC1);
	
	/* my function */
	GetHarris(harris_dst);
	/* opencv
	int blockSize = 2;
	int apertureSize = 3;
	cornerHarris(src_gray, harris_dst, blockSize, apertureSize, k);
	*/

	Mat harris_norm, harris_norm_scaled;
	normalize(harris_dst, harris_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(harris_norm, harris_norm_scaled);

	Mat src_harris = src.clone();
	int harris_cnt = 0;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			if (harris_norm_scaled.at<uchar>(j, i) > harris_thresh && max_laplace_point[j*width + i] == 1) {
				pCenter.x = i;
				pCenter.y = j;
				circle(src_harris, pCenter, radius, c, 2, 8, 0); // red circle
				harris_cnt++;
			}
		}
	}
	#if DEBUG
	imshow("Harris Detect Result", src_harris);
	waitKey();
	#endif
	#pragma endregion

	#pragma region Hessian Detect
	// hessian detect
	Mat hessian_dst = Mat::zeros(src.size(), CV_32FC1);
	hessian(hessian_dst);

	Mat hessian_norm, hessian_norm_scaled;
	normalize(hessian_dst, hessian_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(hessian_norm, hessian_norm_scaled);

	Mat src_hessian = src.clone();
	int hessian_cnt = 0;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			if (hessian_norm_scaled.at<uchar>(j, i) > hessian_thresh &&  max_laplace_point[j*width + i] == 1) {
				pCenter.x = i;
				pCenter.y = j;
				circle(src_hessian, pCenter, radius, c, 2, 8, 0); // red circle
				hessian_cnt++;
			}
		}
	}
	#if DEBUG
	imshow("Hessian Detect Result", src_hessian);
	waitKey();
	#endif
	#pragma endregion

	#pragma region DoG at OGMs
	//DoG at OGMs
	Mat DoG_dst = Mat::zeros(src.size(), CV_8UC1);
	Mat OGMs_dst = Mat::zeros(src.size(), CV_8UC1);
	MakeOGMs(src_gray, OGMs_dst);
	DoGAtOGMs(OGMs_dst, DoG_dst);
	
	#if DEBUG
	Mat circle_OGMs,circle_DoG;
	cvtColor(OGMs_dst, circle_OGMs, COLOR_GRAY2BGR);
	cvtColor(DoG_dst, circle_DoG, COLOR_GRAY2BGR);
	#endif

	int DoG_cnt = 0;
	// print DoG at OGMs
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			if (DoG_dst.at<uchar>(j, i); > DoG_thresh) {
				DoG_cnt++;
				#if DEBUG
				pCenter.x = i;
				pCenter.y = j;
				circle(circle_OGMs, pCenter, radius, c, 2, 8, 0); // red circle
				circle(circle_DoG, pCenter, radius, c, 2, 8, 0); // red circle
				#endif
			}
		}
	}
	#pragma endregion
}

void Free(){
	free(Ix);
	free(Iy);
	free(Ixx);
	free(Iyy);
	free(Ixy);
	free(laplace);
	free(prev_laplace);
	free(next_laplace);
	free(max_laplace_point);
}
#pragma endregion

int main(){
	Initialize();
	#if DEBUG
	imshow("original", src);
	imshow("gray", src_gray);
	waitKey();
	#endif

	// ROI Extraction
	// 1. Binarization
	OtsuBinarization(src_bi);				// v3. Otsu's Binarization
	#if DEBUG
	imshow("binarization", src_bi);
	imwrite("binarization.jpg", src_bi);
	waitKey();
	#endif

	// 2. Morphological image processing		
	MakeMorphologicalImage(src_bi);			// v1. Morphological Image
	#if DEBUG
	imshow("Morphological Image", src_bi);
	imwrite("morphological_image.jpg", src_bi);
	waitKey();
	#endif

	// 3. Extraction Algorithm
	cpoint center_point;
	CalcPalmCentroid(src_bi, &center_point);
	#if DEBUG
	Mat src_copy = src.clone();
	cvtColor(src_bi, src_copy, COLOR_GRAY2BGR);
	redScalar = Scalar(0, 0, 255);
	Point pCenter;
	pCenter.x = p.x;
	pCenter.y = p.y;
	
	int radius = 1;
	circle(src_copy, pCenter, radius, redScalar, 2, 8, 0); // red circle
	for (int i = -50; i < 50; i++) {
		pCenter.x = p.x+i;
		pCenter.y = p.y-50;
		circle(src_copy, pCenter, radius, redScalar, 2, 8, 0); // red circle
		pCenter.x = p.x + i;
		pCenter.y = p.y + 50;
		circle(src_copy, pCenter, radius, redScalar, 2, 8, 0); // red circle
		pCenter.x = p.x - 50;
		pCenter.y = p.y + i;
		circle(src_copy, pCenter, radius, redScalar, 2, 8, 0); // red circle
		pCenter.x = p.x + 50;
		pCenter.y = p.y + i;
		circle(src_copy, pCenter, radius, redScalar, 2, 8, 0); // red circle
	}
	imshow("Center Point", src_copy);
	imwrite("centroid_point.jpg", src_copy);
	waitKey();
	#endif
	max_radius = MaximumInscribedCircle(src_bi, center_point);
	//TODO Extraction

	//TODO 4. Image Pre-processing

	// Description
	GaussianBlurring(src);
	GetLaplacian(src);
	#if DEBUG
	imshow("laplace", laplaceimg);
	waitKey();
	#endif
	MultilevelDetection();
	
	// Matching

	Free();
	return 0;
}