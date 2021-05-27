#include "b_multilevel_detection.h"

Mat ROI, ROI_gray;				// CV_8UC3, CV_8UC1
Mat OGM_maps[8];				// CV_8UC1
Mat OGM_maps_DoG[8];			// CV_8UC1
Mat OGM_maps_ex[8];
Mat ROI_scale_space[4];			// CV_8UC1
Mat ROI_laplacian[4];			// CV_8UC1
Mat ROI_harris_ex[3];			// extrema scale 1.6, 1.6*1.6
Mat ROI_hessian_ex[3];			// extrema scale 1.6, 1.6*1.6
Mat ROI_harris[2];
Mat ROI_hessian[2];

static Scalar c;
static Point pCenter;
static int width, height;
static int radius;

static int *Ix, *Iy, *Ixx, *Iyy, *Ixy;
static int harris_thresh = 160;
static int hessian_thresh = 160;
static int DoG_thresh = 150;
static int kernel_x[9] = { -1,0,1,-1,0,1,-1,0,1 };
static int kernel_y[9] = { -1,-1,-1,0,0,0,1,1,1 };
static int kernel_12[9] = { -1,0,1,-1,0,1,-1,0,1 };
static int kernel_34[9] = { -1,-1,-1,0,0,0,1,1,1 };
static int kernel_56[9] = { 1,0,-1,1,0,-1,1,0,-1 };
static int kernel_70[9] = { 1,1,1,0,0,0,-1,-1,-1 };
static int laplacian_kernel[3][3] = { {0,-1,0},{-1,4,-1},{0,-1,0} };
static double gaussian_kernel[5][5];
static double gaussian_kernel_k[5][5];

void gaussian_filter(Mat gray, Mat dst, double scale) {
	//printf("\nGaussian Filtering!!\n");
	double max = 0.0;
	Mat tmp(gray.size(), CV_64FC1);
	// create gaussian kernel
	double r, s = (2.0 * scale * scale);  // Assigning standard deviation to 1.0
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

	double G = 0.0;

	// gaussian filtering
	int ii, jj;
	for (int j = 2; j < gray.rows-2; j++) {
		for (int i = 2; i < gray.cols-2; i++) {
			for (x = -2; x <= 2; x++) { // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++) {
					jj = j + y;
					ii = i + x;
					/*
					// treat edge
					if (ii < 0)
						ii += gray.cols;
					if (jj < 0)
						jj += gray.rows;
					if (ii >= gray.cols)
						ii -= gray.cols;
					if (jj >= gray.rows)
						jj -= gray.rows;
					*/
					G += ((double)gray.at<uchar>(jj, ii)*gaussian_kernel[x + 2][y + 2]);
				}
			}
			dst.at<uchar>(j, i) = (int)G;
			G = 0.0;
		}
	}
	//norm(tmp, dst);
}

void OGMs() {
	//printf("\nOriented Gradient Maps (OGMs)\n");
	int x, y, I = 0;
	int *kernel;
	Mat tmp[8];
	for (int i = 0; i < 8; i++) {
		tmp[i] = Mat::zeros(ROI.size(), CV_64FC1);
	}
	// create OGMs
	for (int o = 0; o < 8; o++) {
		// select kernel
		switch (o) {
		case 0:
			kernel = kernel_70;
			break;
		case 1:
			kernel = kernel_12;
			break;
		case 2:
			kernel = kernel_12;
			break;
		case 3:
			kernel = kernel_34;
			break;
		case 4:
			kernel = kernel_34;
			break;
		case 5:
			kernel = kernel_56;
			break;
		case 6:
			kernel = kernel_56;
			break;
		case 7:
			kernel = kernel_70;
			break;
		}

		// dir 1,3,5,7
		if (o % 2 == 1) {
			for (int j = 2; j < ROI.rows - 2; j++) {
				for (int i = 2; i < ROI.cols - 2; i++) {
					for (int m = 0; m < 9; m++) {
						x = i - 2 + m / 3 + m % 3;
						y = j + m / 3 - m % 3;
						I += (ROI.at<uchar>(y, x)*kernel[m]);
					}
					//printf("%d\t", I);
					if (I < 0) I = 0;
					tmp[o].at<double>(j, i) = (double)I;
					I = 0;
				}
			}
		}
		else {
			// dir 0,2,4,6
			for (int j = 1; j < ROI.rows - 1; j++) {
				for (int i = 1; i < ROI.cols - 1; i++) {
					for (int m = 0; m < 9; m++) {
						x = i + m % 3 - 1;
						y = j + m / 3 - 1;
						I += (ROI.at<uchar>(y, x)*kernel[m]);
					}
					if (I < 0) I = 0;
					tmp[o].at<double>(j, i) = (double)I;
					I = 0;
				}
			}
		}
		norm(tmp[o], OGM_maps[o]);
	}
}

void calc_harrihessi(Mat gray, int num) {
	
	Ix = (int*)calloc(ROI.rows*ROI.cols, sizeof(int));
	Iy = (int*)calloc(ROI.rows*ROI.cols, sizeof(int));
	Ixx = (int*)calloc(ROI.rows*ROI.cols, sizeof(int));
	Iyy = (int*)calloc(ROI.rows*ROI.cols, sizeof(int));
	Ixy = (int*)calloc(ROI.rows*ROI.cols, sizeof(int));

	width = ROI.cols;
	height = ROI.rows;
	int Ix_val = 0, Iy_val = 0;
	int x, y;
	int det, trace;
	double harris, hessian;
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

			det = Ix_val * Ix_val * Iy_val * Iy_val;
			trace = Ix_val * Ix_val + Iy_val * Iy_val;
			harris = (double)det - k * (double)trace * (double)trace;
			if (num == 1) {
				ROI_harris[0].at<uchar>(j, i) = (int)harris;
			}
			else if (num == 2) {
				ROI_harris[1].at<uchar>(j, i) = (int)harris;
			}
			Ix_val = 0;
			Iy_val = 0;
		}
	}

	double Ixx_val = 0, Iyy_val = 0, Ixy_val = 0;

	for (int j = 1; j < gray.rows - 1; j++) {
		for (int i = 1; i < gray.cols - 1; i++) {
			for (int m = 0; m < 9; m++) {
				x = i + m % 3 - 1;
				y = j + m / 3 - 1;
				Ixx_val += (Ix[y*width + x] * (double)kernel_x[m]);
				Ixy_val += (Ix[y*width + x] * (double)kernel_y[m]);
				Iyy_val += (Iy[y*width + x] * (double)kernel_y[m]);
			}
			Ixx[j*width + i] = Ixx_val;
			Ixy[j*width + i] = Ixy_val;
			Iyy[j*width + i] = Iyy_val;
			det = Ixx_val * Iyy_val - Ixy_val * Ixy_val;
			trace = Ixx_val + Iyy_val;
			hessian = (double)det - k * (double)trace * (double)trace;
			if (num == 1) {
				ROI_hessian[0].at<uchar>(j, i) = (int)hessian;
			}
			else if (num == 2) {
				ROI_hessian[1].at<uchar>(j, i) = (int)hessian;
			}
			Ixx_val = 0;
			Ixy_val = 0;
			Iyy_val = 0;
		}
	}
	free(Ix);
	free(Iy);
	free(Ixx);
	free(Ixy);
	free(Iyy);
}

void laplacian(Mat gray, Mat dst, double scale) {
	int x, y, i, j;
	double L = 0.0;
	int ii, jj;

	for (j = 1; j < gray.rows-1; j++) {
		for (i = 1; i < gray.cols-1; i++) {
			for (x = -1; x <= 1; x++) { // Loop to generate 5x5 kernel
				for (y = -1; y <= 1; y++) {
					jj = j + y;
					ii = i + x;
					/*
					// treat edge
					if (ii < 0)
						ii += gray.cols;
					if (jj < 0)
						jj += gray.rows;
					if (ii >= gray.cols)
						ii -= gray.cols;
					if (jj >= gray.rows)
						jj -= gray.rows;
					*/
					L += (double)(gray.at<uchar>(jj, ii)*laplacian_kernel[x + 1][y + 1]);
				}
			}
			dst.at<uchar>(j, i) = (int)(L*scale*scale);
			L = 0.0;
		}
	}
}

void laplace_extrema(Mat origin[], Mat dst[]) {
	int current, ii, jj;
	int num = 0, count = 0;
	double scale = 1.6;
	for (int m = 0; m < 2; m++) {
		for (int j = 1; j < ROI.rows - 1; j++) {
			for (int i = 1; i < ROI.cols - 1; i++) {
				current = origin[m + 1].at<uchar>(j, i);
				for (int x = -1; x <= 1; x++) {// Loop to generate 5x5 kernel
					for (int y = -1; y <= 1; y++) {
						// treat edge
						jj = j + y;
						ii = i + x;
						if (ii < 0)
							ii += ROI.cols;
						if (jj < 0)
							jj += ROI.rows;
						if (ii >= ROI.cols)
							ii -= ROI.cols;
						if (jj >= ROI.rows)
							jj -= ROI.rows;

						// compare value
						if (current >= origin[m].at<uchar>(jj, ii))
							num++;
						if (current >= origin[m + 1].at<uchar>(jj, ii) && (x != 0 || y != 0))
							num++;
						if (current >= origin[m + 2].at<uchar>(jj, ii))
							num++;
					}
				}
				if (num == 8) {
					dst[m].at<double>(j, i) = scale;
					count++;
				}
				num = 0;
			}
		}
		scale *= 1.6;
	}
}

void DoG(Mat gray, Mat dst, double scale) {
	//printf("\nDifference of Gaussian!!\n");

	Mat tmp1 = Mat::zeros(ROI.size(), CV_64FC1);
	Mat tmp2 = Mat::zeros(ROI.size(), CV_64FC1);
	Mat tmp = Mat::zeros(ROI.size(), CV_64FC1);

	double kscale;
	kscale = scale * 1.6;

	// create gaussian kernel
	double r, s = 2.0 * scale * scale;  // Assigning standard deviation to 1.0
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

	double G = 0.0;

	// DoG
	for (int j = 2; j < ROI.rows - 2; j++) {
		for (int i = 2; i < ROI.cols - 2; i++) {
			for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
				for (y = -2; y <= 2; y++)
					G += ((double)gray.at<uchar>(j + y, i + x)*(gaussian_kernel[x + 2][y + 2] - gaussian_kernel_k[x + 2][y + 2]));
			tmp.at<double>(j, i) = G;
			//printf("%lf\t", G);
			G = 0.0;
		}
	}
	norm(tmp, dst);
}

void thresholing() {
	Mat print[8];
	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;
	int x, y;
	int harris_count = 0, hessian_count = 0;
	int DoG_count[8] = { 0 };

	/*
	for (int i = 0; i < 8; i++) {
		cvtColor(OGM_maps_DoG[i], print[i], COLOR_GRAY2BGR);
	}
	*/

	for (int j = 0; j < ROI.rows; j++) {
		for (int i = 0; i < ROI.cols; i++) {
			if (ROI_harris[0].at<uchar>(j, i) > harris_thresh && ROI_harris_ex[0].at<double>(j, i) != 0.0){
				ROI_harris_ex[2].at<double>(j, i) = ROI_harris_ex[0].at<double>(j,i);
				harris_count++;
			}
			else if (ROI_harris[1].at<uchar>(j, i) > harris_thresh && ROI_harris_ex[1].at<double>(j, i) != 0.0){
				ROI_harris_ex[2].at<double>(j, i) = ROI_harris_ex[1].at<double>(j, i);
				harris_count++;
			}
			if (ROI_hessian[0].at<uchar>(j, i) > hessian_thresh && ROI_hessian_ex[0].at<double>(j, i) != 0.0){
				ROI_hessian_ex[2].at<double>(j, i) = ROI_hessian_ex[0].at<double>(j, i);
				hessian_count++;
			}
			else if (ROI_hessian[1].at<uchar>(j, i) > hessian_thresh && ROI_hessian_ex[1].at<double>(j, i) != 0.0){
				ROI_hessian_ex[2].at<double>(j, i) = ROI_hessian_ex[1].at<double>(j, i);
				hessian_count++;
			}
			/*
			for (int m = 0; m < 8; m++) {
				if (OGM_maps_DoG[m].at<uchar>(j, i) > DoG_thresh) {
					pCenter.x = i;
					pCenter.y = j;
					circle(print[m], pCenter, radius, c, 2, 8, 0); // red circle
					//printf("%d\t", OGM_maps_DoG[m].at<uchar>(j, i));
					DoG_count[m]++;
					OGM_maps_ex[m].at<double>(j, i) = 1.6;				// scale : 1.6
				}
			}
			*/
		}
	}
	/*
	printf("Harris : %d\nHesian : %d\n", harris_count, hessian_count);
	for (int i = 0; i < 8; i++)
		printf("DoG : %d\n", DoG_count[i]);
	*/
}

void norm(Mat gray, Mat dst) {
	double max = 0.0, min = 999.0, term, tmp;
	for (int j = 0; j < gray.rows; j++) {
		for (int i = 0; i < gray.cols; i++) {
			tmp = gray.at<double>(j, i);
			if (min > tmp)
				min = tmp;
			if (max < tmp)
				max = tmp;
		}
	}
	term = max - min;
	for (int j = 0; j < gray.rows; j++) {
		for (int i = 0; i < gray.cols; i++) {
			dst.at<uchar>(j, i) = (int)((gray.at<double>(j, i) - min) * 255.0 / term);
			//printf("%d\t", arr[j*width+i]);
		}
	}
}
