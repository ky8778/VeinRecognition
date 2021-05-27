#include "a_ROI.h"

static Scalar c;
static Point pCenter;
static int radius = 1;
static int width, height;

static int bi_thresh;
static int morpho_kernel[9] = { 0,1,0,1,1,1,0,1,0 };

Mat src, src_gray, src_rotated;

void ROI_extraction(Mat gray, Mat dst) {	// dst should be ROI_SIZE X ROI_SIZE gray image
	printf("\nROI Extraction!!\n");

	c.val[0] = 0;
	c.val[1] = 255;			// green circle
	c.val[2] = 0;

	height = src.rows;
	width = src.cols;

	// Otsu's method binarization
	Mat src_bi(height, width, CV_8UC1);
	Mat rotate_bi(height, width, CV_8UC1);
	otsu_binary(src_gray, src_bi);
	//imshow("src_bi", src_bi);

	// morphological image processing
	// opening
	erosion(src_bi);
	dilation(src_bi);
	// closing
	dilation(src_bi);
	erosion(src_bi);
	//imshow("src_bi_morpho", src_bi);

	// Palm-centroid calculation
	cpoint p;
	centroid_calculation(src_bi, &p);
	/////////////////////// ROI extraction //////////////////////////////
	rotate_calculation(src_bi, rotate_bi, &p);

	Mat print_img;
	cvtColor(rotate_bi, print_img, COLOR_GRAY2BGR);

	// start at centroid
	int x, y;
	centroid_calculation(rotate_bi, &p);
	x = p.x - 30;
	y = p.y + 10;
	//printf("\nx : %d y : %d\n", x, y);

	// �º� 
	int set = 1;
	int count = 0;
	while (set) {
		for (int j = -ROI_SIZE / 2; j < ROI_SIZE / 2; j++) {
			count += rotate_bi.at<uchar>(y + j, x - ROI_SIZE / 2);
			if (rotate_bi.at<uchar>(y + j, x - ROI_SIZE / 2) == 0)
				break;
		}
		if (count == 255 * ROI_SIZE)
			set = 0;
		else {
			x++;
			count = 0;
		}
	}
	//printf("\n1. x : %d y : %d\n", x, y);


	// ����
	set = 1;
	count = 0;
	while (set) {
		count = 0;
		for (int i = -ROI_SIZE / 2; i < ROI_SIZE / 2; i++) {
			count += rotate_bi.at<uchar>(y - ROI_SIZE / 2, x + i);
			if (rotate_bi.at<uchar>(y - ROI_SIZE / 2, x + i) == 0)
				break;
		}
		if (count == 255 * ROI_SIZE)
			set = 0;
		else
			y++;
	}
	//printf("\n2. x : %d y : %d\n", x, y);

	// �Ʒ���
	set = 1;
	count = 0;
	while (set) {
		for (int j = -ROI_SIZE / 2; j < ROI_SIZE / 2; j++) {
			count += rotate_bi.at<uchar>(y + j, x - ROI_SIZE / 2);
			if (rotate_bi.at<uchar>(y + j, x - ROI_SIZE / 2) == 0)
				break;
		}
		if (count == 255 * ROI_SIZE)
			set = 0;
		else {
			x++;
			count = 0;
		}
	}
	//printf("\n3. x : %d y : %d\n", x, y);
	//y += 30;

	/*
	for(int j=-ROI_SIZE/2;j< ROI_SIZE/2;j++)
		printf("%d\n", rotated_binary.at<uchar>(y + j, x - ROI_SIZE/2));
	for (int j = -ROI_SIZE/2; j < ROI_SIZE/2; j++)
		printf("%d\n", rotated_binary.at<uchar>(y - ROI_SIZE/2, x + j));
	*/


	pCenter.x = x;
	pCenter.y = y;
	circle(print_img, pCenter, radius, c, 2, 8, 0); // green circle


	for (int j = -ROI_SIZE / 2; j < ROI_SIZE / 2; j++) {
		pCenter.x = x + j;
		pCenter.y = y - ROI_SIZE / 2;
		circle(print_img, pCenter, radius, c, 2, 8, 0); // green circle
		pCenter.x = x + j;
		pCenter.y = y + ROI_SIZE / 2;
		circle(print_img, pCenter, radius, c, 2, 8, 0); // green circle
		pCenter.x = x - ROI_SIZE / 2;
		pCenter.y = y + j;
		circle(print_img, pCenter, radius, c, 2, 8, 0); // green circle
		pCenter.x = x + ROI_SIZE / 2;
		pCenter.y = y + j;
		circle(print_img, pCenter, radius, c, 2, 8, 0); // green circle
	}

	//imshow("2", print_img);
	//imwrite("ROI_window.jpg", print_img);


	// extract result image
	for (int j = y - ROI_SIZE / 2; j < y + ROI_SIZE / 2; j++) {
		for (int i = x - ROI_SIZE / 2; i < x + ROI_SIZE / 2; i++) {
			dst.at<uchar>(j - y + ROI_SIZE / 2, i - x + ROI_SIZE / 2) = src_rotated.at<uchar>(j, i);
		}
	}
	//imshow("src_rotate", src_rotated);
	//imshow("ROI", dst);
	//imwrite("ROI.jpg", dst);
}

void otsu_binary(Mat gray, Mat dst) {
	//printf("\Otsu Binarization!!\n");
	int bi_loop = 0;
	double bi_sigma, bi_sigma_min = 9999999999999;
	double bi_mean_b = 0, bi_mean_w = 0;
	double bi_variance_b = 0, bi_variance_w = 0;
	double num_b = 0, num_w = 0;
	int gray_value;

	for (int bi_thresh_temp = 60; bi_thresh_temp < 90; bi_thresh_temp += 2) {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				//printf("%d\t", src_gray.at<uchar>(j, i));
				gray_value = gray.at<uchar>(j, i);
				if (gray_value > bi_thresh_temp) {
					dst.at<uchar>(j, i) = 255;
					bi_mean_w += (double)gray_value;
					bi_variance_w += powf(gray_value, 2);
					num_w++;
				}
				else {
					dst.at<uchar>(j, i) = 0;
					bi_mean_b += (double)gray_value;
					bi_variance_b += powf(gray_value, 2);
					num_b++;
				}
			}
		}
		bi_mean_b /= num_b;
		bi_mean_w /= num_w;
		bi_variance_b -= powf(bi_mean_b, 2);
		bi_variance_w -= powf(bi_mean_w, 2);
		bi_sigma = (bi_variance_b * num_b + bi_variance_w * num_w) / (double)(height*width);
		if (bi_sigma_min > bi_sigma) {
			bi_sigma_min = bi_sigma;
			bi_thresh = bi_thresh_temp;
		}
		//printf("%lf\t", bi_sigma_min);
		bi_mean_b = 0;
		bi_mean_w = 0;
		bi_variance_b = 0;
		bi_variance_w = 0;
		num_b = 0;
		num_w = 0;
	}
	//printf("%d", bi_thresh);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			//printf("%d\t", src_gray.at<uchar>(j, i));
			gray_value = gray.at<uchar>(j, i);
			if (gray_value > bi_thresh)
				dst.at<uchar>(j, i) = 255;
			else
				dst.at<uchar>(j, i) = 0;
		}
	}
}

void erosion(Mat img) {
	//printf("\nErosion!!\n");

	Mat result = Mat::zeros(height, width, CV_64FC1);
	int check = 0;
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			// erosion
			if (img.at<uchar>(j, i) == 255) {
				for (int m = 0; m < 9; m++)
					check += (img.at<uchar>(j + m / 3 - 1, i + m % 3 - 1)*morpho_kernel[m] / 255);
				if (check == 5)							// need to change if kernel is changed
					result.at<double>(j, i) = 255;
				else
					result.at<double>(j, i) = 0;
			}
			else
				result.at<double>(j, i) = 0;
			check = 0;
		}
	}
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			img.at<uchar>(j, i) = (int)result.at<double>(j, i);
			//printf("%d\t", result.at<uchar>(j, i));
		}
	}
}

void dilation(Mat img) {
	//printf("\nDilation!!\n");

	Mat result = Mat::zeros(height, width, CV_64FC1);
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			// dilation
			if (img.at<uchar>(j, i) == 255) {
				for (int m = 0; m < 9; m++) {			// need to change if kernel is changed
					if (result.at<double>(j + m / 3 - 1, i + m % 3 - 1) != 255)
						result.at<double>(j + m / 3 - 1, i + m % 3 - 1) = 255 * morpho_kernel[m];
				}
			}
			else if (result.at<double>(j, i) != 255)
				result.at<double>(j, i) = 0;
		}
	}
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			img.at<uchar>(j, i) = (int)result.at<double>(j, i);
		}
	}
}

void centroid_calculation(Mat img, cpoint *p) {
	//printf("\nCentroid Calculation!!\n");
	int m00 = 0, m01 = 0, m10 = 0;
	int value;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			value = img.at<uchar>(j, i);
			m00 += value / 255;
			m01 += (value*j / 255);
			m10 += (value*i / 255);
		}
	}
	p->x = m10 / m00;
	p->y = m01 / m00;

	//printf("\n%d %d %d\n", m00, m10, m01);
	//printf("x : %d, y : %d\n", p->x, p->y);
	//printf("int %d double %lf\n", m01/m00,(double)m01/(double)m00);
}

void rotate_calculation(Mat img, Mat dst, cpoint *p) {				// binary img
	//printf("\nRotate Image!!\n");

	// img for print
	Mat print_img = img.clone();

	cvtColor(img, print_img, COLOR_GRAY2BGR);

	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;


	/////////////////////////////// rotating 1 /////////////////////////////////////////
	double degree;

	// start at centroid point
	int y = p->y;
	int x = p->x;
	int x1, x2, y1, y2;
	//printf("\nx : %d y : %d\n", x, y);

	pCenter.x = x;
	pCenter.y = y;
	circle(print_img, pCenter, radius, c, 2, 8, 0); // red circle


	// point 1
	while (img.at<uchar>(y, ++x) != 0);
	//printf("\nx : %d y : %d\n", x, y);
	x1 = x;
	y1 = y;

	pCenter.x = x1;
	pCenter.y = y1;
	circle(print_img, pCenter, radius, c, 2, 8, 0); // red circle


	// point 2
	x = p->x;
	//while (img.at<uchar>(++y, x) != 0);
	y += 80;
	while (img.at<uchar>(y, ++x) != 0);
	//printf("\nx : %d y : %d\n", x, y);
	x2 = x;
	y2 = y;

	pCenter.x = x;
	pCenter.y = y;
	circle(print_img, pCenter, radius, c, 2, 8, 0); // red circle

	// print the img for debugging
	// imshow("binary", print_img);
	// imwrite("binary.jpg", print_img);


	// calculate degree
	degree = atan2(y1 - y2, x1 - x2) * 180 / Pi;
	// printf("%lf\n%d\t%d\n", degree,height,width);


	// Rotate the image
	Mat matRotation = getRotationMatrix2D(Point(width / 2, height / 2), degree, 1);

	Mat print_img1, rotate_img_1st, rotate_gray1;
	warpAffine(print_img, print_img1, matRotation, img.size());		// rotating img for print
	warpAffine(img, rotate_img_1st, matRotation, img.size());			// rotating result img
	warpAffine(src_gray, rotate_gray1, matRotation, img.size());			// rotating gray img

	// save the 1st rotated image
	//imwrite("rotation1.jpg", print_img1);


	///////////////////////////// 2nd rotation ///////////////////////////////////
	// start at centroid point
	centroid_calculation(rotate_img_1st, p);
	x = p->x;
	y = p->y;

	// point 1
	while (rotate_img_1st.at<uchar>(y, --x) != 0);
	//printf("\nx : %d y : %d\n", x, y);
	x1 = x;
	y1 = y;

	pCenter.x = x1;
	pCenter.y = y1;
	circle(print_img1, pCenter, radius, c, 2, 8, 0); // red circle

	// point 2
	x = p->x;
	//while (img.at<uchar>(++y, x) != 0);
	y -= 30;
	while (rotate_img_1st.at<uchar>(y, --x) != 0);
	//printf("\nx : %d y : %d\n", x, y);
	x2 = x;
	y2 = y;

	pCenter.x = x;
	pCenter.y = y;
	circle(print_img1, pCenter, radius, c, 2, 8, 0); // red circle

	//imshow("1", print_img1);

	// calculate degree
	degree = atan2(y1 - y2, x1 - x2) * 180 / Pi - 90;
	// printf("%lf\n%d\t%d\n", degree, height, width);

	// rotate
	Mat print_img2;
	matRotation = getRotationMatrix2D(Point(width / 2, height / 2), degree, 1);
	warpAffine(print_img1, print_img2, matRotation, print_img.size());
	warpAffine(rotate_img_1st, dst, matRotation, print_img.size());
	warpAffine(rotate_gray1, src_rotated, matRotation, print_img.size());
}
