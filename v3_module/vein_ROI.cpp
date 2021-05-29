#include "vein_ROI.h"
#define method2
//#define debugging
//#define ROI_check

char **ROI_write;//"result_~~~~.jpg"

static int Threshold_Binary;
static Scalar G;
static Scalar R;
static Scalar B;
static Point max_Rcent;											// max 인접원의 중심점
static Point midfinger;
static Point pCenter;
static vector<Point> ref_points;
static int bi_thresh;											// otsu binary threshold
static int morpho_kernel[9] = { 0,1,0,1,1,1,0,1,0 };
static int ROI_r = (int)(118.*sqrt(2));
static float cos_v[360], sin_v[360];							// cosine value, sine value
static float max_R = 0.;										// max 인접원의 반지름

Mat Rec_Box, ROI_Box;

Mat canny_input;
Mat canny_gray;
Mat canny_edges;
Mat canny_dst;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int kernel_size = 3;
const char* window_name = "Edge Map";


void show_histogram(Mat GrayImg){
	
	Mat gray_plane = GrayImg.clone();

	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	Mat gray_hist;

	calcHist(&gray_plane, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("histogram", histImage);
	imwrite("histogram.jpg", histImage);
	waitKey(1000);
}

static bool cmp_x(const Point &p1, const Point &p2) {
	if (p1.x < p2.x) {
		return true;
	}
	else if (p1.x == p2.x) {
		return p1.y < p2.y;
	}
	else {
		return false;
	}
}

static bool cmp_y(const Point &p1, const Point &p2) {
	if (p1.y < p2.y) {
		return true;
	}
	else if (p1.y == p2.y) {
		return p1.x < p2.x;
	}
	else {
		return false;
	}
}

void tri_value() {
	for (int i = 0; i < 360; i++) {
		cos_v[i] = cosf((float)i*Pi / 180);
		sin_v[i] = sinf((float)i*Pi / 180);
		//printf("%d. cos : %f, sin : %f\n",i ,cos_v[i], sin_v[i]);
	}
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
		for (int j = 0; j < gray.rows; j++) {
			for (int i = 0; i < gray.cols; i++) {
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
		bi_sigma = (bi_variance_b * num_b + bi_variance_w * num_w) / (double)(gray.cols*gray.rows);
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
	Threshold_Binary = bi_thresh-10;
	for (int j = 0; j < gray.rows; j++) {
		for (int i = 0; i < gray.cols; i++) {
			//printf("%d\t", src_gray.at<uchar>(j, i));
			gray_value = gray.at<uchar>(j, i);
			if (gray_value > Threshold_Binary)
				dst.at<uchar>(j, i) = 255;
			else
				dst.at<uchar>(j, i) = 0;
		}
	}
}

void erosion(Mat img) {
	//printf("\nErosion!!\n");

	Mat result = Mat::zeros(img.rows, img.cols, CV_64FC1);
	int check = 0;
	for (int j = 1; j < img.rows - 1; j++) {
		for (int i = 1; i < img.cols - 1; i++) {
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
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			img.at<uchar>(j, i) = (int)result.at<double>(j, i);
			//printf("%d\t", result.at<uchar>(j, i));
		}
	}
}

void dilation(Mat img) {
	//printf("\nDilation!!\n");

	Mat result = Mat::zeros(img.rows, img.cols, CV_64FC1);
	for (int j = 1; j < img.rows - 1; j++) {
		for (int i = 1; i < img.cols - 1; i++) {
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
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			img.at<uchar>(j, i) = (int)result.at<double>(j, i);
		}
	}
}

void centroid_calculation(Mat img, cpoint *p) {
	//printf("\nCentroid Calculation!!\n");
	int m00 = 0, m01 = 0, m10 = 0;
	int value;
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			value = img.at<uchar>(j, i);
			m00 += value / 255;
			m01 += (value*j / 255);
			m10 += (value*i / 255);
		}
	}
	p->x = m10 / m00;
	p->y = m01 / m00;
}

// 특정 중심에서 r만큼 떨어진 점들 중 검정색이 있는지 확인하는 함수. 검정 점의 갯수를 리턴.
int check_circle(Mat gray, Point center, float r) {
	int xx, yy, count = 0;
	for (int s = 0; s < 360; s++) {
		xx = center.x + (int)(r * cos_v[s]);
		yy = center.y + (int)(r * sin_v[s]);
		if (xx >= 0 && yy >= 0 && xx < gray.cols&&yy < gray.rows&&gray.at<uchar>(yy, xx) == 0) count++;
	}
	return count;
}

// 최대 반지름을 갖는 내접원 찾는 함수.
void MaxInscriCir(Mat gray, Point center) {
	// 빠르게 하기 위해서는 r을 증가시키는 걸
	// 10씩 증가시키고 5씩 감소시키고 2씩 증가시키는 식으로 하면 더 빠를 것으로 예상됨.
	Point cent;
	int flag = 0;
	int start_x = center.x - 50, end_x = center.x + 50;
	int start_y = center.y - 50, end_y = center.y + 50;
	float r;

	max_R = 0.;
	for (int j = start_y; j < end_y; j++) {
		for (int i = start_x; i < end_x; i++) {
			cent.x = i;
			cent.y = j;
			r = 0.;
			flag = 0;
			while (flag == 0) {
				flag = check_circle(gray, cent, r);
				r += 1.;
			}
			if (r > max_R) {
				max_R = r;
				max_Rcent.x = cent.x;
				max_Rcent.y = cent.y;
			}
		}
	}
}

void CutRotate_Interpol(Mat gray, Mat dst) {
	int tmp_size1 = (int)max_R;
	Mat tmp1(2 * tmp_size1, 2 * tmp_size1, CV_8UC1, Scalar(0)); // black 0 으로 초기화
	for (int i = -tmp_size1; i < tmp_size1; i++) {
		for (int j = -tmp_size1; j < tmp_size1; j++) {
			if (max_Rcent.y + j >= 0 && max_Rcent.y + j < gray.rows&&max_Rcent.x + i >= 0 && max_Rcent.x + i < gray.cols)
				tmp1.at<uchar>(tmp_size1 + j, tmp_size1 + i) = gray.at<uchar>(max_Rcent.y + j, max_Rcent.x + i);
		}
	}

	// Rotate the image
	double degree;
	if (max_Rcent.x != midfinger.x) degree = atan2((max_Rcent.y - midfinger.y), (max_Rcent.x - midfinger.x));
	else {
		if (max_Rcent.y == midfinger.y) degree = 0.;
		else if (max_Rcent.y > midfinger.y) degree = 90.;
		else degree = -90.;
	}
	degree = degree * 180 / Pi;
	printf("degree : %lf\n", degree);
	Mat matRotation = getRotationMatrix2D(Point(tmp_size1, tmp_size1), degree, 1);
	Mat rotate_tmp;
	warpAffine(tmp1, rotate_tmp, matRotation, tmp1.size());			// rotating gray img

	// 원에 내접하는 정사각형 최대는 R을 루트2로 나누고 곱하기 2한 것이 한 변의 길이.
	int tmp_size2 = (int)(max_R / sqrtf(2));
	printf("SIZE : %d\n", tmp_size2);
	Mat tmp2(2 * tmp_size2, 2 * tmp_size2, CV_8UC1, Scalar(0)); // black 0 으로 초기화
	for (int i = -tmp_size2; i < tmp_size2; i++) {
		for (int j = -tmp_size2; j < tmp_size2; j++) {
			if (tmp_size1 + j >= 0 && tmp_size1 + j < rotate_tmp.rows && tmp_size1 + i >= 0 && tmp_size1 + i < rotate_tmp.cols)
				tmp2.at<uchar>(tmp_size2 + j, tmp_size2 + i) = rotate_tmp.at<uchar>(tmp_size1 + j, tmp_size1 + i);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////// dst에 bicubic interpolation //////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	resize(tmp2, dst, Size(ROI_SIZE, ROI_SIZE), INTER_CUBIC);
}

// p1에서 p2로 선을 그어 그 거리의 3/2에 해당하는 점을 구하는 함수.
void calc_Recpoints(Point p1, Point p2, Point *result) {
	int x_distance, y_distance;
	x_distance = p2.x - p1.x;
	y_distance = p2.y - p1.y;

	x_distance *= 3;
	x_distance /= 2;
	y_distance *= 3;
	y_distance /= 2;

	result->x = p1.x + x_distance;
	result->y = p1.y + y_distance;
}

// p1, p2로 p3, p4를 계산하는 함수.
void make_Rectangle(Point p1, Point p2, Point *p3, Point *p4) {
	int x_distance, y_distance;
	x_distance = p2.x - p1.x;
	y_distance = p2.y - p1.y;
	p3->x = p1.x + y_distance;
	p3->y = p1.y - x_distance;
	p4->x = p2.x + y_distance;
	p4->y = p2.y - x_distance;
}

// rec_r로 박스를 자르고 회전을 하게되면 rec_r에 해당하는 정사각형을 모두 얻을 수 있다.
// -> 그리고나서 전체 사이즈를 ROI_r에 맞추게되면 회전을 고려했을 때 ROI_SIZE의 정사각형을 얻을 수 있다.
void CutRotate_Box(Mat gray, Mat *dst, Point _cent, int r, double _angle) {
	int x = _cent.x;
	int y = _cent.y;
	Mat tmp;
	cvtColor(gray, tmp, COLOR_GRAY2BGR);
	for (int i = -r; i < r; i++) {
		for (int j = -r; j < r; j++) {
			if (y + j >= 0 && x + i >= 0 && y + j < gray.rows && x + i < gray.cols) {
				dst->at<uchar>(j + r, i + r) = gray.at<uchar>(y + j, x + i);
				//circle(tmp, Point(x + i, y + j), 1, R, 5, 8);
			}
		}
	}
	//imshow("1", tmp);
	Mat matRotation = getRotationMatrix2D(Point(r, r), _angle - 90., 1);

	// Rotate the image
	Mat imgRotated;
	warpAffine(*dst, imgRotated, matRotation, dst->size());

	//imshow("rotate", imgRotated);
	resize(imgRotated, *dst, cv::Size(ROI_r * 2, ROI_r * 2), 0, 0, INTER_CUBIC);
}

// 얻은 박스에서 검정색 점들이 포함되어 있는 경우(check_img에)
// 그 안에서 기준점 3개의 ref points에서 얻은 2개의 rec points에서 그 중심을 기준으로 가장 큰 정사각형을 얻어내는 함수.
// 정사각형의 한 변을 리턴한다. 그 변의 길이를 이용해서 ROI추출 후 다시 resize가 가능하다.
int getMinSquare(Mat Check_Img, int s, int *min, int *max) {
	int result = 1;
	int check = 0;
	int ref_y = Check_Img.rows / 2;
	int ref_x = Check_Img.cols / 2;
	*min = ref_y;
	*max = ref_y;
	while (check == 0) {
		for (int i = -s; i <= s; i++) {
			if (Check_Img.at<uchar>(*max, ref_x + i) == 0) check++;
			//printf("%d\t", Check_Img.at<uchar>(j + ref_y, i + ref_x));
		}
		(*max)++;
	}
	check = 0;
	while (check == 0) {
		for (int i = -s; i <= s; i++) {
			if (Check_Img.at<uchar>(*min, ref_x + i) == 0) check++;
			//printf("%d\t", Check_Img.at<uchar>(j + ref_y, i + ref_x));
		}
		(*min)--;
	}
	//printf("%d %d", *min, *max);
	// 확인해보는 부분
	//printf("=======================================test======================================\n");
	cvtColor(Check_Img, Check_Img, COLOR_GRAY2BGR);
	if(*max-*min<2*s){
		for (int j = *min; j <= *max; j++) {
			for (int i = -s; i <= s; i++) {
				circle(Check_Img, Point(i + ref_x, j), 3, G, 1, 8);
			}
		}
	}
	else {
		for (int j = *max-2*s; j <= *max; j++) {
			for (int i = -s; i <= s; i++) {
				circle(Check_Img, Point(i + ref_x, j), 3, G, 1, 8);
			}
		}
	}
	for (int j = *min; j < *max; j++)
		circle(Check_Img, Point(ref_x - s, j), 3, R, 2, 8);

#ifdef debugging
	imshow("Check_Box", Check_Img);
#endif
	//imwrite("Check_Box.jpg",Check_Img);

	return *max - ref_y;
}

void Make_Box(Mat gray, Mat bi, Mat dst) {
	Point Recpoint1, Recpoint2, Recpoint3, Recpoint4;
	Point Rec_center;
	double angle;
	int Rec_r;

	calc_Recpoints(ref_points[1], ref_points[0], &Recpoint1);
	calc_Recpoints(ref_points[1], ref_points[2], &Recpoint2);

	circle(dst, Recpoint1, 6, Scalar(255, 0, 255), 2);
	circle(dst, Recpoint2, 6, Scalar(255, 0, 255), 2);

	make_Rectangle(Recpoint1, Recpoint2, &Recpoint3, &Recpoint4);
	circle(dst, Recpoint3, 6, Scalar(255, 0, 255), 2);
	circle(dst, Recpoint4, 6, Scalar(255, 0, 255), 2);
	Rec_center.x = (Recpoint1.x + Recpoint2.x + Recpoint3.x + Recpoint4.x) / 4;
	Rec_center.y = (Recpoint1.y + Recpoint2.y + Recpoint3.y + Recpoint4.y) / 4;
	circle(dst, Rec_center, 6, Scalar(255, 0, 255), 2);

	angle = atan2(Recpoint2.y - Recpoint1.y, Recpoint2.x - Recpoint1.x);
	angle = angle * 180 / Pi;															// radian -> degree

	//printf("Rectangle r : %lf ROI_r : %lf angle : %lf\n", Rec_r, ROI_r, angle);
	Rec_r = (int)sqrt(((Rec_center.x - Recpoint1.x)*(Rec_center.x - Recpoint1.x) + (Rec_center.y - Recpoint1.y)*(Rec_center.y - Recpoint1.y)) / 2);

	int min_s = 9999;
	if (min_s > gray.rows - Rec_center.y) min_s = gray.rows - Rec_center.y;
	if (min_s > gray.cols - Rec_center.x) min_s = gray.cols - Rec_center.x;
	if (min_s > Rec_center.x) min_s = Rec_center.x;
	if (min_s > Rec_center.y) min_s = Rec_center.y;
	//printf("%d\n", min_s);
	min_s = min_s - 5;
	Mat BigBox(min_s * 2, min_s * 2, CV_8UC1);
	Mat BigCheckBox(min_s * 2, min_s * 2, CV_8UC1);
	for (int j = -min_s; j < min_s; j++) {
		for (int i = -min_s; i < min_s; i++) {
			BigBox.at<uchar>(j + min_s, i + min_s) = gray.at<uchar>(j + Rec_center.y, i + Rec_center.x);
			BigCheckBox.at<uchar>(j + min_s, i + min_s) = bi.at<uchar>(j + Rec_center.y, i + Rec_center.x);
		}
	}
#ifdef debugging
	imshow("BigBox", BigBox);
#endif
	//printf("%d %d", gray.rows, gray.cols);

	Mat tmpRotation = getRotationMatrix2D(Point(min_s, min_s), angle - 90., 1);

	// Rotate the image
	Mat BigBox_Rotated, BigCheckBox_Rotated;
	warpAffine(BigBox, BigBox_Rotated, tmpRotation, BigBox.size());
	warpAffine(BigCheckBox, BigCheckBox_Rotated, tmpRotation, BigBox.size());
	
	//cvtColor(BigCheckBox_Rotated, BigCheckBox_Rotated, COLOR_GRAY2BGR);
	//rectangle(BigCheckBox_Rotated, Point(min_s-Rec_r, min_s - Rec_r), Point(min_s + Rec_r, min_s + Rec_r), cv::Scalar(0, 255, 0), 1);

	int max_y = min_s, min_y = min_s;
	int error = getMinSquare(BigCheckBox_Rotated, Rec_r, &min_y, &max_y);
	//printf("error : %d, Rec_r : %d\n", error,Rec_r);
	//printf("=======================================test======================================\n");
#ifdef debugging
	printf("=======================================test======================================\n");
	imshow("BigBox_Ro", BigBox_Rotated);
	imshow("BigCheckBox_Ro", BigCheckBox_Rotated);
#endif

#ifdef ROI_check
	imwrite(ROI_write[2], BigBox_Rotated);
#endif
	if (error < Rec_r) {
		///////////////////////////////////////////수정한 부분/////////////////////////////////////////////////////////
		if (max_y - min_y > Rec_r * 2) {
			Rec_Box = Mat::zeros(Rec_r * 2, Rec_r * 2, CV_8UC1);
			for (int j = max_y - 2 * Rec_r; j < max_y; j++) {
				for (int i = -Rec_r; i < Rec_r; i++) {
					Rec_Box.at<uchar>(j - max_y + 2 * Rec_r, i + Rec_r) = BigBox_Rotated.at<uchar>(j, i + min_s);
				}
			}
		}
		else {
			///////////////////////////////////////////수정한 부분/////////////////////////////////////////////////////////
			int tmp = max_y - min_y;
			//printf("\ntmp : %d\n", tmp);
			Rec_Box = Mat::zeros(tmp, Rec_r * 2, CV_8UC1);
			for (int j = min_y; j < max_y; j++) {
				for (int i = -Rec_r; i < Rec_r; i++) {
					Rec_Box.at<uchar>(j - min_y, i + Rec_r) = BigBox_Rotated.at<uchar>(j, i + min_s);
				}
			}
		}
		//printf("=======================================test======================================\n");
	}
	else {
		Rec_Box = Mat::zeros(Rec_r * 2, Rec_r * 2, CV_8UC1);
		for (int j = -Rec_r; j < Rec_r; j++) {
			for (int i = -Rec_r; i < Rec_r; i++) {
				Rec_Box.at<uchar>(j + Rec_r, i + Rec_r) = BigBox_Rotated.at<uchar>(j + min_s, i + min_s);
			}
		}
		//printf("=======================================test======================================\n");
	}
	resize(Rec_Box, ROI_Box, cv::Size(ROI_SIZE, ROI_SIZE), 0, 0, INTER_CUBIC);
}

int ROI_extraction(Mat gray, Mat dst) {	// dst should be ROI_SIZE X ROI_SIZE gray image
	// printf("ROI Extraction!!\n");
	// Green
	G.val[0] = 0;
	G.val[1] = 255;
	G.val[2] = 0;
	// Red
	R.val[0] = 0;
	R.val[1] = 0;
	R.val[2] = 255;
	// Blue
	B.val[0] = 255;
	B.val[1] = 0;
	B.val[2] = 0;
	
	//show_histogram(gray);
	/*canny_input = gray.clone();
	canny_dst.create(canny_input.size(), canny_input.type());
	CannyThreshold(0,0); */

	/////////////////////////////////////////////////////////////////////
	/////////////////// Otsu's method binarization //////////////////////
	/////////////////////////////////////////////////////////////////////

	Mat src_bi(gray.rows, gray.cols, CV_8UC1);
	otsu_binary(gray, src_bi);
	//Mat src_bi_otsu;
	//cv::threshold(gray, src_bi_otsu, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("opencv", src_bi_otsu);
	//imshow("mine", src_bi);
	//waitKey(5000);

	/*
	for (int j = 0; j < 3; j++) {
		// opening
		erosion(src_bi);
		erosion(src_bi_otsu);
		dilation(src_bi);
		dilation(src_bi_otsu);
		dilation(src_bi);
		dilation(src_bi_otsu);
		// closing
		dilation(src_bi);
		dilation(src_bi_otsu);
		dilation(src_bi);
		dilation(src_bi_otsu);
		erosion(src_bi);
		erosion(src_bi_otsu);
	}
	*/
	//imshow("ori", src_bi);
	
	//for (int i = 0; i < 4; i++) dilate(src_bi, src_bi, Mat());
	//for (int i = 0; i < 4; i++) erode(src_bi, src_bi, Mat());

	/*
	for(int i=0;i<4;i++){
		morphologyEx(src_bi, src_bi, MORPH_OPEN,Mat());
		morphologyEx(src_bi, src_bi, MORPH_CLOSE, Mat());
		morphologyEx(src_bi, src_bi, MORPH_CLOSE, Mat());
		morphologyEx(src_bi, src_bi, MORPH_OPEN, Mat());
	}
	*/
	/////////////////////////////////////////////////////////////////////
	/////////////////////// ROI extraction //////////////////////////////
	/////////////////////////////////////////////////////////////////////
	Mat result1, result2;
	cvtColor(src_bi, result1, COLOR_GRAY2BGR);
	cvtColor(src_bi, result2, COLOR_GRAY2BGR);


	// 바이너리이미지에서 흰색을 포함하는 윤곽들의 집합을 모두 구함
	vector<vector<Point>> contours;
	findContours(src_bi, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//cout << "contours.size()=" << contours.size() << endl;

	if (contours.size() < 1)
		return -1;

	// 흰색을 포함하는 윤곽들의 면적을 계산해서 최대 면적만을 선택 -> 손 윤곽만 사용.
	int maxK = 0;
	double maxArea = contourArea(contours[0]);
	for (int i = 1; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > maxArea)
		{
			maxK = i;
			maxArea = area;
		}
	}

	vector<Point> handContour = contours[maxK];

	// 손 윤곽을 포함하는 볼록껍질을 찾아냄. hull에 좌표의 handcontour배열 index를 push함.
	// Convex hull obtained using convexHull that should contain indices of the contour points that make the hull.
	vector<int> hull;
	convexHull(handContour, hull);
	//cout << " hull.size()=" << hull.size() << endl;

	// hull 배열의 각 값으로 좌표를 받음
	vector<Point> ptsHull;
	for (int i = 0; i < hull.size(); i++)
	{
		int j = hull[i];
		ptsHull.push_back(handContour[j]);
	}

	// 좌표로 볼록 껍질그리기.
	drawContours(result1, vector<vector<Point>>(1, ptsHull), 0,
		Scalar(255, 0, 0), 2);

	// 윤곽으로부터 defects 점 찾기.
	vector<Vec4i> defects;
	convexityDefects(handContour, hull, defects);

	// The output vector of convexity defects. 
	// In C++ and the new Python/Java interface 
	// each convexity defect is represented as 4-element integer vector (a.k.a. Vec4i)
	// : (start_index, end_index, farthest_pt_index, fixpt_depth), 
	// where indices are 0-based indices in the original contour of the convexity defect beginning, end and the farthest point
	// , and fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the farthest contour point and the hull. 
	// That is, to get the floating-point value of the depth will be fixpt_depth/256.0.
	for (int i = 0; i < defects.size(); i++)
	{
		Vec4i v = defects[i];
		Point ptStart = handContour[v[0]];
		Point ptEnd = handContour[v[1]];
		Point ptFar = handContour[v[2]];
		float depth = v[3] / 256.0;
		if (depth > 10)
		{
			ref_points.push_back(ptFar);
			line(result1, ptStart, ptFar, Scalar(0, 255, 0), 2);
			line(result1, ptEnd, ptFar, Scalar(0, 255, 0), 2);
			circle(result1, ptStart, 6, Scalar(0, 0, 255), 2);
			circle(result1, ptEnd, 6, Scalar(0, 0, 255), 2);
			circle(result1, ptFar, 6, Scalar(255, 0, 255), 2);
		}
	}
	//cout << " defects.size()=" << defects.size() << endl;

	/* center점 잡기
	*/
	int start_x = 0, start_y = 0;
	for (int i = 0; i < ref_points.size(); i++) {
		start_x += ref_points[i].x;
		start_y += ref_points[i].y;
	}
	start_x /= ref_points.size();
	start_y /= ref_points.size();

	pCenter.x = start_x;
	pCenter.y = start_y;

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////// 3개의 ref points 잡기 ///////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////

	// x축으로 sorting -> 0,1,2 가 원하는 points가 됨.
	sort(ref_points.begin(), ref_points.end(), cmp_x);
	// 맨 처음 잡힌 3점
	for (int i = 0; i < ref_points.size(); i++) {
		if (i < 3) circle(result1, ref_points[i], 6, B, 2);
	}

	// x좌표로 에러처리
	int error_ref_points = 3;
	//printf("point1 : %d, point2 : %d\n", ref_points[0].x, ref_points[1].x);
	while (abs(ref_points[0].x - ref_points[1].x) > 70) {
		printf("------------------------test1-------------------------------\n");
		printf("point1 : %d, point2 : %d\n", ref_points[0].x, ref_points[1].x);
		circle(result2, ref_points[0], 6, R, 2);
		circle(result2, ref_points[error_ref_points], 6, R, 2);
		ref_points[0].x = ref_points[error_ref_points].x;
		ref_points[0].y = ref_points[error_ref_points].y;
		error_ref_points++;
		if (error_ref_points >= (int)ref_points.size()) {
			printf("cannot detect correct defects!!\n");
			imshow("1", result1);
			imshow("2", result2);
#ifdef ROI_check
			imwrite(ROI_write[0], result1);
			imwrite(ROI_write[1], result2);
#endif
			////////////// initialize //////////////////
			contours.clear();
			handContour.clear();
			hull.clear();
			ptsHull.clear();
			defects.clear();
			ref_points.clear();
			return -1;
		}
	}

	if (abs(ref_points[2].x - ref_points[1].x) > 70) {
		printf("point1 : %d, point2 : %d\n", ref_points[1].x, ref_points[2].x);
		printf("cannot detect correct defects!!\n");
		dst = result1.clone();
		////////////// initialize //////////////////
		contours.clear();
		handContour.clear();
		hull.clear();
		ptsHull.clear();
		defects.clear();
		ref_points.clear();
		return -1;
	}


	// y좌표로 에러처리 
	// 먼저 3점 sorting.
	int key_y, key_x;
	for (int i = 1; i < 3; i++) {
		key_y = ref_points[i].y;
		key_x = ref_points[i].x;

		int j;
		for (j = i - 1; j >= 0 && ref_points[j].y > key_y; j--) {
			ref_points[j + 1].y = ref_points[j].y;
			ref_points[j + 1].x = ref_points[j].x;
		}

		ref_points[j + 1].x = key_x;
		ref_points[j + 1].y = key_y;
	}
	error_ref_points = 3;
	while (abs(ref_points[2].y - ref_points[1].y) > 100 || abs(ref_points[1].y - ref_points[0].y)>100) {
		
		printf("------------------------test2-------------------------------\n");
		printf("point1 : %d, point2 : %d, point3 : %d\n", ref_points[0].y, ref_points[1].y, ref_points[2].y);
		
		if(abs(ref_points[2].y - ref_points[1].y)> 100){
			if (abs(ref_points[1].y - ref_points[0].y) > 100) {
				printf("cannot detect correct defects!!\n");
				imshow("1", result1);
				imshow("2", result2);
#ifdef ROI_check
				imwrite(ROI_write[0], result1);
				imwrite(ROI_write[1], result2);
#endif
				return -1;
			}
			ref_points[2].x = ref_points[error_ref_points].x;
			ref_points[2].y = ref_points[error_ref_points].y;
			error_ref_points++;
			if (error_ref_points-1 >= (int)ref_points.size()) {
				printf("cannot detect correct defects!!\n");
				imshow("1", result1);
				imshow("2", result2);
#ifdef ROI_check
				imwrite(ROI_write[0], result1);
				imwrite(ROI_write[1], result2);
#endif
				////////////// initialize //////////////////
				contours.clear();
				handContour.clear();
				hull.clear();
				ptsHull.clear();
				defects.clear();
				ref_points.clear();
				return -1;
			}
		}
		else{
			ref_points[0].x = ref_points[error_ref_points].x;
			ref_points[0].y = ref_points[error_ref_points].y;
			error_ref_points++;
			if (error_ref_points - 1 >= (int)ref_points.size()) {
				printf("cannot detect correct defects!!\n");
				imshow("1", result1);
				imshow("2", result2);
#ifdef ROI_check
				imwrite(ROI_write[0], result1);
				imwrite(ROI_write[1], result2);
#endif
				////////////// initialize //////////////////
				contours.clear();
				handContour.clear();
				hull.clear();
				ptsHull.clear();
				defects.clear();
				ref_points.clear();
				return -1;
			}
		}
		// 다시 확인해야하기 때문에 다시 sorting
		for (int i = 1; i < 3; i++) {
			key_y = ref_points[i].y;
			key_x = ref_points[i].x;

			int j;
			for (j = i - 1; j >= 0 && ref_points[j].y > key_y; j--) {
				ref_points[j + 1].y = ref_points[j].y;
				ref_points[j + 1].x = ref_points[j].x;
			}

			ref_points[j + 1].x = key_x;
			ref_points[j + 1].y = key_y;
		}
	}
	//waitKey();


	// 3개의 점만 남기고 pop
	int tmp = ref_points.size();
	for (int i = 3; i < tmp; i++) ref_points.pop_back();

	// 3개의 점을 y축으로 sorting.
	sort(ref_points.begin(), ref_points.end(), cmp_y);
	circle(result2, ref_points[1], 6, G, 2);
	circle(result2, ref_points[0], 6, G, 2);
	circle(result2, ref_points[2], 6, G, 2);

#ifdef debugging
	imshow("1", result1);
	imshow("2", result2);
#endif

#ifdef ROI_check
	imwrite(ROI_write[0], result1);
	imwrite(ROI_write[1], result2);
#endif

#ifdef method1

	tri_value();
	cpoint p;
	centroid_calculation(src_bi, &p);

	/* Palm-centroid calculation
	pCenter.x = p.x;
	pCenter.y = p.y;
	*/
	circle(result2, pCenter, 3, G, 3, 8);
	circle(result2, Point(p.x, p.y), 3, R, 3, 8);

	midfinger.x = ref_points[1].x;
	midfinger.y = ref_points[1].y;
	MaxInscriCir(src_bi, pCenter);
	Mat temp_src = gray.clone();
	cvtColor(temp_src, temp_src, COLOR_GRAY2BGR);
	circle(temp_src, max_Rcent, max_R, R, 1, 8);
	CutRotate_Interpol(gray, dst);
	//imshow("dst", dst);

#endif

#ifdef method2

	// 3개의 점으로 정사각형을 만들고 회전시켜서 ROI_Box 만들기
	Make_Box(gray, src_bi, result2);
	//imshow("ROI_Box", ROI_Box);

	// ROI_Box를 ROI인 dst에 픽셀마다 대입
	for (int j = 0; j < ROI_SIZE; j++) {
		for (int i = 0; i < ROI_SIZE; i++) {
			dst.at<uchar>(j, i) = ROI_Box.at<uchar>(j, i);
		}
	}

#endif

	//imshow("1", temp_src);
	//imshow("result2", result2);
	//imshow("result1", result1);
	//imshow("dst", dst);
	//waitKey(500);

	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////// Image Enhancement ///////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	Mat tmp_original = dst.clone();

	/* normalization
	int ROI_max = -1, ROI_min = 999, ROI_term, ROI_tmp;
	for (int j = 0; j < ROI_SIZE; j++) {
		for (int i = 0; i < ROI_SIZE; i++) {
			ROI_tmp = dst.at<uchar>(j, i);
			if (ROI_min > ROI_tmp)
				ROI_min = ROI_tmp;
			if (ROI_max < ROI_tmp)
				ROI_max = ROI_tmp;
		}
	}
	ROI_term = ROI_max - ROI_min;
	if (ROI_term == 0)
		ROI_term = 1;
	for (int j = 0; j < ROI_SIZE; j++) {
		for (int i = 0; i < ROI_SIZE; i++) {
			dst.at<uchar>(j, i) = (int)((dst.at<uchar>(j, i) - ROI_min) * 255 / ROI_term);
		}
	}
	imshow("norm", dst);
	waitKey(500);
	*/

	/* histogram equalization
	equalizeHist(dst, dst);
	imshow("histo_eq", dst);
	waitKey(500);
	*/

	/* CLAHE Method */
	Ptr<CLAHE> clahe = createCLAHE();

	clahe->setClipLimit(4);
	clahe->apply(dst, dst);

	
#ifdef debugging
	imshow("original", tmp_original);
	imshow("CLAHE", dst);
	waitKey(500);
#endif

	////////////// initialize //////////////////
	contours.clear();
	handContour.clear();
	hull.clear();
	ptsHull.clear();
	defects.clear();
	ref_points.clear();

	return 1;
}

