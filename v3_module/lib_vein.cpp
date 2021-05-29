#include "d_definition.h"

//#define DEBUG_MODE2
//#define DEBUG_MODE3
//#define DEBUG_MODE4
//#define DEBUG_MODE5
// CLAHE Method
Ptr<CLAHE> clahe = createCLAHE();

void otsu_binary(Mat gray, Mat *dst) {
	double bi_sigma = 0., bi_sigma_min = 9999999999999;
	int Threshold_Binary;
	int gray_value = 0;
	Mat tmp = gray.clone();

	//imshow("tmp", tmp);
	//waitKey();
	//printf("\Otsu Binarization!!\n");
	
	//medianBlur(gray, tmp, 5);
	GaussianBlur(gray, tmp, Size(5, 5), 3, 3);

	//clahe->setClipLimit(4);
	//clahe->apply(tmp, tmp);


	for (int thresh = 60; thresh < 90; thresh += 2) {
		double bi_mean_b = 0, bi_mean_w = 0;
		double bi_variance_b = 0, bi_variance_w = 0;
		double num_b = 0, num_w = 0;

		for (int j = 0; j < tmp.rows; j++) {
			for (int i = 0; i < tmp.cols; i++) {
				gray_value = tmp.at<uchar>(j, i);

				if (gray_value > thresh) {
					bi_mean_w += (double)gray_value;
					bi_variance_w += powf(gray_value, 2);
					num_w++;
				}
				else {
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
		bi_sigma = (bi_variance_b * num_b + bi_variance_w * num_w) / (double)(tmp.cols*tmp.rows);
		
		if (bi_sigma_min > bi_sigma) {
			bi_sigma_min = bi_sigma;
			Threshold_Binary = thresh;
		}
	}

	Threshold_Binary -= 10;
	for (int j = 0; j < tmp.rows; j++) {
		for (int i = 0; i < tmp.cols; i++) {
			gray_value = tmp.at<uchar>(j, i);
			
			if (gray_value > Threshold_Binary)
				(*dst).at<uchar>(j, i) = 255;
			else
				(*dst).at<uchar>(j, i) = 0;
		}
	}
}

void find_defects(Mat binary, vector<Point> *dst) {

#ifdef DEBUG_MODE2
	Mat img_hulldefect;
	cvtColor(binary, img_hulldefect, COLOR_GRAY2BGR);
#endif
	
	vector<vector<Point>> contours;
	findContours(binary, contours, noArray(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	
	if (contours.size() < 1) {
		cout << "contours.size()=" << contours.size() << endl;
		return;
	}

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
	
	// Convex hull obtained using convexHull that should contain indices of the contour points that make the hull.
	vector<int> hull;
	convexHull(handContour, hull);

	if (hull.size() < 1) {
		cout << " hull.size()=" << hull.size() << endl;
		return;
	}

	vector<Point> ptsHull;
	for (int i = 0; i < hull.size(); i++)
	{
		int j = hull[i];
		ptsHull.push_back(handContour[j]);
	}

	vector<Vec4i> defects;
	convexityDefects(handContour, hull, defects);


#ifdef DEBUG_MODE2
	drawContours(img_hulldefect, vector<vector<Point>>(1, ptsHull), 0, B, 2);
	cout << " defects.size()=" << defects.size() << endl;
#endif

	for (int i = 0; i < defects.size(); i++)
	{
		Vec4i v = defects[i];
		Point ptStart = handContour[v[0]];
		Point ptEnd = handContour[v[1]];
		Point ptFar = handContour[v[2]];
		float depth = v[3] / 256.0;
		if (depth > 40)
		{
			(*dst).push_back(ptFar);

#ifdef DEBUG_MODE2
			line(img_hulldefect, ptStart, ptFar, G, 2);
			line(img_hulldefect, ptEnd, ptFar, G, 2);
			circle(img_hulldefect, ptStart, 6, R, 2);
			circle(img_hulldefect, ptEnd, 6, R, 2);
			circle(img_hulldefect, ptFar, 6, Scalar(255, 0, 255), 2);
#endif

		}
	}

#ifdef DEBUG_MODE2
	imshow("binary", binary);
	imshow("convex&hull", img_hulldefect);
	waitKey();
#endif

	//init
	vector<vector<Point>> empty1;
	vector<Point> empty2;
	vector <int> empty3;
	vector<Point> empty4;
	vector<Vec4i> empty5;
	contours.swap(empty1);
	handContour.swap(empty2);
	hull.swap(empty3);
	ptsHull.swap(empty4);
	defects.swap(empty5);

}

void ROI_extraction(Mat img_gray, Mat *dst) {	// dst should be ROI_SIZE X ROI_SIZE img_gray image

	//printf("ROI Extraction!!\n");
	/////////////////////////////////////////////////////////////////////
	/////////////////// Otsu's method binarization //////////////////////
	/////////////////////////////////////////////////////////////////////

	Mat img_binary(img_gray.rows, img_gray.cols, CV_8UC1);
	otsu_binary(img_gray, &img_binary);

#ifdef DEBUG_MODE1
	imshow("original", img_gray);
	imshow("binary", img_binary);
	waitKey();
#endif


	///////////////////////////////////////////////////////////////////////////
	/////////////////////// convex hull & defect //////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	vector<Point> defect_points;
	find_defects(img_binary, &defect_points);
	
	
	/////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////// 3ref points ////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////
	
	sort(defect_points.begin(), defect_points.end(), cmp_x);

	int loop = defect_points.size();
	for (int i = 0; i < loop - 3; i++) defect_points.pop_back();

	sort(defect_points.begin(), defect_points.end(), cmp_y);


	/* algorithm
	vector <Point> real_point;
	int cnt = 0;
	real_point.push_back(defect_points[0]);
	int index = 1;
	int threshold_dist = 40;
	while(cnt<2){
		Point next_point = defect_points[index++];
		int dist_point = 0;
		int dx = real_point[cnt].x-next_point.x;
		int dy = real_point[cnt].y-next_point.y;
		if(dx<0) dx = -dx;
		if(dy<0) dy = -dy;
		dist_point = dx + dy;
		if(dist_point<threshold_dist){
			real_point.push_back(next_point);
			cnt++;
		}
	}

	x축으로 sorting 하고 양끝점 사용.
	*/


	// Rotate the image
	Point p1 = defect_points[0];
	Point p2 = defect_points[2];
	Point center;
	center.x = (p1.x + p2.x) / 2;
	center.y = (p1.y + p2.y) / 2;

	double rotate_degree = 0.;
	double dx = (double)(p2.x - p1.x);
	double dy = (double)(p2.y - p1.y);
	double distance = pow(dx,2) + pow(dy,2);
	
	if (dx == 0) rotate_degree = 90.;
	else rotate_degree = atan2(dy, dx)*180/Pi-90.;
	Mat matRotation = getRotationMatrix2D(center, rotate_degree, 1);
	Mat imgRotated;
	//warpAffine(img_binary, imgRotated, matRotation, img_binary.size());
	warpAffine(img_gray, imgRotated, matRotation, img_gray.size());

#ifdef DEBUG_MODE3
	printf("dx : %lf, dy : %lf\n", dx, dy);
	printf("distance : %lf\n", sqrt(distance));

	Mat img_refpoints;
	cvtColor(img_binary, img_refpoints, COLOR_GRAY2BGR);
	//warpAffine(img_refpoints, imgRotated, matRotation, img_refpoints.size());

	printf("%d %d\n", p2.x, p2.y);
	printf("%d %d\n", p1.x, p1.y);
	printf("%lf", rotate_degree);

	for (int i = 0; i < 3; i++) circle(img_refpoints, defect_points[i], 6, B, 2);
	printf("%d %d\n", defect_points[0].x, defect_points[0].y);
	printf("%d %d\n", defect_points[1].x, defect_points[1].y);
	printf("%d %d\n", defect_points[2].x, defect_points[2].y);

	circle(img_refpoints, center, 6, R, 2);
	circle(imgRotated, center, 6, R, 2);

	imshow("refpoints", img_refpoints);
	imshow("rotate", imgRotated);

	waitKey();
#endif


	/////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////// Image Enhancement ///////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////

	double size_ratio = sqrt(distance) / ref_distance;
	int size = (int)((double)ROI_SIZE * size_ratio);
	//printf("size_ratio : %lf size : %d\n", size_ratio, size);

	Mat ROI_cut = Mat::zeros(size, size, CV_8UC1);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			//printf("i : %d j : %d\n", center.y - size / 2 + i, center.x + j + MARGIN);
			ROI_cut.at<uchar>(i, j) = imgRotated.at<uchar>(center.y - size/ 2 + i, center.x + j + MARGIN);
		}
	}
	resize(ROI_cut, *dst, Size(ROI_SIZE, ROI_SIZE), INTER_CUBIC);
	
	// CLAHE Method
	
	clahe->setClipLimit(4);
	clahe->apply(*dst, *dst);
	
	//medianBlur(*dst, *dst, 5);
	
#ifdef DEBUG_MODE4
	imshow("CLAHE", ROI_cut);
	waitKey();
#endif

	// init
	vector<Point> empty6;
	defect_points.swap(empty6);

}


///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////// matching method //////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 

/* histogram intersection H(p,q) = sum{min(p,q)}/sum{q}*/
float HistogramIntersect(int arr1[], int arr2[], int N) {
	int up_sum = 0, down_sum = 0;
	float result;

	// histogram
	for (int i = 0; i < N; i++) {
		int p = arr1[i];
		int q = arr2[i];
		if (p > q) up_sum += q;
		else up_sum += p;
		down_sum += q;
	}
	result = (float)up_sum / (float)down_sum;

	return result;
}


///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////// LDP /////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 

// size_histogram : 256, window : 4, index_degree: 0,45,90,135 -> 0 : 0~256*16-1, 45 : 256*16 ~ 256*16*2-1 
void GetHistogram_LDP(Mat Descriptor, int histogram[], int size_histogram, int window, int index_degree) {
	index_degree /= 45;
	index_degree *= size_histogram * window*window;
	int height_window = Descriptor.rows / window;
	int width_window = Descriptor.cols / window;
	//printf("degree : %d,h : %d,w : %d\n", index_degree,height_window,width_window);
	for (int h = 0; h < window; h++) {
		for (int w = 0; w < window; w++) {
			int index_window = h * window + w;
			for (int y = h * height_window; y < height_window*(h + 1); y++) {
				for (int x = w * width_window; x < width_window *(w + 1); x++) {
					int index_histogram;
					if (y >= 0 && x >= 0 && y < Descriptor.rows&&x < Descriptor.cols)
						index_histogram = Descriptor.at<uchar>(y, x);
					histogram[index_degree + index_window * size_histogram + index_histogram]++;
					//printf("y : %d, x: %d, value : %d\n",y,x, Descriptor.at<uchar>(y, x));
				}
			}
		}
	}
}

// alpha = 0,45,90,135
// gray : CV_32S1 , dst : CV_32S1
void NextOrder(Mat gray, Mat *dst, int scale, int alpha) {
	if (alpha == 0) {
		for (int y = 0; y < gray.rows; y++) {
			for (int x = 0; x < gray.cols; x++) {
				/* 원래꺼
				if (x + scale < gray.cols)
					dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(y, x + scale);
				*/
				//////////////////// scale을 넘어가는 부분은 그냥 안하도록 했는데 그걸 반대편 이미지를 이용해서 사용하는 방식.//////////////////////////////
				int tmp_x = x + scale;
				if (x + scale >= gray.cols) tmp_x -= gray.cols;
				dst->at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(y, tmp_x);
			}
		}
	}
	else if (alpha == 45) {
		for (int y = 0; y < gray.rows; y++) {
			for (int x = 0; x < gray.cols; x++) {
				/* 원래꺼
				if (y - scale >= 0 && x + scale < gray.cols)
					dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(y - scale, x + scale);
				*/
				//////////////////// scale을 넘어가는 부분은 그냥 안하도록 했는데 그걸 반대편 이미지를 이용해서 사용하는 방식.//////////////////////////////
				int tmp_y = y - scale;
				int tmp_x = x + scale;
				if (y - scale < 0) tmp_y += gray.rows;
				if (x + scale >= gray.cols) tmp_x -= gray.cols;
				dst->at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(tmp_y, tmp_x);
			}
		}
	}
	else if (alpha == 90) {
		for (int y = 0; y < gray.rows; y++) {
			for (int x = 0; x < gray.cols; x++) {
				/* 원래꺼
				if(y - scale >= 0)
					dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(y - scale, x);
				*/
				//////////////////// scale을 넘어가는 부분은 그냥 안하도록 했는데 그걸 반대편 이미지를 이용해서 사용하는 방식.//////////////////////////////
				int tmp_y = y - scale;
				if (y - scale < 0) tmp_y += gray.rows;
				dst->at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(tmp_y, x);
			}
		}
	}
	else if (alpha == 135) {
		for (int y = 0; y < gray.rows; y++) {
			for (int x = 0; x < gray.cols; x++) {
				/* 원래꺼
				if (y - scale >= 0 && x - scale >= 0)
					dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(y - scale, x - scale);
				*/
				//printf("dst : %d\n", dst.at<int>(y, x));
//////////////////// scale을 넘어가는 부분은 그냥 안하도록 했는데 그걸 반대편 이미지를 이용해서 사용하는 방식.//////////////////////////////
				int tmp_x = x - scale;
				int tmp_y = y - scale;
				if (tmp_y < 0) tmp_y += gray.rows;
				if (tmp_x < 0) tmp_x += gray.cols;
				dst->at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(tmp_y, tmp_x);
			}
		}
	}
}

void Make_LDP(Mat gray, Mat *dst, int scale) {
	for (int y = 0; y < gray.rows; y++) {
		for (int x = 0; x < gray.cols; x++) {
			int bit = 0;
			int center = gray.at<int>(y, x);
			int tmp_x, tmp_y;

			tmp_x = x - scale;
			tmp_y = y - scale;
			if (tmp_x < 0) tmp_x += gray.cols;
			if (tmp_y < 0) tmp_y += gray.rows;
			int tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= (1 << 7);

			tmp_x = x;
			tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= (1 << 6);

			tmp_x = x + scale;
			if (tmp_x >= gray.cols) tmp_x -= gray.cols;
			tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= (1 << 5);

			tmp_y = y;
			tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= (1 << 4);

			tmp_y = y + scale;
			if (tmp_y >= gray.rows) tmp_y -= gray.rows;
			tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= (1 << 3);

			tmp_x = x;
			tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= (1 << 2);

			tmp_x = x - scale;
			if (tmp_x < 0) tmp_x += gray.cols;
			tmp = gray.at<int>(tmp_y, tmp_x);
			if (tmp*center <= 0) bit |= 1;

			dst->at<uchar>(y, x) = bit;
		}
	}
}

float vein_LDP(Mat gray1, Mat gray2, int order, int scale, int window_size) {
	int total_window = window_size * window_size;
	int size_histogram = total_window * 4 * 256;

	int *LDP1_histogram;
	LDP1_histogram = (int*)calloc(size_histogram, sizeof(int));
	int *LDP2_histogram;
	LDP2_histogram = (int*)calloc(size_histogram, sizeof(int));

	Mat tmp_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	Mat tmp_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);

	for (int j = 0; j < gray1.rows; j++) {
		for (int i = 0; i < gray1.cols; i++) {
			tmp_gray1.at<int>(j, i) = (int)gray1.at<uchar>(j, i);
			//printf("%d %d\t\t", gray1.at<uchar>(j, i),tmp_gray1.at<int>(j,i));
		}
	}

	for (int j = 0; j < gray2.rows; j++) {
		for (int i = 0; i < gray2.cols; i++) {
			tmp_gray2.at<int>(j, i) = (int)gray2.at<uchar>(j, i);
		}
	}

	// 0도
	Mat order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	Mat order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, &order1_gray1, scale, 0);
	NextOrder(tmp_gray2, &order1_gray2, scale, 0);
	Mat order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	Mat order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, &order2_gray1, scale, 0);
	NextOrder(order1_gray2, &order2_gray2, scale, 0);
	Mat LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	Mat LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, &LDP1, scale);
	Make_LDP(order2_gray2, &LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size, 0);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size, 0);
	//imshow("1", LDP1);
	//waitKey(1000);
	// 45도
	order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, &order1_gray1, scale, 45);
	NextOrder(tmp_gray2, &order1_gray2, scale, 45);
	order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, &order2_gray1, scale, 45);
	NextOrder(order1_gray2, &order2_gray2, scale, 45);
	LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, &LDP1, scale);
	Make_LDP(order2_gray2, &LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size, 45);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size, 45);


	// 90도
	order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, &order1_gray1, scale, 90);
	NextOrder(tmp_gray2, &order1_gray2, scale, 90);
	order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, &order2_gray1, scale, 90);
	NextOrder(order1_gray2, &order2_gray2, scale, 90);
	LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, &LDP1, scale);
	Make_LDP(order2_gray2, &LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size, 90);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size, 90);


	// 135도
	order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, &order1_gray1, scale, 135);
	NextOrder(tmp_gray2, &order1_gray2, scale, 135);
	order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, &order2_gray1, scale, 135);
	NextOrder(order1_gray2, &order2_gray2, scale, 135);
	LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, &LDP1, scale);
	Make_LDP(order2_gray2, &LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size, 135);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size, 135);

	float result = HistogramIntersect(LDP1_histogram, LDP2_histogram, size_histogram);

	printf("%f\n", result);
	free(LDP1_histogram);
	free(LDP2_histogram);

	return result;
}



#ifdef DEBUG_MODE5
int main() {
	Mat src1 = imread("test.jpg", IMREAD_GRAYSCALE);
	//Mat src2 = imread("ex3.jpg", IMREAD_GRAYSCALE);
	//Mat src3 = imread("ex4.jpg", IMREAD_GRAYSCALE);
	//Mat src4 = imread("ex5.jpg", IMREAD_GRAYSCALE);
	Mat dst1(ROI_SIZE, ROI_SIZE, CV_8UC1);
	//Mat dst2(ROI_SIZE, ROI_SIZE, CV_8UC1);
	//Mat dst3(ROI_SIZE, ROI_SIZE, CV_8UC1);
	//Mat dst4(ROI_SIZE, ROI_SIZE, CV_8UC1);
	ROI_extraction(src1, &dst1);
	//ROI_extraction(src2, &dst2);
	//ROI_extraction(src3, &dst3);
	//ROI_extraction(src4, &dst4);
	imshow("result1", dst1);
	//imshow("result2", dst2);
	//imshow("result3", dst3);
	//imshow("result4", dst4);
	waitKey();
	return 0;
}
#endif

/*
int vein_SURF(Mat gray1, Mat gray2) {
	////////////////////////////// SURF /////////////////////////////////////////////////
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors

	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	detector->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);

	//printf("d1 rows : %d\t d2 rows : %d\n", descriptors1.rows, descriptors2.rows);
	//printf("d1 cols : %d\t d2 cols : %d\n", descriptors1.cols, descriptors2.cols);

	// matching 점이 없는 경우
	if (descriptors1.rows < 2 || descriptors2.rows < 2)
	{
		Mat img_matches;
		hconcat(gray1, gray2, img_matches);
		printf("0 Matching \n\n");
		detector.reset();
		keypoints1.clear();
		keypoints2.clear();
		return 9999;
		//-- Show detected matches
		//imshow("Good Matches", img_matches);
		//imwrite("result.jpg", img_matches);
	}
	//imshow("1", descriptors1);
	//imshow("2", descriptors2);

	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
	// Since SURF is a floating-point descriptor NORM_L2 is used
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	std::vector< std::vector<DMatch> > knn_matches;								// 2차원 vector
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);				// 2차원 vector에서 channel 수 2


	//printf("\ncheck\n\n");


	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.7f;
	std::vector<DMatch> good_matches;
	int count = 0;
	//printf("%d\n", knn_matches.size());
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
			count++;
		}
	}
	printf("count : %d\n\n", count);

	//-- Draw matches
	Mat img_matches;
	drawMatches(gray1, keypoints1, gray2, keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	//imshow("Good Matches", img_matches);
	//imwrite("result.jpg", img_matches);
	printf("test\n");

	detector.reset();
	keypoints1.clear();
	keypoints2.clear();
	matcher.reset();
	knn_matches.clear();
	good_matches.clear();
	return count;
}
*/