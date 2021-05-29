#include "vein_match.h"
#define use_uniformLBP
#define intersect1

static int minHessian = 200;//400;
static int LookUpTable[65540];
static bool check[65540];

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










///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
//////////////////////////////// matching method //////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 

// arr[]의 start에서 end까지 ChiSquare 값 sum(i에서 값 - 기대값)^2/기대값을 반환한다.
/*
float ChiSquare(int arr[],int N,float expect) {
	float CHI = 0;
	for (int i = 0; i < N; i++) {
		if (arr[i] <= 5) continue;
		CHI += (((float)arr[i] - expect)*((float)arr[i] - expect) / expect);
	}
	return CHI;
}
*/

/* CHI square = sum{(i-th histogram 차이)^2/(i-th histogram 합)} / 2
float ChiSquare(int arr1[], int arr2[], int N) {
	float CHI = 0;
	for (int i = 0; i < N; i++) {
		if (arr1[i] <= 5 || arr2[i] <= 5) continue;
		int diff_2 = (arr1[i] - arr2[i])*(arr1[i] - arr2[i]);
		int sum = arr1[i] + arr2[i];
		if (sum == 0) sum = 1;
		//float tmp = (float)diff_2 / (float)sum;
		//printf("diff : %d, sum : %d, tmp : %f\n",diff_2,sum, tmp);
		CHI += (float)diff_2 / (float)sum;
	}
	//CHI /= 243.;
	CHI /= 2.;
	return CHI;
}*/

/* histogram intersection H(p,q) = sum{min(p,q)}/sum{q}*/
float HistogramIntersect(int arr1[], int arr2[], int N) {
	int up_sum = 0, down_sum = 0;
	float result;

	/* histogram
	*/
	for (int i = 0; i < N; i++) {
		int p = arr1[i];
		int q = arr2[i];
		if (p > q) up_sum += q;
		else up_sum += p;
		down_sum += q;
	}
	result = (float)up_sum / (float)down_sum;

	/* code for check 
	for (int i = 0; i < 4096; i++) {
		int p = arr1[i];
		int q = arr2[i];
		if (p > q) up_sum += q;
		else up_sum += p;
		down_sum += q;
		//printf("i : %d, p : %d ,q : %d ,up_sum :%d,down_sum: %d\n",i, p, q, up_sum, down_sum);
	}
	result = (float)up_sum / (float)down_sum;
	up_sum = 0;
	down_sum = 0;
	printf("%f\n", result);

	for (int i = 4096; i < 4096*2; i++) {
		int p = arr1[i];
		int q = arr2[i];
		if (p > q) up_sum += q;
		else up_sum += p;
		down_sum += q;
	}
	result = (float)up_sum / (float)down_sum;
	up_sum = 0;
	down_sum = 0;
	printf("%f\n", result);

	for (int i = 4096*2; i < 4096*3; i++) {
		int p = arr1[i];
		int q = arr2[i];
		if (p > q) up_sum += q;
		else up_sum += p;
		down_sum += q;
	}
	result = (float)up_sum / (float)down_sum;
	up_sum = 0;
	down_sum = 0;
	printf("%f\n", result);

	for (int i = 4096*3; i < 4096*4; i++) {
		int p = arr1[i];
		int q = arr2[i];
		if (p > q) up_sum += q;
		else up_sum += p;
		down_sum += q;
	}
	result = (float)up_sum / (float)down_sum;
	up_sum = 0;
	down_sum = 0;
	printf("%f\n", result);
	*/
	return result;
}

///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////// LBP /////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 

// 숫자와 비트수를 인수로 받아서 몇번 비트가 바뀌는지 리턴하는 함수
int getBitChange(int number, int bit_n) {
	int count = 0;
	// 첫번째 비트
	int now = (number & 1);
	//printf("number : %d, bit : %d, i : %d, now : %d\n", number, bit_n, 0, (number >> 0) & 1);
	for (int i = 1; i < bit_n; i++) {
		//printf("number : %d, bit : %d, i : %d, now : %d\n", number, bit_n, i, (number >> i) & 1);
		if (now != ((number >> i) & 1)) {
			count++;
			now = ((number >> i) & 1);
		}
		//printf("count : %d\n", count);
	}
	return count;
}

// arr[]와 bit수 n을 받아서 arr[]에 
// 비트가 2번 이하로 바뀌는 숫자는 count를 1씩 증가시키면서 값을 담고
// 나머지 배열은 마지막 count로 모두 담는 함수.
int getLUT(int arr[], int bit_n) {
	int max = 1;
	int count = 0;
	for (int i = 0; i < bit_n; i++) max *= 2;

	for (int i = 0; i < max; i++) {
		int num = getBitChange(i, bit_n);
		if (num <= 2) {
			LookUpTable[i] = count++;
			check[i] = true;
		}
	}
	for (int i = 0; i < max; i++) {
		if (check[i] == false) LookUpTable[i] = count;
	}
	return (++count);
}

// P = 16, R = 
// P = 16, R = 7 LBP를 얻는 함수.
void getLBP(Mat gray, lbp dst, int P, int R) {							// gray & dst are same size CV_8UC1 image
	if (R == 7) {
		for (int j = R; j < gray.rows - R; j++) {
			for (int i = R; i < gray.cols - R; i++) {
				int center = gray.at<uchar>(j, i);
				int code = 0;
				int tmp = 0;
				tmp = gray.at<uchar>(j + 2, i + 6) + gray.at<uchar>(j + 2, i + 7) + gray.at<uchar>(j + 3, i + 6) + gray.at<uchar>(j + 3, i + 7);
				code |= ((tmp / 4) > center) << 15;
				tmp = gray.at<uchar>(j + 4, i + 4) + gray.at<uchar>(j + 4, i + 5) + gray.at<uchar>(j + 5, i + 4) + gray.at<uchar>(j + 5, i + 5);
				code |= ((tmp / 4) > center) << 14;
				tmp = gray.at<uchar>(j + 6, i + 2) + gray.at<uchar>(j + 6, i + 3) + gray.at<uchar>(j + 7, i + 2) + gray.at<uchar>(j + 7, i + 3);
				code |= ((tmp / 4) > center) << 13;
				code |= (gray.at<uchar>(j + 7, i) > center) << 12;

				tmp = gray.at<uchar>(j + 6, i - 2) + gray.at<uchar>(j + 6, i - 3) + gray.at<uchar>(j + 7, i - 2) + gray.at<uchar>(j + 7, i - 3);
				code |= ((tmp / 4) > center) << 11;
				tmp = gray.at<uchar>(j + 4, i - 4) + gray.at<uchar>(j + 4, i - 5) + gray.at<uchar>(j + 5, i - 4) + gray.at<uchar>(j + 5, i - 5);
				code |= ((tmp / 4) > center) << 10;
				tmp = gray.at<uchar>(j + 2, i - 6) + gray.at<uchar>(j + 2, i - 7) + gray.at<uchar>(j + 3, i - 6) + gray.at<uchar>(j + 3, i - 7);
				code |= ((tmp / 4) > center) << 9;
				code |= (gray.at<uchar>(j, i - 7) > center) << 8;

				tmp = gray.at<uchar>(j - 2, i - 6) + gray.at<uchar>(j - 2, i - 7) + gray.at<uchar>(j - 3, i - 6) + gray.at<uchar>(j - 3, i - 7);
				code |= ((tmp / 4) > center) << 7;
				tmp = gray.at<uchar>(j - 4, i - 4) + gray.at<uchar>(j - 4, i - 5) + gray.at<uchar>(j - 5, i - 4) + gray.at<uchar>(j - 5, i - 5);
				code |= ((tmp / 4) > center) << 6;
				tmp = gray.at<uchar>(j - 6, i - 2) + gray.at<uchar>(j - 6, i - 3) + gray.at<uchar>(j - 7, i - 2) + gray.at<uchar>(j - 7, i - 3);
				code |= ((tmp / 4) > center) << 5;
				code |= (gray.at<uchar>(j - 7, i) > center) << 4;

				tmp = gray.at<uchar>(j - 6, i + 2) + gray.at<uchar>(j - 6, i + 3) + gray.at<uchar>(j - 7, i + 2) + gray.at<uchar>(j - 7, i + 3);
				code |= ((tmp / 4) > center) << 3;
				tmp = gray.at<uchar>(j - 4, i + 4) + gray.at<uchar>(j - 4, i + 5) + gray.at<uchar>(j - 5, i + 4) + gray.at<uchar>(j - 5, i + 5);
				code |= ((tmp / 4) > center) << 2;
				tmp = gray.at<uchar>(j - 2, i + 6) + gray.at<uchar>(j - 2, i + 7) + gray.at<uchar>(j - 3, i + 6) + gray.at<uchar>(j - 3, i + 7);
				code |= ((tmp / 4) > center) << 1;
				code |= (gray.at<uchar>(j, i + 7) > center);

				//////////////////////////// uniform LBP ///////////////////////////
#ifdef use_uniformLBP
				dst.lbpArr[dst.width*j + i] = LookUpTable[code];
#endif
				//printf("j : %d, i : %d, lbp : %d\n", j, i, LookUpTable[code]);
			}
		}
	}
}

// window_num : window갯수
void getLBPHistogram(lbp _LBP,int window_num,int max) {
	int h = _LBP.height / window_num;
	int w = _LBP.width / window_num;
	// window loop
	for (int nh = 0; nh < window_num; nh++) {
		for (int nw = 0; nw < window_num; nw++) {
			int window_index = window_num * nh + nw;							
			// window_index = 0 1 2 3 -> 4 5 6 7-> ... 마다 max(243)개의 histogram
			
			// pixel loop in one window
			for (int y = nh * h; y < nh*h + h; y++) {
				for (int x = nw * w; x < nw*w + w; x++) {
					int LBP_index = y * _LBP.width + x;
					int bin = _LBP.lbpArr[LBP_index];
					int histogram_index = window_index * max + bin;			// box index * max(243) + histogram bin값
					_LBP.LBP_histogram[histogram_index]++;
				}
			}
			//printf("box : %d\n", window_index);
			//for (int i = window_index*max; i < (window_index+1)*max; i++) printf("i : %d, histogram : %d\n",i, _LBP.LBP_histogram[i]);
		}
	}
}


float matchingLBP(lbp LBP1,lbp LBP2,int window_num) {
	int tmp[243];
	int tmp1[243],tmp2[243];
	float E_HI = 0.;

#ifdef intersect1
	int N = 243 * window_num*window_num;
	E_HI = HistogramIntersect(LBP1.LBP_histogram, LBP2.LBP_histogram, N);
#endif

#ifdef intersect2
	for (int t = 0; t < window_num*window_num; t++) {
		int sum = 0;
		float expectation;
		float HI,chi;
		
		/* difference method
		for (int i = 243 * t; i < 243 * (t + 1); i++) {
			int difference = LBP1.LBP_histogram[i] - LBP2.LBP_histogram[i];
			if (difference < 0) difference = -difference;
			printf("LBP1, i : %d, histogram : %d\n", i, LBP1.LBP_histogram[i]);
			printf("LBP2, i : %d, histogram : %d\n", i, LBP2.LBP_histogram[i]);
			printf("difference : %d\n", difference);
			sum += difference;
		}
		printf("sum : %d\n", sum);
		*/

		/* LBP1 chi 
		printf("LBP1\n");
		for (int i = 243 * t; i < 243 * (t + 1); i++) {
			sum += LBP1.LBP_histogram[i];
			tmp[i - 243 * t] = LBP1.LBP_histogram[i];
		}
		expectation = (float)sum / 243.;
		chi = ChiSquare(tmp, 243, expectation);
		printf("sum : %d, E : %f, CHI : %f\n", sum, expectation,chi);
		sum = 0;
		*/
		/* LBP2 chi
		printf("\n\nLBP2\n");
		for (int i = 243 * t; i < 243 * (t + 1); i++) {
			sum += LBP2.LBP_histogram[i];
			tmp[i - 243 * t] = LBP2.LBP_histogram[i];
		}
		expectation = (float)sum / 243.;
		chi = ChiSquare(tmp, 243, expectation);
		printf("sum : %d, E : %f, CHI : %f\n", sum, expectation, chi);
		*/

		/* LBP CHI method2
		for (int i = 243 * t; i < 243 * (t + 1); i++) {
			tmp1[i - 243 * t] = LBP1.LBP_histogram[i];
			tmp2[i - 243 * t] = LBP2.LBP_histogram[i];
		}
		chi = ChiSquare(tmp1, tmp2, 243);
		printf("CHI^2 : %f\n", chi);
		*/

		/* histogram intersection */
		for (int i = 243 * t; i < 243 * (t + 1); i++) {
			tmp1[i - 243 * t] = LBP1.LBP_histogram[i];
			tmp2[i - 243 * t] = LBP2.LBP_histogram[i];
		}
		HI = HistogramIntersect(tmp1, tmp2, 243);
		//printf("H(p,q) = %f\n", HI);
		E_HI += HI;
		//waitKey(1000);
	}
	E_HI /= (window_num*window_num);
#endif

	printf("E_H(p,q) = %f\n", E_HI);

	return E_HI;
}

// gray1,gray2를 LBP matching하는 함수. P = 16, R = 7
float vein_LBP(Mat gray1, Mat gray2, int P, int R, int window_num) {
	lbp LBP1, LBP2;
	int max;
	float result;
	max = getLUT(LookUpTable, P);
	LBP1.makeLBP(gray1.rows, gray1.cols);
	LBP2.makeLBP(gray2.rows, gray2.cols);
	LBP1.makeHistogram(window_num,max);
	LBP2.makeHistogram(window_num,max);
	getLBP(gray1, LBP1, P, R);
	getLBP(gray2, LBP2, P, R);
	getLBPHistogram(LBP1, window_num,max);
	getLBPHistogram(LBP2, window_num,max);

	result = matchingLBP(LBP1, LBP2,window_num);

	LBP1.deleteLBP();
	LBP2.deleteLBP();
	LBP1.deleteHistogram();
	LBP2.deleteHistogram();

	return result;
}


///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////// LDP /////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 
///////////////////////////////////////////////////////////////////////////////// 

// size_histogram : 256, window : 4, index_degree: 0,45,90,135 -> 0 : 0~256*16-1, 45 : 256*16 ~ 256*16*2-1 
void GetHistogram_LDP(Mat Descriptor, int histogram[],int size_histogram ,int window,int index_degree) {
	index_degree /= 45;
	index_degree *= size_histogram * window*window;
	int height_window = Descriptor.rows / window;
	int width_window = Descriptor.cols / window;
	//printf("degree : %d,h : %d,w : %d\n", index_degree,height_window,width_window);
	for (int h = 0; h < window; h++) {
		for (int w = 0; w < window; w++) {
			int index_window = h * window + w;
			for (int y = h * height_window; y < height_window*(h+1); y++) {
				for (int x = w * width_window; x < width_window *(w+ 1); x++) {
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
void NextOrder(Mat gray, Mat dst, int scale,int alpha) {
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
				dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(y, tmp_x);
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
				dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(tmp_y, tmp_x);
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
				dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(tmp_y, x);
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
				dst.at<int>(y, x) = gray.at<int>(y, x) - gray.at<int>(tmp_y, tmp_x);
			}
		}
	}
}

/* 원래거 
// gray : CV_32S, dst : CV_8UC1
void Make_LDP(Mat gray, Mat dst, int scale) {
	for (int y = scale; y < gray.rows - scale; y++) {
		for (int x = scale; x < gray.cols - scale; x++) {
			int bit = 0;
			int center = gray.at<int>(y, x);
			
			int tmp = gray.at<int>(y - scale, x - scale);
			if (tmp*center <= 0) bit |= (1<<7);
			tmp = gray.at<int>(y - scale, x);
			if (tmp*center <= 0) bit |= (1<<6);
			tmp = gray.at<int>(y - scale, x + scale);
			if (tmp*center <= 0) bit |= (1 << 5);
			tmp = gray.at<int>(y, x + scale);
			if (tmp*center <= 0) bit |= (1 << 4);
			tmp = gray.at<int>(y + scale, x + scale);
			if (tmp*center <= 0) bit |= (1 << 3);
			tmp = gray.at<int>(y + scale, x);
			if (tmp*center <= 0) bit |= (1 << 2);
			tmp = gray.at<int>(y + scale, x - scale);
			if (tmp*center <= 0) bit |= 1;
			dst.at<uchar>(y, x) = bit;
		}
	}
}
*/
void Make_LDP(Mat gray, Mat dst, int scale) {
	for (int y = 0; y < gray.rows ; y++) {
		for (int x = 0; x < gray.cols ; x++) {
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

			dst.at<uchar>(y, x) = bit;
		}
	}
}

float vein_LDP(Mat gray1, Mat gray2, int order, int scale,int window_size) {
	int total_window=window_size*window_size;
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
	NextOrder(tmp_gray1, order1_gray1, scale, 0);
	NextOrder(tmp_gray2, order1_gray2, scale, 0);
	Mat order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	Mat order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, order2_gray1, scale, 0);
	NextOrder(order1_gray2, order2_gray2, scale, 0);
	Mat LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	Mat LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, LDP1, scale);
	Make_LDP(order2_gray2, LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size,0);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size,0);
	//imshow("1", LDP1);
	//waitKey(1000);
	// 45도
	order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, order1_gray1, scale, 45);
	NextOrder(tmp_gray2, order1_gray2, scale, 45);
	order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, order2_gray1, scale, 45);
	NextOrder(order1_gray2, order2_gray2, scale, 45);
	LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, LDP1, scale);
	Make_LDP(order2_gray2, LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size,45);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size,45);


	// 90도
	order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, order1_gray1, scale, 90);
	NextOrder(tmp_gray2, order1_gray2, scale, 90);
	order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, order2_gray1, scale, 90);
	NextOrder(order1_gray2, order2_gray2, scale, 90);
	LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, LDP1, scale);
	Make_LDP(order2_gray2, LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size,90);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size,90);


	// 135도
	order1_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order1_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(tmp_gray1, order1_gray1, scale, 135);
	NextOrder(tmp_gray2, order1_gray2, scale, 135);
	order2_gray1 = Mat::zeros(gray1.rows, gray1.cols, CV_32SC1);
	order2_gray2 = Mat::zeros(gray2.rows, gray2.cols, CV_32SC1);
	NextOrder(order1_gray1, order2_gray1, scale, 135);
	NextOrder(order1_gray2, order2_gray2, scale, 135);
	LDP1 = Mat::zeros(gray1.rows, gray1.cols, CV_8UC1);
	LDP2 = Mat::zeros(gray2.rows, gray2.cols, CV_8UC1);
	Make_LDP(order2_gray1, LDP1, scale);
	Make_LDP(order2_gray2, LDP2, scale);
	GetHistogram_LDP(LDP1, LDP1_histogram, 256, window_size,135);
	GetHistogram_LDP(LDP2, LDP2_histogram, 256, window_size,135);

	float result = HistogramIntersect(LDP1_histogram, LDP2_histogram, size_histogram);

	printf("%f\n", result);
	free(LDP1_histogram);
	free(LDP2_histogram);

	return result;
}