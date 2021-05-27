//#define vein_recog
#ifdef vein_recog

#include "a_ROI.h"
#include "b_multilevel_detection.h"
#include "c_SIFT.h"
#include <string.h>

#define NUM_IMAGE 22

// const char *s = "hihihi";	-> 접근은 가능하나 바꿀순 없다.
// char s[10] = "hihiihi";		-> 접근도 가능하고 바꿀수도있다.
const char* source0 = "ex.jpg";
char* source1;
char* source2;//"result.jpg"
const char* ex = "ex";
const char* jpg = ".jpg";
const char* res = "result";
char img_n[10];
int index;

extern Mat src, src_gray, src_rotated;
extern Mat ROI, ROI_gray;				// CV_8UC3, CV_8UC1
Mat ROI1, ROI2,result;
extern Mat OGM_maps[8];				// CV_8UC1
extern Mat OGM_maps_DoG[8];			// CV_8UC1
extern Mat OGM_maps_ex[8];			// CV_8UC1
extern Mat ROI_scale_space[4];			// CV_8UC1
extern Mat ROI_laplacian[4];			// CV_8UC1
extern Mat ROI_harris_ex[3];			// extrema scale 1.6, 1.6*1.6	&	result -> scale
extern Mat ROI_hessian_ex[3];			// extrema scale 1.6, 1.6*1.6	&	result -> scale
extern Mat ROI_harris[2];
extern Mat ROI_hessian[2];
extern double *histogram2;
extern int num_Keypoint;
Keypoint *tail, *keys1, *keys2,*hessi1,*hessi2;

Scalar c;
Point pCenter;
int width, height;
int radius=1;

void init_keypoint() {
	keys1 = (Keypoint*)malloc(sizeof(Keypoint));
	keys2 = (Keypoint*)malloc(sizeof(Keypoint));
	hessi1 = (Keypoint*)malloc(sizeof(Keypoint));
	hessi2 = (Keypoint*)malloc(sizeof(Keypoint));
	tail = (Keypoint*)malloc(sizeof(Keypoint));
	keys1->next = tail;
	keys2->next = tail;
	hessi1->next = tail;
	hessi2->next = tail;
	tail->next = tail;
}

/////////////////////////////// main //////////////////////////////////
int main() {
	////////////////////////// first setting ////////////////////////////////////////
	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;
	init_keypoint();
	width = ROI_SIZE;
	height = ROI_SIZE;

	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////// 1st IMAGE //////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////

	src = imread(source0, IMREAD_COLOR);
	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return -1;
	}
	src_gray = src.clone();
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	
	//width = src.cols;
	//height = src.rows;

	///////////////////////// ROI extraction ////////////////////////////////
	
	
	ROI = Mat::zeros(height, width, CV_8UC1);
	ROI_extraction(src_gray, ROI);

	/* normalization
	int ROI_max = -1, ROI_min = 999, ROI_term, ROI_tmp;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			ROI_tmp = ROI.at<uchar>(j,i);
			if (ROI_min > ROI_tmp)
				ROI_min = ROI_tmp;
			if (ROI_max < ROI_tmp)
				ROI_max = ROI_tmp;
		}
	}
	ROI_term = ROI_max - ROI_min;
	if (ROI_term == 0)
		ROI_term = 1;
	for(int j=0;j<height;j++){
		for (int i = 0; i < width; i++) {
			ROI.at<uchar>(j,i) = (int)((ROI.at<uchar>(j,i) - ROI_min) * 255 / ROI_term);
			//printf("%d\t", arr[j*width+i]);
		}
	}
	*/

	//ROI = src_gray.clone();
	Mat ROI_temp = ROI.clone();

	
	imshow("original", src);
	imshow("ROI", ROI);
	imwrite("ROI_result1.jpg", ROI);
	////////////////////////// make OGM-maps /////////////////////////////////////
	
	/*
	for (int j = 0; j < 8; j++) {
		OGM_maps[j] = Mat::zeros(ROI.size(), CV_8UC1);
		OGM_maps_DoG[j] = Mat::zeros(ROI.size(), CV_8UC1);
		OGM_maps_ex[j] = Mat::zeros(ROI.size(), CV_64FC1);
	}
	OGMs();
	*/

	///////////////////////// make gray scale space ///////////////////////////////
	
	
	double scale = 1.0;
	for (int i = 0; i < 4; i++) {
		ROI_scale_space[i] = Mat::zeros(ROI.size(), CV_8UC1);
		ROI_laplacian[i] = Mat::zeros(ROI.size(), CV_8UC1);
		gaussian_filter(ROI, ROI_scale_space[i], scale);
		laplacian(ROI_scale_space[i], ROI_laplacian[i], scale);
		scale *= 1.6;
	}
	scale = 1.0;


	//////////////////////// calc laplace extrema /////////////////////////////////
	
	
	for (int i = 0; i < 3; i++) {
		ROI_harris_ex[i] = Mat::zeros(ROI.size(), CV_64FC1);
		ROI_hessian_ex[i] = Mat::zeros(ROI.size(), CV_64FC1);
	}
	laplace_extrema(ROI_laplacian, ROI_harris_ex);
	laplace_extrema(ROI_laplacian, ROI_hessian_ex);


	///////////////////////// harris & hessian detection //////////////////////////
	
	
	for (int i = 0; i < 2; i++) {
		ROI_harris[i] = Mat::zeros(ROI.size(), CV_8UC1);
		ROI_hessian[i] = Mat::zeros(ROI.size(), CV_8UC1);
	}
	calc_harrihessi(ROI_scale_space[1], 1);						// gray : CV_8UC1, num : 몇번째 scale인지
	calc_harrihessi(ROI_scale_space[2], 2);						// gray : CV_8UC1, num : 몇번째 scale인지


	//////////////////////// DoG at OGMs ///////////////////////////////////////////
	/*
	for (int i = 0; i < 8; i++) {
		DoG(OGM_maps[i], OGM_maps_DoG[i], 1.6);
	}
	*/
	//////////////////////// harris & hessian & DoG thresholding ////////////////////////
	
	
	thresholing();


	//////////////////////////////// SIFT ////////////////////////////////////////////////
	
	
	SIFT(ROI_scale_space[1], ROI_scale_space[2], ROI_harris_ex[2], keys1, tail);				// result : 128 array
	SIFT(ROI_scale_space[1], ROI_scale_space[2], ROI_hessian_ex[2], hessi1,tail);				// result : 128 array


	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////// 2nd IMAGE //////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	

	for(index=13;index<= NUM_IMAGE;index++){
		itoa(index, img_n, 10);
		source1 = (char*)malloc(15 * sizeof(char));
		source2 = (char*)malloc(20 * sizeof(char));
		
		strcpy(source1, ex);
		strcat(source1, img_n);
		strcat(source1, jpg);

		strcpy(source2, res);
		strcat(source2, img_n);
		strcat(source2, jpg);
		printf("%s\n%s\n", source1,source2);
		
		src = imread(source1, IMREAD_COLOR);
		if (src.empty())
		{
			cout << "Could not open or find the image!\n" << endl;
			return -1;
		}
		cvtColor(src, src_gray, COLOR_BGR2GRAY);

		/* rotation
		Mat tmp_gray;
		tmp_gray = src.clone();
		cvtColor(src, tmp_gray, COLOR_BGR2GRAY);

		// Rotate the image
		Mat matRotation = getRotationMatrix2D(Point(src.cols / 2, src.rows / 2), 10, 1);
		warpAffine(tmp_gray, src_gray, matRotation, src.size());			// rotating gray img
		*/

		///////////////////////// ROI extraction ////////////////////////////////


		ROI = Mat::zeros(height, width, CV_8UC1);
		ROI_extraction(src_gray, ROI);

		/* normalization
		ROI_max = -1;
		ROI_min = 999;
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				ROI_tmp = ROI.at<uchar>(j, i);
				if (ROI_min > ROI_tmp)
					ROI_min = ROI_tmp;
				if (ROI_max < ROI_tmp)
					ROI_max = ROI_tmp;
			}
		}
		ROI_term = ROI_max - ROI_min;
		if (ROI_term == 0)
			ROI_term = 1;
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				ROI.at<uchar>(j, i) = (int)((ROI.at<uchar>(j, i) - ROI_min) * 255 / ROI_term);
				//printf("%d\t", arr[j*width+i]);
			}
		}
		*/

		//ROI = src_gray.clone();
		imshow("original_", src_gray);
		imshow("ROI_", ROI);
		imwrite("ROI_result2.jpg", ROI);


		////////////////////////// make OGM-maps /////////////////////////////////////
		/*
		for (int j = 0; j < 8; j++) {
			OGM_maps[j] = Mat::zeros(ROI.size(), CV_8UC1);
			OGM_maps_DoG[j] = Mat::zeros(ROI.size(), CV_8UC1);
			OGM_maps_ex[j] = Mat::zeros(ROI.size(), CV_64FC1);
		}
		OGMs();
		*/
		///////////////////////// make gray scale space ///////////////////////////////


		scale = 1.0;
		for (int i = 0; i < 4; i++) {
			ROI_scale_space[i] = Mat::zeros(ROI.size(), CV_8UC1);
			ROI_laplacian[i] = Mat::zeros(ROI.size(), CV_8UC1);
			gaussian_filter(ROI, ROI_scale_space[i], scale);
			laplacian(ROI_scale_space[i], ROI_laplacian[i], scale);
			scale *= 1.6;
		}
		scale = 1.0;


		///////////////////////// calc laplace extrema /////////////////////////////////


		for (int i = 0; i < 3; i++) {
			ROI_harris_ex[i] = Mat::zeros(ROI.size(), CV_64FC1);
			ROI_hessian_ex[i] = Mat::zeros(ROI.size(), CV_64FC1);
		}
		laplace_extrema(ROI_laplacian, ROI_harris_ex);
		laplace_extrema(ROI_laplacian, ROI_hessian_ex);


		///////////////////////// harris & hessian detection //////////////////////////


		for (int i = 0; i < 2; i++) {
			ROI_harris[i] = Mat::zeros(ROI.size(), CV_8UC1);
			ROI_hessian[i] = Mat::zeros(ROI.size(), CV_8UC1);
		}
		calc_harrihessi(ROI_scale_space[1], 1);						// gray : CV_8UC1, num : 몇번째 scale인지
		calc_harrihessi(ROI_scale_space[2], 2);						// gray : CV_8UC1, num : 몇번째 scale인지


		//////////////////////// DoG at OGMs ///////////////////////////////////////////
		/*
		for (int i = 0; i < 8; i++) {
			DoG(OGM_maps[i], OGM_maps_DoG[i], 1.6);
		}
		*/
		//////////////////////// harris & hessian & DoG thresholding ////////////////////////


		thresholing();
		SIFT(ROI_scale_space[1], ROI_scale_space[2], ROI_harris_ex[2], keys2, tail);				// result : 128 array
		SIFT(ROI_scale_space[1], ROI_scale_space[2], ROI_hessian_ex[2], hessi2, tail);				// result : 128 array
		
		result = FindMatches(ROI_temp, ROI, keys1, keys2, tail);
		result = FindMatches(ROI_temp, ROI, hessi1, hessi2, tail);

		imshow("result", result);
		imwrite(source2, result);

		deleteKeys(keys2, tail);
		deleteKeys(hessi2,tail);

		waitKey(2000);
		free(source1);
		free(source2);
	}

	deleteKeys(keys1, tail);
	deleteKeys(hessi1, tail);
	return 0;
}

#endif