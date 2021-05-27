#pragma once
#include "d_definition.h"


void OGMs();
void gaussian_filter(Mat gray, Mat dst, double scale);						// gray : CV_8UC1, dst : CV_8UC1
void calc_harrihessi(Mat gray, int num);										// gray : CV_8UC1, num : 몇번째 scale인지
void laplacian(Mat gray, Mat dst, double scale);								// gray : CV_8UC1, dst : CV_8UC1
void laplace_extrema(Mat origin[], Mat dst[]);				// origin : CV8UC1 (laplacian), dst : CV_32FC1 (extreme -> scale)
void DoG(Mat gray, Mat dst, double scale);									// gray : CV_8UC1, dst : CV_8UC1
void thresholing();
void norm(Mat gray, Mat dst);								// gray : CV_32FC1, dst : CV_8UC1
