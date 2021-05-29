#pragma once
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<stdio.h>
#include<iostream>
#include<algorithm>
#include<math.h>
#include<stdlib.h>
#include <string.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <Windows.h>

/*------------------------------ Macros  ---------------------------------*/

#define ABS(x)    (((x) > 0) ? (x) : (-(x)))
#define MAX(x,y)  (((x) > (y)) ? (x) : (y))
#define MIN(x,y)  (((x) < (y)) ? (x) : (y))
#define Pi 3.14159265359
#define ROI_SIZE 236
#define k 0.04


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// BGR
const Scalar B = { 255,0,0 };
const Scalar G = { 0,255,0 };
const Scalar R = { 0,0,255 };
const int MARGIN = 10;
const int NUM_IMAGE = 6;
const int minHessian = 200;//400;
const double ref_distance = 160.;

typedef struct Cpoint {
	int x;
	int y;
}cpoint;

/* Data structure for a keypoint.  Lists of keypoints are linked
   by the "next" field. */

typedef struct KeypointSt {
	int x, y;             /* Subpixel location of keypoint. */
	float scale, ori;           /* Scale and orientation (range [-PI,PI]) */
	int *descrip;     /* Vector of descriptor values */
	struct KeypointSt *next;    /* Pointer to next keypoint in list. */
} Keypoint;

typedef struct LBPimg {
	int height, width;
	int *lbpArr;
	int *LBP_histogram;
	int histogram_size;
	void makeLBP(int h,int w){
		height = h;
		width = w;
		lbpArr = (int*)calloc(height*width,sizeof(int));
	}
	void deleteLBP() {
		free(lbpArr);
	}
	void makeHistogram(int window_num,int max) {
		histogram_size = max * window_num*window_num;
		LBP_histogram = (int*)calloc(histogram_size, sizeof(int));
	}
	void deleteHistogram() {
		free(LBP_histogram);
	}
}lbp;

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