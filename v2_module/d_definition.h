#pragma once
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<stdlib.h>

/*------------------------------ Macros  ---------------------------------*/

#define ABS(x)    (((x) > 0) ? (x) : (-(x)))
#define MAX(x,y)  (((x) > (y)) ? (x) : (y))
#define MIN(x,y)  (((x) < (y)) ? (x) : (y))
#define Pi 3.14159265359
#define ROI_SIZE 190
#define k 0.04


using namespace cv;
using namespace std;

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