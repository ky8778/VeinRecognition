#pragma once
#include "d_definition.h"

int vein_SURF(Mat gray1, Mat gray2);

int getBitChange(int number, int bit_n);
int getLUT(int arr[], int bit_n);
void getLBP(Mat gray, lbp dst, int P,int R);
float HistogramIntersect(int arr1[], int arr2[], int N);
void getLBPHistogram(lbp _LBP, int window_num, int max);
float matchingLBP(lbp LBP1, lbp LBP2, int window_num);
float vein_LBP(Mat gray1, Mat gray2, int P, int R, int window_num);

void GetHistogram_LDP(Mat Descriptor, int histogram[], int size_histogram, int window, int index_degree);
void NextOrder(Mat gray, Mat dst, int scale, int alpha);
void Make_LDP(Mat gray, Mat dst, int scale);
float vein_LDP(Mat gray1, Mat gray2, int order, int scale, int window_size);