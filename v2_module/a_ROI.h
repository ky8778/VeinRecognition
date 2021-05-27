#pragma once
#include "d_definition.h"

void erosion(Mat img);
void dilation(Mat img);
void centroid_calculation(Mat img, cpoint *p);
void rotate_calculation(Mat img, Mat dst, cpoint *p);
void otsu_binary(Mat gray, Mat dst);
void ROI_extraction(Mat img, Mat dst);