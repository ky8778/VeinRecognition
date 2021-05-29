#pragma once
#include "d_definition.h"

void show_histogram(Mat GrayImg);

void otsu_binary(Mat gray, Mat dst);
void erosion(Mat img);
void dilation(Mat img);
void centroid_calculation(Mat img, cpoint *p);
void tri_value(); 

int check_circle(Mat gray, Point center, float r);
void MaxInscriCir(Mat gray, Point center);
void CutRotate_Interpol(Mat gray, Mat dst);

void calc_Recpoints(Point p1, Point p2, Point *result);
void make_Rectangle(Point p1, Point p2, Point *p3, Point *p4);
void CutRotate_Box(Mat gray, Mat *dst, Point _cent, int _r,double angle);
int getMinSquare(Mat Check_Img,int s, int *min, int *max);
void Make_Box(Mat gray,Mat bi, Mat dst);
int ROI_extraction(Mat img, Mat dst);

