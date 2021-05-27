#pragma once
#include "d_definition.h"


void make_Gaussian_kernel();
void SIFT(Mat g_map, Mat gk_map, Mat feature, Keypoint *head,Keypoint *tail);
void SIFT_histogram(Mat gray, int y, int x, int bin_num, double scale, double *histo_array);
void norm_histogram();

void init_keypoint();
void InsertKeys(int descrip_array[],int x, int y, double scale, Keypoint *head, Keypoint *tail);
void printKeys(Keypoint *head, Keypoint *tail);
void deleteKeys(Keypoint *head, Keypoint *tail);
Mat CombineImagesVertically(Mat im1, Mat im2);
Mat FindMatches(Mat im1,Mat im2, Keypoint *head1, Keypoint *head2, Keypoint *tail);
Keypoint* CheckForMatch(Keypoint *key, Keypoint *klist, Keypoint *tail);
int DistSquared(Keypoint *k1, Keypoint *k2);


