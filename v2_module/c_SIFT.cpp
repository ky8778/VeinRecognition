#include "c_SIFT.h"

static double gaussian_kernel[5][5];
static double gaussian_kernel_k[5][5];
//static double gradi_mag[5][5];
static double *histogram;
static int *histogram_norm;
static double max_histogram;
static int ori,trigger;
int num_Keypoint;


void make_Gaussian_kernel() {
	int x, y;

	// make Gaussian Kernel
	double r, s = (2.0 * 1.6 * 1.6);  // Assigning standard deviation to 1.0
	double ks = 1.6*1.6*s;
	double sum = 0.0, ksum = 0.0;   // Initialization of sun for normalization

	for (x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
	{
		for (y = -2; y <= 2; y++)
		{
			r = sqrt(x*x + y * y);

			gaussian_kernel[x + 2][y + 2] = (exp(-(r*r) / s)) / (Pi * s);
			sum += gaussian_kernel[x + 2][y + 2];

			gaussian_kernel[x + 2][y + 2] = (exp(-(r*r) / ks)) / (Pi * ks);
			ksum += gaussian_kernel[x + 2][y + 2];
		}
	}

	for (int i = 0; i < 5; ++i) { // Loop to normalize the kernel
		for (int j = 0; j < 5; ++j) {
			gaussian_kernel[i][j] /= sum;
			gaussian_kernel_k[i][j] /= ksum;
		}
	}
}

void SIFT(Mat g_map, Mat gk_map, Mat feature, Keypoint *head, Keypoint *tail) {					// result : 128 array
	
	double scale,tmp_ori;
	int n1 = 0;
	num_Keypoint = 0;
	
	max_histogram = -1;
	ori = 0;
	trigger = 0;
	
	printf("Scale Invariant Feature Transform !!\n");

	make_Gaussian_kernel();

	histogram = (double*)calloc(128, sizeof(double));
	histogram_norm = (int*)calloc(128, sizeof(int));

	// entire image loop	big window : 19 X 19 & gaussian_window : 2 X 2
	for (int j = 10; j < feature.rows - 10; j++) {
		for (int i = 10; i < feature.cols - 10; i++) {
			scale = feature.at<double>(j, i);
			if (scale == 1.6)
			{
				//printf("%d %d %lf \n", i, j, scale);
				SIFT_histogram(g_map, j, i, 10, scale, histogram);
				// find max hitogram value
				for (int m = 0; m < 36; m++) {
					//printf("%-2lf\t", histogram[m]);
					if (max_histogram < histogram[m]) {
						max_histogram = histogram[m];
						tmp_ori = (double)m;
					}
					histogram[m] = 0;
				}
				tmp_ori /= 4.5;
				ori = (int)tmp_ori;
				//printf("\nmax : %-2lf\n", max_histogram);

				trigger = 1;
				SIFT_histogram(g_map, j - 8, i - 8, 45, scale, histogram);
				SIFT_histogram(g_map, j - 8, i - 4, 45, scale, histogram + 8);
				SIFT_histogram(g_map, j - 8, i + 1, 45, scale, histogram + 16);
				SIFT_histogram(g_map, j - 8, i + 5, 45, scale, histogram + 24);
				SIFT_histogram(g_map, j - 4, i - 8, 45, scale, histogram + 32);
				SIFT_histogram(g_map, j - 4, i - 4, 45, scale, histogram + 40);
				SIFT_histogram(g_map, j - 4, i + 1, 45, scale, histogram + 48);
				SIFT_histogram(g_map, j - 4, i + 5, 45, scale, histogram + 56);
				SIFT_histogram(g_map, j + 1, i - 8, 45, scale, histogram + 64);
				SIFT_histogram(g_map, j + 1, i - 4, 45, scale, histogram + 72);
				SIFT_histogram(g_map, j + 1, i + 1, 45, scale, histogram + 80);
				SIFT_histogram(g_map, j + 1, i + 5, 45, scale, histogram + 88);
				SIFT_histogram(g_map, j + 5, i - 8, 45, scale, histogram + 96);
				SIFT_histogram(g_map, j + 5, i - 4, 45, scale, histogram + 104);
				SIFT_histogram(g_map, j + 5, i + 1, 45, scale, histogram + 112);
				SIFT_histogram(g_map, j + 5, i + 5, 45, scale, histogram + 120);

				norm_histogram();
				InsertKeys(histogram_norm, i, j, scale, head, tail);
			}
			else if (scale == 1.6*1.6)
			{
				//printf("%d %d %lf \n", i, j, scale);
				SIFT_histogram(gk_map, j, i, 10, scale, histogram);
				
				// find max hitogram value
				for (int m = 0; m < 36; m++) {
					//printf("%-2lf\t", histogram[m]);
					if (max_histogram < histogram[m]) {
						max_histogram = histogram[m];
						tmp_ori = (double)m;
					}
					histogram[m] = 0;
				}
				//printf("\nmax : %-2lf\n", max_histogram);
				tmp_ori /= 4.5;
				ori = (int)tmp_ori;

				trigger = 1;
				SIFT_histogram(gk_map, j - 8, i - 8, 45, scale, histogram);
				SIFT_histogram(gk_map, j - 8, i - 4, 45, scale, histogram + 8);
				SIFT_histogram(gk_map, j - 8, i + 1, 45, scale, histogram + 16);
				SIFT_histogram(gk_map, j - 8, i + 5, 45, scale, histogram + 24);
				SIFT_histogram(gk_map, j - 4, i - 8, 45, scale, histogram + 32);
				SIFT_histogram(gk_map, j - 4, i - 4, 45, scale, histogram + 40);
				SIFT_histogram(gk_map, j - 4, i + 1, 45, scale, histogram + 48);
				SIFT_histogram(gk_map, j - 4, i + 5, 45, scale, histogram + 56);
				SIFT_histogram(gk_map, j + 1, i - 8, 45, scale, histogram + 64);
				SIFT_histogram(gk_map, j + 1, i - 4, 45, scale, histogram + 72);
				SIFT_histogram(gk_map, j + 1, i + 1, 45, scale, histogram + 80);
				SIFT_histogram(gk_map, j + 1, i + 5, 45, scale, histogram + 88);
				SIFT_histogram(gk_map, j + 5, i - 8, 45, scale, histogram + 96);
				SIFT_histogram(gk_map, j + 5, i - 4, 45, scale, histogram + 104);
				SIFT_histogram(gk_map, j + 5, i + 1, 45, scale, histogram + 112);
				SIFT_histogram(gk_map, j + 5, i + 5, 45, scale, histogram + 120);

				norm_histogram();
				InsertKeys(histogram_norm, i, j, scale, head, tail);
			}
			else {
				//printf("wrong scale!!\n");
				continue;
			}
			//printf("\n-----------\n");
			max_histogram = -1;
			ori = 0;
			trigger = 0;
		}
	}
	free(histogram);
	free(histogram_norm);

	printf("%d Feature Points!!,%d\n", num_Keypoint, n1);
	//printKeys(head,tail);
}


///////////////////////// SIFT at feature point
// need to apply weighted gaussian filtering at gradient magnitude
void SIFT_histogram(Mat gray, int y, int x, int bin_num, double scale, double *histo_array) {				// (x,y) : center point of window
	int xx, yy;
	double gradi_magnitude = 0.0;
	double tan = 0.0;
	double s = 1.6, ks = 1.6*1.6;
	int index;

	// make SIFT histogram
	if (trigger == 0) {
		for (int j = -2; j <= 2; j++) {
			for (int i = -2; i <= 2; i++) {
				xx = x + i;
				yy = y + j;
				gradi_magnitude += pow(double(gray.at<uchar>(yy, xx + 1) - gray.at<uchar>(yy, xx - 1)), 2);
				gradi_magnitude += pow(double(gray.at<uchar>(yy + 1, xx) - gray.at<uchar>(yy - 1, xx)), 2);
				gradi_magnitude = sqrt(gradi_magnitude);
				/*
				if (scale == s)
				{
					gradi_magnitude *= gaussian_kernel[i + 2][j + 2];
				}
				else if (scale == ks)
				{
					gradi_magnitude *= gaussian_kernel_k[i + 2][j + 2];
				}
				else
					printf("wrong scale!!\n");
					*/
				tan += (double)(gray.at<uchar>(yy + 1, xx) - gray.at<uchar>(yy - 1, xx));
				if (gray.at<uchar>(yy, xx + 1) - gray.at<uchar>(yy, xx - 1) != 0) {
					tan /= (double)(gray.at<uchar>(yy, xx + 1) - gray.at<uchar>(yy, xx - 1));
					tan = atan(tan) * 180.0 / Pi;
					if (tan < 0)
						tan += 360.0;
				}
				else {
					tan < 0 ? tan = 270.0 : tan = 90.0;
				}

				tan /= (double)bin_num;
				index = (int)tan;
				if (index < 0)
					index += 36;
				else if (index >= 36)
					index = 0;
				//printf("%lf, %d\n",tan, index);
				*(histo_array + index) += gradi_magnitude;
				gradi_magnitude = 0.0;
				tan = 0.0;
			}
		}
	}
	else{
		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				xx = x + i;
				yy = y + j;
				gradi_magnitude += pow(double(gray.at<uchar>(yy, xx + 1) - gray.at<uchar>(yy, xx - 1)), 2);
				gradi_magnitude += pow(double(gray.at<uchar>(yy + 1, xx) - gray.at<uchar>(yy - 1, xx)), 2);
				gradi_magnitude = sqrt(gradi_magnitude);
				/*
				if (scale == s)
				{
					gradi_magnitude *= gaussian_kernel[i + 2][j + 2];
				}
				else if (scale == ks)
				{
					gradi_magnitude *= gaussian_kernel_k[i + 2][j + 2];
				}
				else
					printf("wrong scale!!\n");
				*/
				tan += (double)(gray.at<uchar>(yy + 1, xx) - gray.at<uchar>(yy - 1, xx));
				if (gray.at<uchar>(yy, xx + 1) - gray.at<uchar>(yy, xx - 1) != 0) {
					tan /= (double)(gray.at<uchar>(yy, xx + 1) - gray.at<uchar>(yy, xx - 1));
					tan = atan(tan) * 180.0 / Pi;
					if (tan < 0)
						tan += 360.0;
				}
				else {
					tan < 0 ? tan = 270.0 : tan = 90.0;
				}
				
				tan /= (double)bin_num;
				index = (int)tan;
				index -= ori;
					
				if (index < 0)
					index += 8;
				else if (index >= 8)
					index = 0;
				//printf("%lf, %d\n",tan, index);
				*(histo_array + index) += gradi_magnitude;
				gradi_magnitude = 0.0;
				tan = 0.0;
			}
		}
	}
}

void norm_histogram() {
	double max = -999.0, min = 999.0, term, tmp;
	for (int m = 0; m < 128; m++) {
		histogram[m] -= max_histogram;
		//printf("%-2lf\t", histogram[m]);
	}
	
	for (int j = 0; j < 128; j++) {
		tmp = histogram[j];
		if (min > tmp)
			min = tmp;
		if (max < tmp)
			max = tmp;
	}
	term = max - min;
	if (term == 0)
		term = 1;
	for (int i = 0; i < 128; i++) {
		histogram_norm[i] = (int)((histogram[i] - min) * 255.0 / term);
		//printf("%d\t", arr[j*width+i]);
	}
}


void InsertKeys(int descrip_array[], int x, int y, double scale, Keypoint *head, Keypoint *tail)
{
	int len = 128;
	int val;
	num_Keypoint++;

	Keypoint *kp, *key;

	/* Allocate memory for the keypoint. */
	key = (Keypoint*)malloc(sizeof(Keypoint));

	// insert node
	kp = head;

	while (kp->next != tail) kp = kp->next;
	kp->next = key;
	key->next = tail;

	key->scale = scale;
	key->x = x;
	key->y = y;
	key->descrip = (int*)calloc(len, sizeof(int));

	// value
	for (int j = 0; j < len; j++) {
		val = descrip_array[j];
		if (val > 0 && val < 255)
			key->descrip[j] = (int)val;
	}
	for (int i = 0; i < 128; i++) {
		histogram[i] = 0;
		histogram_norm[i] = 0;
	}
}

void printKeys(Keypoint *head, Keypoint *tail)
{
	int x, y, len = 128;
	int n = 0;
	double scale;
	Keypoint *kp;
	int tmp_array[128] = { 0 };
	printf("Print Featurepoint!");

	kp = head;

	while (kp->next != tail) {
		kp = kp->next;
		x = kp->x;
		y = kp->y;
		scale = kp->scale;
		printf("%dth point x : %d, y : %d, scale : %lf\n", n++, x, y, scale);
		for (int j = 0; j < len; j++) {
			tmp_array[j] = kp->descrip[j];
			printf("descrip %d : %d\n", j, tmp_array[j]);
		}
	}
}

void deleteKeys(Keypoint *head, Keypoint *tail)
{
	double scale;
	Keypoint *kp,*kp_next;
	kp = head->next;

	while (kp != tail) {
		kp_next = kp->next;
		free(kp);
		kp = kp_next;
	}
	head->next = tail;
}

Mat CombineImagesVertically(Mat im1, Mat im2)
{
	int height, width, r, c;

	height = im1.rows;
	width = im1.cols;
	Mat result = Mat::zeros(height, width * 2, CV_8UC1);

	/* Copy images into result. */
	for (r = 0; r < height; r++)
		for (c = 0; c < width; c++)
			result.at<uchar>(r, c) = im1.at<uchar>(r, c);
	for (r = 0; r < height; r++)
		for (c = 0; c < width; c++)
			result.at<uchar>(r, c + width) = im2.at<uchar>(r, c);

	return result;
}

Mat FindMatches(Mat im1, Mat im2, Keypoint *head1, Keypoint *head2, Keypoint *tail)
{
	Mat result;
	Mat result_tmp;
	result_tmp = CombineImagesVertically(im1, im2);
	cvtColor(result_tmp, result, COLOR_GRAY2BGR);

	Keypoint *kp, *match;
	int count = 0;

	printf("next\n");
	// Match the keys in list keys1 to their best matches in keys2.
	kp = head1->next;
	while (kp != tail) {
		match = CheckForMatch(kp, head2->next, tail);

		/* Draw a line on the image from keys1 to match.  Note that we
	   must add row count of first image to row position in second so
	   that line ends at correct location in second image. */

		if (match != NULL) {
			count++;
			circle(result, Point(kp->x, kp->y), 3, Scalar(0, 255, 0), 2, 8, 0); // red circle
			circle(result, Point((match->x) + im1.cols, match->y), 3, Scalar(0, 255, 0), 2, 8, 0); // red circle
			line(result, Point(kp->x, kp->y), Point((match->x) + im1.cols, match->y), Scalar(255, 0, 0), 1, 8);
		}
		kp = kp->next;
	}
	printf("count : %d\n", count);

	////////////////////////// circle feature point //////////////////////////////
	/*
	kp = head1->next;
	while (kp != tail) {
		circle(result, Point(kp->x, kp->y), 1, Scalar(0, 0, 255), 2, 8, 0); // red circle
		kp = kp->next;
	}
	kp = head2->next;
	while (kp != tail) {
		circle(result, Point(kp->x+im1.cols, kp->y), 1, Scalar(0, 0, 255), 2, 8, 0); // red circle
		kp = kp->next;
	}
	printf("finish\n");
	*/

	return result;
}

Keypoint* CheckForMatch(Keypoint *key, Keypoint *klist, Keypoint *tail)
{
	int dsq, distsq1 = 100000000, distsq2 = 100000000;
	Keypoint *kp, *minkey = NULL;

	// Find the two closest matches, and put their squared distances in distsq1 and distsq2.
	for (kp = klist; kp != tail; kp = kp->next) {
		dsq = DistSquared(key, kp);
		
		if (dsq < distsq1) {
			distsq2 = distsq1;
			distsq1 = dsq;
			minkey = kp;
		}
		else if (dsq < distsq2) {
			distsq2 = dsq;
		}
	}

	/* Check whether closest distance is less than 0.6 of second. */
	if (10 * 10 * distsq1 < 7 * 8 * distsq2)
		return minkey;
	else return NULL;
}


// Return squared distance between two keypoint descriptors.
int DistSquared(Keypoint *k1, Keypoint *k2)
{
	int i, dif, distsq = 0;
	int *pk1, *pk2;
	if (k1->scale != k2->scale)
		return 100000000;
	pk1 = k1->descrip;
	pk2 = k2->descrip;

	for (i = 0; i < 128; i++) {
		dif = pk1[i] - pk2[i];
		distsq += dif * dif;
	}
	return distsq;
}