#include "lib_vein.h"

#define EER
#ifdef EER
const char* ex = "_l_940_0";
const char* jpg = ".jpg";
char* source1;
char* source2;

char img_n1[10], img_n2[10];
char img_n[10];
char hand[105];


/////////////////////////////// main //////////////////////////////////
int main() {

	FILE *fp = NULL;

	fp = fopen("result.txt", "w");


	if (fp == NULL) {
		printf("File open error.\n");
		return 0;
	}

	source1 = (char*)malloc(20 * sizeof(char));
	source2 = (char*)malloc(20 * sizeof(char));
	
	fprintf(fp, "Same hand\n");
	for (int hand_index = 1; hand_index <= 100; hand_index++) {
		for (int index1 = 1; index1 < NUM_IMAGE; index1++) {
			for (int index2 = index1 + 1; index2 <= NUM_IMAGE; index2++) {

				if (hand_index == 100) {
					strcpy(source1, "100");
					strcpy(source2, "100");
				}
				else if (hand_index < 10) {
					strcpy(source1, "00");
					strcpy(source2, "00");
					_itoa(hand_index, hand, 10);
					strcat(source1, hand);
					strcat(source2, hand);
				}
				else {
					strcpy(source1, "0");
					strcpy(source2, "0");
					_itoa(hand_index, hand, 10);
					strcat(source1, hand);
					strcat(source2, hand);
				}

				_itoa(index1, img_n1, 10);
				_itoa(index2, img_n2, 10);

				strcat(source1, ex);
				strcat(source1, img_n1);
				strcat(source1, jpg);

				strcat(source2, ex);
				strcat(source2, img_n2);
				strcat(source2, jpg);

				printf("%s\t%s\t", source1, source2);
				fprintf(fp, "%s\t%s\t", source1, source2);

				// error 넘기기
				//if ((hand_index == 30 && (index1 == 4 || index2 == 4))|| (hand_index == 74 && (index1 == 3 || index2 == 3))) {
				//	fprintf(fp, "error\n");
				//	continue;
				//}

				Mat src1 = imread(source1, IMREAD_GRAYSCALE);
				Mat src2 = imread(source2, IMREAD_GRAYSCALE);
				if (src1.empty() || src2.empty())
				{
					cout << "Could not open or find the image!\n" << endl;
					return -1;
				}

				///////////////////////// ROI extraction ////////////////////////////////
				Mat ROI1 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);
				Mat ROI2 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);							// black 0 으로 초기화
				ROI_extraction(src1, &ROI1);
				ROI_extraction(src2, &ROI2);

				float result;
				result = vein_LDP(ROI1, ROI2, 3, 6, 4);
				fprintf(fp, "%f\n", result);
			}
		}
	}
	

	fprintf(fp, "Different Hand\n");
	for (int t = 0; t < 10; t++) {
	//for (int t = 8; t < 9; t++) {
		for (int index1 = (10 * t) + 1; index1 < 10 * (t + 1); index1++) {
			for (int index2 = index1 + 1; index2 <= 10 * (t + 1); index2++) {
				for (int tt = 1; tt <= NUM_IMAGE; tt++) {
					if (index1 < 10) {
						strcpy(source1, "00");
						_itoa(index1, hand, 10);
						strcat(source1, hand);
					}
					else {
						strcpy(source1, "0");
						_itoa(index1, hand, 10);
						strcat(source1, hand);
					}

					if (index2 == 100) {
						strcpy(source2, "100");
					}
					else if (index2 < 10) {
						strcpy(source2, "00");
						_itoa(index2, hand, 10);
						strcat(source2, hand);
					}
					else {
						strcpy(source2, "0");
						_itoa(index2, hand, 10);
						strcat(source2, hand);
					}

					_itoa(tt, img_n, 10);

					strcat(source1, ex);
					strcat(source1, img_n);
					strcat(source1, jpg);

					strcat(source2, ex);
					strcat(source2, img_n);
					strcat(source2, jpg);


					printf("%s\t%s\t", source1, source2);
					fprintf(fp, "%s\t%s\t", source1, source2);

					// error 넘기기
					//if ((tt == 4 && (index1 == 30 || index2 == 30)) || (tt == 3 && (index1 == 74 || index2 == 74))) {
					//	fprintf(fp, "error\n");
					//	continue;
					//}
					if (index1 == 86 || index2 == 86) {
						fprintf(fp, "error\n");
						continue;
					}

					Mat src1 = imread(source1, IMREAD_GRAYSCALE);
					Mat src2 = imread(source2, IMREAD_GRAYSCALE);
					if (src1.empty() || src2.empty())
					{
						cout << "Could not open or find the image!\n" << endl;
						return -1;
					}

					///////////////////////// ROI extraction ////////////////////////////////
					Mat ROI1 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);
					Mat ROI2 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);							// black 0 으로 초기화
					ROI_extraction(src1, &ROI1);
					ROI_extraction(src2, &ROI2);

					float result;
					result = vein_LDP(ROI1, ROI2, 3, 6, 4);
					fprintf(fp, "%f\n", result);
				}
			}
		}
	}
	waitKey();

	free(source1);
	free(source2);
	fclose(fp);

	return 0;
}
#endif


//#define simple_test
#ifdef simple_test

int main() {
	Mat src1 = imread("t1.jpg", IMREAD_GRAYSCALE);
	Mat src2 = imread("t2.jpg", IMREAD_GRAYSCALE);
	//Mat src3 = imread("ex3.jpg", IMREAD_GRAYSCALE);
	//Mat src4 = imread("ex4.jpg", IMREAD_GRAYSCALE);
	//Mat src5 = imread("ex5.jpg", IMREAD_GRAYSCALE);
	Mat dst1(ROI_SIZE, ROI_SIZE, CV_8UC1);
	Mat dst2(ROI_SIZE, ROI_SIZE, CV_8UC1);
	//Mat dst3(ROI_SIZE, ROI_SIZE, CV_8UC1);
	//Mat dst4(ROI_SIZE, ROI_SIZE, CV_8UC1);
	//Mat dst5(ROI_SIZE, ROI_SIZE, CV_8UC1);
	ROI_extraction(src1, &dst1);
	ROI_extraction(src2, &dst2);
	//ROI_extraction(src3, &dst3);
	//ROI_extraction(src4, &dst4);
	//ROI_extraction(src5, &dst5);
	imshow("result1", dst1);
	imshow("result2", dst2);
	//imshow("result3", dst3);
	//imshow("result4", dst4);
	//imshow("result5", dst5);
	float result;
	result = vein_LDP(dst2, dst1, 3, 6, 4);
	printf("result : %f\n", result);
	//printf("result : %d\n", vein_SURF(dst1, dst2));
	
	waitKey();
	return 0;
}
#endif