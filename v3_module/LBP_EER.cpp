#include "vein_ROI.h"
#include "vein_match.h"

//#define ROI_check
//#define samehand
#define differenthand
#define NUM_IMAGE 6


const char* ex = "_l_940_0";
const char* jpg = ".jpg";
const char* str_error = "error_";
char* source1;//"ex?.jpg"
char* source2;//"ex?.jpg"
char* source3;//"error.jpg"
char* source4;//"error.jpg"
extern char **ROI_write;//"result_~~~~.jpg"

char img_n1[10], img_n2[10];
char img_n[10];
char hand[105];

Mat src1, src2;
Mat ROI1, ROI2;

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
	source3 = (char*)malloc(30 * sizeof(char));
	source4 = (char*)malloc(30 * sizeof(char));

#ifdef ROI_check
	ROI_write = (char**)malloc(4 * sizeof(char*));
	for(int i=0;i<4;i++)
		ROI_write[i] = (char*)malloc(30 * sizeof(char));

	for (int hand_index = 1; hand_index <= 100; hand_index++) {
		for (int index = 1; index <= NUM_IMAGE; index++) {
			if (hand_index == 21 && index == 2) continue;
			strcpy(ROI_write[0], "result");
			strcpy(ROI_write[1], "result");
			strcpy(ROI_write[2], "result");
			strcpy(ROI_write[3], "result");

			if (hand_index == 100) {
				strcpy(source1, "100");
			}
			else if (hand_index < 10) {
				strcpy(source1, "00");
				_itoa(hand_index, hand, 10);
				strcat(source1, hand);
			}
			else {
				strcpy(source1, "0");
				_itoa(hand_index, hand, 10);
				strcat(source1, hand);
			}

			_itoa(index, img_n1, 10);

			strcat(ROI_write[0], source1);
			strcat(ROI_write[1], source1);
			strcat(ROI_write[2], source1);
			strcat(ROI_write[3], source1);
			strcat(ROI_write[0], "_");
			strcat(ROI_write[1], "_");
			strcat(ROI_write[2], "_");
			strcat(ROI_write[3], "_");
			strcat(ROI_write[0], img_n1);
			strcat(ROI_write[1], img_n1);
			strcat(ROI_write[2], img_n1);
			strcat(ROI_write[3], img_n1);
			strcat(ROI_write[0], "_1");
			strcat(ROI_write[1], "_2");
			strcat(ROI_write[2], "_3");
			strcat(ROI_write[3], "_4");
			strcat(ROI_write[0], jpg);
			strcat(ROI_write[1], jpg);
			strcat(ROI_write[2], jpg);
			strcat(ROI_write[3], jpg);

			strcat(source1, ex);
			strcat(source1, img_n1);
			strcat(source1, jpg);

			printf("%s\t%s\t", source1, ROI_write[0]);
			//////////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////// 1st IMAGE //////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////

			src1 = imread(source1, IMREAD_GRAYSCALE);
			if (src1.empty())
			{
				cout << "Could not open or find the image!\n" << endl;
				return -1;
			}

			///////////////////////// ROI extraction ////////////////////////////////
			ROI1 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);
			ROI_extraction(src1, ROI1);
			imwrite(ROI_write[3], ROI1);
			waitKey(500);
		}
	}
	free(ROI_write);
#endif

	int ROI_error;
#ifdef samehand
	fprintf(fp, "Same hand\n");
	for (int hand_index = 1; hand_index <= 100; hand_index++) {
		for (int index1 = 1; index1 < NUM_IMAGE; index1++) {
			for (int index2 = index1 + 1; index2 <= NUM_IMAGE; index2++) {
				if (hand_index == 21 && (index1 == 2 || index2 == 2)) {
					fprintf(fp, "wrong\n");
					continue;
				}
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

				strcpy(source3, str_error);
				strcpy(source4, str_error);
				strcat(source3, source1);
				strcat(source4, source2);

				printf("%s\t%s\t", source1, source2);
				fprintf(fp, "%s\t%s\t", source1, source2);
				//////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////// 1st IMAGE //////////////////////////////////////////
				//////////////////////////////////////////////////////////////////////////////////////

				src1 = imread(source1, IMREAD_GRAYSCALE);
				if (src1.empty())
				{
					cout << "Could not open or find the image!\n" << endl;
					return -1;
				}

				///////////////////////// ROI extraction ////////////////////////////////
				ROI1 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);
				ROI_error = ROI_extraction(src1, ROI1);
				if (ROI_error < 0) {
					fprintf(fp, "%s\n", source3);
					imwrite(source3, ROI1);
					continue;
				}

				//////////////////////////////////////////////////////////////////////////////////////
				///////////////////////////////// 2nd IMAGE //////////////////////////////////////////
				//////////////////////////////////////////////////////////////////////////////////////

				src2 = imread(source2, IMREAD_GRAYSCALE);
				if (src2.empty())
				{
					cout << "Could not open or find the image!\n" << endl;
					return -1;
				}

				///////////////////////// ROI extraction ////////////////////////////////
				ROI2 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);							// black 0 으로 초기화
				ROI_error = ROI_extraction(src2, ROI2);
				if (ROI_error < 0) {
					fprintf(fp, "%s\n", source4);
					imwrite(source4, ROI1);
					continue;
				}

				/* SURF 
				int count = vein_SURF(ROI1, ROI2);
				fprintf(fp, "%d\t", count);
				*/
				/* LBP/LDP */
				float result;
				//result = vein_LBP(ROI1, ROI2, 16, 7, 4);
				result = vein_LDP(ROI1, ROI2, 3, 6, 4);
				fprintf(fp, "%f\n", result);

			}
		}
	}
#endif
#ifdef differenthand
	fprintf(fp, "Different Hand\n");
	for (int t = 6; t < 10; t++) {
		for (int index1 = (10 * t) + 1; index1 < 10 * (t + 1); index1++) {
			for (int index2 = index1 + 1; index2 <= 10 * (t + 1); index2++) {
				for (int tt = 1; tt <= NUM_IMAGE; tt++) {
					if (tt == 2 && (index1 == 21 || index2 == 21)) {
						fprintf(fp, "wrong\n");
						continue;
					}
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

					strcpy(source3, str_error);
					strcpy(source4, str_error);
					strcat(source3, source1);
					strcat(source4, source2);

					printf("%s\t%s\t", source1, source2);
					fprintf(fp, "%s\t%s\t", source1, source2);

					//////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////// 1st IMAGE //////////////////////////////////////////
					//////////////////////////////////////////////////////////////////////////////////////

					src1 = imread(source1, IMREAD_GRAYSCALE);
					if (src1.empty())
					{
						cout << "Could not open or find the image!\n" << endl;
						return -1;
					}

					///////////////////////// ROI extraction ////////////////////////////////
					ROI1 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);
					ROI_error = ROI_extraction(src1, ROI1);
					if (ROI_error < 0) {
						fprintf(fp, "%s\n", source3);
						imwrite(source3, ROI1);
						continue;
					}
					//////////////////////////////////////////////////////////////////////////////////////
					///////////////////////////////// 2nd IMAGE //////////////////////////////////////////
					//////////////////////////////////////////////////////////////////////////////////////

					src2 = imread(source2, IMREAD_GRAYSCALE);
					if (src2.empty())
					{
						cout << "Could not open or find the image!\n" << endl;
						return -1;
					}

					///////////////////////// ROI extraction ////////////////////////////////
					ROI2 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);							// black 0 으로 초기화
					ROI_error = ROI_extraction(src2, ROI2);
					if (ROI_error < 0) {
						fprintf(fp, "%s\n", source4);
						imwrite(source4, ROI1);
						continue;
					}

					/* SURF 
					int count = vein_SURF(ROI1, ROI2);
					fprintf(fp, "%d\t", count);
					*/

					/* LBP/LDP */
					float result;
					//result = vein_LBP(ROI1, ROI2, 16, 7, 4);
					result = vein_LDP(ROI1, ROI2, 3, 6, 4);
					fprintf(fp, "%f\n", result);

				}
			}
		}
	}
#endif
	waitKey();

	free(source1);
	free(source2);
	free(source3);
	free(source4);
	fclose(fp);

	return 0;
}
