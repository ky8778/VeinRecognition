#include "vein_ROI.h"
#include "vein_match.h"

#define NUM_IMAGE 6
#define START 2

//ex1~3 : same hand
//ex4~10 : different hand
const char* ex = "ex";
const char* surf = "surf";
const char* jpg = ".jpg";
const char* roi = "ROI";
const char* res = "result";
const char* source = "ex7.jpg";
char* source1;//"ex?.jpg"
char* source2;//"ex?.jpg"
//char* source3;//"surf?.jpg"
//char* source4;//"result?.jpg"
char img_n1[10],img_n2[10];

Mat src1, src2;
Mat ROI1, ROI2;



/////////////////////////////// main //////////////////////////////////
int main() {
	DWORD start;
	DWORD end;

	FILE *fp = NULL;
	fp = fopen("result.txt", "w");

	if (fp == NULL) {
		printf("File open error.\n");
		return 0;
	}

	source1 = (char*)malloc(15 * sizeof(char));
	source2 = (char*)malloc(20 * sizeof(char));
	//source3 = (char*)malloc(20 * sizeof(char));
	//source4 = (char*)malloc(20 * sizeof(char));

	for (int index1 = START; index1 < START+NUM_IMAGE; index1++) {
		for (int index2 = index1 + 1; index2 <= START+NUM_IMAGE; index2++) {
			start = GetTickCount();

			_itoa(index1, img_n1, 10);
			_itoa(index2, img_n2, 10);

			strcpy(source1, ex);
			strcat(source1, img_n1);
			strcat(source1, jpg);

			strcpy(source2, ex);
			strcat(source2, img_n2);
			strcat(source2, jpg);

			/*
			strcpy(source3, surf);
			strcat(source3, img_n1);
			strcat(source3, "vs");
			strcat(source3, img_n2);
			strcat(source3, jpg);

			strcpy(source4, res);
			strcat(source4, img_n1);
			strcat(source4, jpg);
			printf("%s\n%s\n%s\n%s\n", source1, source2,source3,source4);
			*/
			printf("%s\n%s\n", source1, source2);

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
			int ROI_error = ROI_extraction(src1, ROI1);
			if (ROI_error < 0) continue;
			//imshow("ROI1", ROI1);
			//waitKey();

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
			//ROI2 = Mat(ROI_SIZE, ROI_SIZE, CV_8UC1, Scalar(0));					// black 0 으로 초기화
			ROI2 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);							// black 0 으로 초기화
			ROI_error = ROI_extraction(src2, ROI2);
			if (ROI_error < 0) continue;
			//imshow("ROI2", ROI2);

			float result;
			vein_SURF(ROI1, ROI2);
			result = vein_LBP(ROI1, ROI2, 16, 7, 4);
			fprintf(fp, "%f\t", result);
			result = vein_LDP(ROI1,ROI2,3,6,4);
			fprintf(fp, "%f\n", result);
			printf("test\n");

			end = GetTickCount();
			printf("실행시간 : %lf\n", (end - start)/1000.);
		}
	}

	waitKey();

	free(source1);
	free(source2);
	//free(source3);
	//free(source4);
	fclose(fp);
	
	return 0;
}
