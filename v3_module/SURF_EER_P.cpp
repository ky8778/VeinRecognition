#include "vein_ROI.h"
#define NUM_IMAGE 6

const char* ex = "_l_940_0";
const char* surf = "surf";
const char* vs = "vs";
const char* jpg = ".jpg";
char* source1;							//"ex?.jpg"	-> base image
char* source2;							//"ex?.jpg"	-> image
char* source3;							//"surf?vs?.jpg"
char img_n[10], base_img_n[10];
char hand[105];
char ex_[12];

Mat src1, src2;
Mat ROI1, ROI2;

int minHessian = 200;					//400;

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
	for (int hand_index = 87; hand_index <= 100; hand_index++) {
		printf("HAND %d\n",hand_index);
		for (int base_index = 1; base_index < NUM_IMAGE; base_index++) {
			if (hand_index != 100) {
				strcpy(ex_, "0");
				_itoa(hand_index, hand, 10);
				strcat(ex_, hand);
			}
			else {
				_itoa(hand_index, hand, 10);
				strcpy(ex_, hand);
			}

			_itoa(base_index, base_img_n, 10);

			strcpy(source1, ex_);
			strcat(source1, ex);
			strcat(source1, base_img_n);
			strcat(source1, jpg);

			//////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////// 1st IMAGE //////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////
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

			// normalization
			int ROI_max = -1, ROI_min = 999, ROI_term, ROI_tmp;
			for (int j = 0; j < ROI_SIZE; j++) {
				for (int i = 0; i < ROI_SIZE; i++) {
					ROI_tmp = ROI1.at<uchar>(j, i);
					if (ROI_min > ROI_tmp)
						ROI_min = ROI_tmp;
					if (ROI_max < ROI_tmp)
						ROI_max = ROI_tmp;
				}
			}
			ROI_term = ROI_max - ROI_min;
			if (ROI_term == 0)
				ROI_term = 1;
			for (int j = 0; j < ROI_SIZE; j++) {
				for (int i = 0; i < ROI_SIZE; i++) {
					ROI1.at<uchar>(j, i) = (int)((ROI1.at<uchar>(j, i) - ROI_min) * 255 / ROI_term);
				}
			}



			//////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////
			///////////////////////////////// 2nd IMAGE //////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////
			//////////////////////////////////////////////////////////////////////////////////////
			for (int index = base_index + 1; index <= NUM_IMAGE; index++) {
				_itoa(index, img_n, 10);

				strcpy(source2, ex_);
				strcat(source2, ex);
				strcat(source2, img_n);
				strcat(source2, jpg);
				
				strcpy(source3, surf);
				strcat(source3, base_img_n);
				strcat(source3, vs);
				strcat(source3, img_n);
				strcat(source3, jpg);
				printf("%s\n%s\n", source1,source2);

				src2 = imread(source2, IMREAD_GRAYSCALE);
				if (src2.empty())
				{
					cout << "Could not open or find the image!\n" << endl;
					return -1;
				}

				///////////////////////// ROI extraction ////////////////////////////////
				ROI2 = Mat::zeros(ROI_SIZE, ROI_SIZE, CV_8UC1);							// black 0 으로 초기화
				ROI_extraction(src2, ROI2);

				// normalization
				ROI_max = -1;
				ROI_min = 999;
				for (int j = 0; j < ROI_SIZE; j++) {
					for (int i = 0; i < ROI_SIZE; i++) {
						ROI_tmp = ROI2.at<uchar>(j, i);
						if (ROI_min > ROI_tmp)
							ROI_min = ROI_tmp;
						if (ROI_max < ROI_tmp)
							ROI_max = ROI_tmp;
					}
				}
				ROI_term = ROI_max - ROI_min;
				if (ROI_term == 0)
					ROI_term = 1;
				for (int j = 0; j < ROI_SIZE; j++) {
					for (int i = 0; i < ROI_SIZE; i++) {
						ROI2.at<uchar>(j, i) = (int)((ROI2.at<uchar>(j, i) - ROI_min) * 255 / ROI_term);
					}
				}

				////////////////////////////// SURF /////////////////////////////////////////////////
				//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
				Ptr<SURF> detector = SURF::create(minHessian);
				std::vector<KeyPoint> keypoints1, keypoints2;
				Mat descriptors1, descriptors2;
				detector->detectAndCompute(ROI1, noArray(), keypoints1, descriptors1);
				detector->detectAndCompute(ROI2, noArray(), keypoints2, descriptors2);
				printf("%d %d\n", descriptors1.rows, descriptors2.rows);

				// matching 점이 없는 경우
				if (descriptors1.rows < 2 || descriptors2.rows < 2)
				{
					Mat img_matches;
					hconcat(ROI1, ROI2, img_matches);
					printf("0 Matching \n\n");
					//-- Show detected matches
					imshow("Good Matches", img_matches);
					imwrite(source3, img_matches);
					continue;
				}

				//-- Step 2: Matching descriptor vectors with a FLANN based matcher
				// Since SURF is a floating-point descriptor NORM_L2 is used
				Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
				std::vector< std::vector<DMatch> > knn_matches;								// 2차원 vector
				matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);				// 2차원 vector에서 channel 수 2



				//-- Filter matches using the Lowe's ratio test
				const float ratio_thresh = 0.7f;
				std::vector<DMatch> good_matches;
				int count = 0;
				for (size_t i = 0; i < knn_matches.size(); i++)
				{
					if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
					{
						good_matches.push_back(knn_matches[i][0]);
						count++;
					}
				}
				printf("count : %d\n\n", count);
				fprintf(fp, "%d\t", count);

				//-- Draw matches
				Mat img_matches;
				drawMatches(ROI1, keypoints1, ROI2, keypoints2, good_matches, img_matches, Scalar::all(-1),
					Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				//-- Show detected matches
				imshow("Good Matches", img_matches);
				imwrite(source3, img_matches);

				detector.reset();
				keypoints1.clear();
				keypoints2.clear();
				matcher.reset();
				knn_matches.clear();
				good_matches.clear();

			}
		}
		fprintf(fp, "\n");
	}
	free(source1);
	free(source2);
	free(source3);

	fclose(fp);
	waitKey(1000);

	return 0;
}
