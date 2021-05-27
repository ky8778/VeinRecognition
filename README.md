# Vein Recognition

- 프로젝트 성격 : Computer Vision
- 개발기간 : 7개월 (2019.01 ~ 2019.07)
- 개발환경 : window OS, visual studio (c++, opencv)
- 프로젝트 목적 : 적외선 센서로 촬영된 손바닥 이미지를 이용해 같은 사람의 손인지 아닌지 판별. EER 5%
- 내용 : 적외선 센서로 촬영된 손바닥 이미지의 정맥 기반 인식
- DataBase : CASIA-Palmprint images (by NIR camera)

## Project Process

1. ROI(Region Of Interest) Extraction

   흑백이미지인 손바닥 사진에서 손 영역은 백색에 가깝고 배경은 흑색에 가까운 특성을 이용해 손바닥 영역(`ROI`)을 추출

   - Binarization : two class classification

     > 0/1 둘 중 하나의 값으로 영상을 분류하는 과정

   - Extraction Algorithm

     > 이진화된 이미지에서 손바닥(`ROI`)영역 추출

   - Image Pre-processing

2. Extract description value or histogram of ROI

3. Matching Algorithm

## V1

### ROI Extraction : Binarization

**1) Global fixed thresholding**

고정임계값 T를 기준으로 영상 픽셀을 분류

- 결과 : 임계값을 조절한다 하더라도 손가락 마디들이 끊긴상태 혹은 매우 뭉툭한 손으로 분류됨 손과 배경을 구분했다고 하기 어려운 상태

- 개선 : Morphological Image Processing

  끊긴 손을 연결하기 위한 방법으로 Erosion과 Dilation 방식으로 끊긴 부분을 연결해보려 시도

  효과가 있긴 하였지만 수백장의 data 에 동일하게 효과있지 않음



**2) Locally adaptive thresholding**

지역 가변 임계값을 사용하는 방식

NxN 주변 영역의 밝기 평균에 일정한 상수 C를 빼서 임계값을 결정하는 방식

- 결과 : N과 C를 조절해가며 실험해보았지만 이또한 수백장의 data 에 동일하게 효과있지 않음


**3) Otzu's Binarization (적용)**

Thresholding은 결국 Histogram을 구분하는 임계를 정하는 것이 핵심

data의 밝기 Histogram(0~255)은 두개의 봉우리가 생기는 Histogram으로 이루어짐.

영상 픽셀들을 두 클래스로 분류했을 때 두 클래스 간의 intra-class variance를 최소화하거나 또는 inter-class variance를 최대화하는 T를 찾는 이진화 방법 `Otzu's Method` 적용

- 결과 : 모든 data에 가변적으로 최적의 Binarization 가능

- 추가 개선 : Morphological Image Processing를 추가로 적용하여 이미지 보정.



### ROI Extraction : Extraction Algorithm

**Find the maximum inscribed circle in palm**

1. Palm-Centroid calculation : 손바닥의 무게중심점 계산
2. Find the maximum inscribed circle in palm : 무게중심점으로부터 원의 크기를 늘려가며 최대반지름 탐색
3. Rotate Image

- 결과 : 연산시간이 오래걸리며 손바닥 바깥영역도 함께 추출되는 data 존재

- 개선

  - 이진탐색 알고리즘을 적용하여 연산시간 감소 (TODO)

  - Binarization이 매우 정교하지 않은 이미지에 대해서는 손바닥 바깥영역이 벗어나는 문제가 해결되지 않음.

    (손바닥 아래쪽은 빛이 많이 들어오는 경우가 많았기 때문에 그 부분은 Binarization에 한계가 있음)



### Descriptor/Histogram of ROI description

**Multi Level Keypoint Detection**

1. Harris cornerness : 1st order 변화량

2. Hessian blobness : 2nd order 변화량

3. Difference of Gaussian (at OGMs)

   OGMs : Oriented Gradient Maps 8가지 방향의 변화량으로 만든 maps, 8가지 방향에 대한 변화량으로 된 maps

- 결과 : Description으로 Matching 진행했을 때 결과가 안좋음.




## V2

### ROI Extraction : Extraction Algorithm (추가)

Maximum inscribed circle 내에서 내접하는 정사각형 ROI 추출



### ROI Extraction : Image Preprocessing (추가)

**Image Resizing**

Descriptor를 추출하여 비교하는 과정에서 Image의 Size가 동일하게 조정

Bilinear Interpolation 방식을 사용해서 정교하게 Size를 특정 ROI size에 맞추어 resize

- 결과 : 영상 보정 없이 사이즈만 조절해주는 방식으로 개선이 되지 않음.



### Matching Algorithm (추가)

**SIFT feature Matching**

- Orientation assignment
- Keypoint descriptor

16x16 block 에서 main direction을 잡고 sub block 에서 feature vector를 얻는 방식.

Multi Level Keypoint Detection 에서 검출된 점들을 SIFT 한 뒤 BruteForce Matching.

- 결과 : Description으로 Matching 진행했을 때 결과가 안좋음.

