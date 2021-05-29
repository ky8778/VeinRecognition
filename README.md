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

---

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

---

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

---

## V3

### ROI Extraction : Extraction Algorithm (변경)

**Extract square box (using in-between-finger points)**

1. Find Contours
2. Convex (Hulls & Defects)
3. Extract square box

- 결과 : 손 테두리의 점들로 부터 in-between-finger points를 잡고 ROI를 추출할 수 있으며 일정한 규칙으로 ROI 추출 가능

- 추가 개선 : 손바닥의 size에 따라 square box의 크기를 조절하여 ROI를 손바닥에서 벗어나지 않고 올바르게 추출해내었습니다.

**최적화 : 영상에서 벗어나는 영역 처리**

Description을 추출할 때 keypoint 방식이든, Histogram 방식이든 block으로 이미지를 나누어 추출하는 과정을 거친다.

이때 영역을 벗어나는 영역의 픽셀값을 필요로하는 경우가 생긴다.

- 반대편 영상 부분을 가져와 사용하는 방식.
- 0으로 처리하는 방식
- 필요한 만큼 ROI size를 block에 맞게 크게 추출하는 방식

세가지 방법을 사용해서 구현한 결과 마지막 방식을 사용하는 것이 결과가 가장 좋았다.

### ROI Extraction : Image Preprocessing (변경)

**Image Contrast Enhancement**

- Original Image

  Histogram으로 Binarization 했기 때문에, Histogram이 몰려있음.

  즉, Image의 Contrast가 매우 적기 때문에 다음 단계에서 특징을 추출하기 어려움.

- Normalization

  Histogram을 최저~최대를 0~255로 정규화시켜주는 방법.

- Histogram Equalization

  몰려있는 Histogram을 평활하게 만들어주는 방법.

- Contrast Limited Adaptive Histogram Equalization

부분마다 적응으로 Histogram을 평활하게 만들어주는 기법인 AHE를 적용하면 더 효과가 있을 것이라 판단하여 사용하였으나, 이미지에 노이즈가 있는 경우, 타일 단위의 히스토그램 균일화(AHE)를 적용하면 노이즈가 커짐.

노이즈를 감쇠시킬 수 있는 Contrast Limiting 기법을 적용.

세가지 방법을 모두 사용해서 결과를 얻었을 때 가장 성능이 좋았던 CLAHE 적용.

### Descriptor/Histogram of ROI description (변경)

**LBP (Local Binary Patterns) description Histogram**

> 이미지의 질감 표현에 활용되는 방법으로, 중심 픽셀을 기준으로 일정 방향으로(시계방향) 주위 8개의 픽셀의 이진패턴을 분석하는 방식.
>
> 값이 크거나 같으면 1로, 작으면 0으로 비트를 할당하여 8비트 값을 얻으며, 결과는 0~255까지 총 256개의 경우의 수에 LBP 히스토그램을 얻음.
>
> 즉, 하나의 영상의 질감을 256개의 숫자로 표현하는 것이다.
>
> LBP는 영상의 밝기가 변해도 robust하다는 장점이 있음.

- 결과 : 이전의 Description보다 성능이 향상되었으며 block의 크기 중 3x3의 성능이 가장 좋았음.

- 추가개선 : Uniform LBP

  어떤 패턴들은 좀 더 영상 내에서 자주 발견되는 반면 어떤 패턴들은 드물게 발견된다.

  0에서 1로의 변화 또는 1에서 0으로의 변화가 2번 이내인 패턴은 uniform 패턴이라고 명하고 각각 하나의 라벨을 부여해주고, 변화가 3번 이상인 패턴은 non-uniform 패턴이라고 명하고 한 그룹으로 묶은 다음 그룹 전체에 단 한 개의 라벨을 부여한다.

  이를 통해 8개의 이웃점을 고려하는 경우, 256개의 bin을 59개의 bin으로 대응시켜 빠르게 연산할 수 있다.

  rotation을 고려한 LBP 방식도 있었으나, 영상의 rotation은 ROI 단계에서 고정시켰다고 생각하여 적용하지 않았다.

**LDP (Local Derivative Pattern) description Histogram**

> LBP는 first-order non-directional patterns를 추출해내는 반면, LDP는 derivative direction variation information (nth-order) pattern을 추출해낸다.

- 결과 : LBP보다 성능 향상

### Matching Algorithm (변경)

**Histogram Matching : Chi Square test**

> 카이제곱검정 : 관찰된 빈도가 기대되는 빈도와 의미있게 다른지의 여부를 검증하기 위해 사용되는 검증방법
>
> x^2 = Sum((관측값 - 기댓값)^2/기댓값)

기본가정

1. 변인의 제한 : 종속변인이 명목변인에 의한 질적변인이거나 범주변인이어야 한다.
2. 무선표집 : 표본이 모집단에서 무선으로 추출되어야 한다.
3. 기대빈도의 크기 : 각 범주에 포함할 수 있도록 기대되는 빈도를 기대빈도라고 하는데, 이 기대빈도가 5 이상이어야 한다. 5보다 적으면 사례 수를 증가시켜야 한다.
4. 관찰의 독립 : 각 칸에 있는 빈도는 다른 칸의 사례와 상관없이 독립적이어야 한다.

- 결과 : 기본가정이 dataset에서의 Description Histogram에 적용하데 어려움이 있음.

##### Histogram Intersection

Dintersection(H1,H2) = Sum( min( H1(i) , H2(i) ) )

신호 element를 비교하여 최솟값을 합한 결과, 두 신호가 정규화되어있을 경우 신호가 같으면, 완전히 다르면 0이 된다.

이러한 값을 사용해서 Thresholding을 하여 Matching 결과를 만드는 방식을 사용.

- 결과 : 기존 Matching 방식보다 성능 향상

---

## EER Test

Equal Error Ratio Test

- Intra-group

  Pair-wise matching among six samples of each individual in the group

  100개의 그룹으로 각 손을 나누고 하나의 손 그룹에 해당되는 6개 중 2개씩 골라서 matching 진행. (일치해야함)

  6C2 * 100 = 1500 cases

- Inter-group

  Randomly divided all individuals into 10 groups

  10개씩 10개의 그룹으로 나누어 한 그룹에서 2개의 손을 고른 뒤 해당 손에 대해 6개의 sample에 대해 matching을 진행. (불일치해야함)

  10C2 * 6 * 10 = 270

| TA (TRUE ACCEPTANCE) | FA (FALSE ACCEPTANCE) |
| -------------------- | --------------------- |
| FR (FALSE REJECT)    | TR (TRUE REJECT)      |

4가지 영역에 대해 ratio를 구했을 때, FA ratio == FR ratio 인 지점이 Equal Error Ratio.

이 두가지 값은 trade off 관계 (FR을 줄이게 되면 matching 결과를 강하게 잡아줘야 하기 때문에 FA가 올라가게 되고, FA를 줄이게 되면 matching 결과를 약하게 잡아줘야 하기 때문에 FR이 올라가게 된다.)