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
