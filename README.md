# DACON_2301 : 포디블록 구조 추출 AI 경진대회
- Computer Vision
- Multi-label Classification


## 1. 대회 결과
- 최종 성적
    - Public  :
        - **Accuracy : 0.95981  ( 22 / 430 )**
            - 1위 : 0.97876
    - Private :
        - **Accuracy : 0.9541  ( 23 / 461 , top 5% )**
            - 1위 : 0.9726

## 2. 대회 개요
- 주최 : 포디랜드, AI Frenz
- 주관 : 데이콘
- 대회기간 : 2022.0102 ~ 2023.01.30
- Task
    - **multi-label image classification**
    - 2D 이미지 기반 블록 구조 추출 AI 모델 개발
- Data : block images
    - train data
        - 32,994개
    - test data
        - 1,460개
- 상금 : 총 1000만원
    - 1위 : 500만 원  
    - 2위 : 300만 원  
    - 3위 : 100만 원  
    - 4위 : 100만 원  

## 3. 진행 과정
### 데이터 전처리
- train data에는 배경 X, test data에는 배경 O  
    - image segmentation과 관련된 pretrained model 을 이용하여 background removal 진행  
    - train data 및 test data에 모두 적용
        - data leakeage? 문제 없음. 왜냐하면 train data에 fit 된 모델이기 때문에  
- image는 (384+64 , 384+64)로 resize 적용 후, (384, 384)로 center crop 적용

### 학습
- backbone : ConvNeXt large (in22k, in1k, @ 384)  
- loss function : BCEWithLogitsLoss  
- optimizer : madgradw  
- scheduler : OneCycleLR    
- model ensemble : soft voting (24 models)

## 4. Self-feedback?
- 성적도 좋았고, pytorch-lightning API를 배웠다는 점에 만족  # Dacon2301_4dblocks
