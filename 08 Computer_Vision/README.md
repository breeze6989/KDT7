컴퓨터 비전 처리를 활용한 새의 종류 분류 및 자체 제작 함수 업그레이드<hr>
사용모듈
- cv2
- numpy
- pandas
- sklearn
- pickle

**요약**

- 저화질의 새 이미지를 사용하여 고화질의 동일 새 이미지를 입력하였을 때 해당 새의 목,과,속,종을 분류 할 수 있는지 여러가지의 머신러닝 모델을 사용하여 분석

**역할**

- 목 분류 1인, 과 분류 1인, 속 분류 1인, 종 분류 1인, 데이터 분석 및 전처리 4인 전부
- 저화질의 새들을 ‘과’로 분류하여 고화질의 동일 이미지를 입력하였을 때 올바르게 분류하는지 확인

      

**사용 모델**

- Logistic, SVC, KNN, Randomforest, GradientBoost

**성과**

- 새 이미지를 어느 정도 분류하긴 하였으나 대부분의 새를 할미새과로 분류함
- GPU 사용을 통한 병렬처리가 불가능한 머신러닝의 한계를 느낄 수 있었으며 모형에 따른 격차가 심하고 데이터 분석에 있어 여러가지 모형을 사용하고 가장 적합한 모델을 찾는것이 필수적이라는 사실을 알 수 있었음

**시기**

- 프로젝트 진행 기간 ( 2025.3.23 ~ 2025.3.25) .
