전이학습(efficientdet)을 활용한 객체탐지<hr>
사용모듈
- torch
- torchvision
- PIL
- lxml
- pathlib
- tqdm
- effdet

**요약**

- 보안검색대 위해물품 X-ray 데이터와 객체인식 모델을 사용하여 위해물품 인식 시도

**역할**

- RetinaNet 사용 1인, EfficientDet 사용 1인, Faster CNN 사용 1인, YOLOv5 사용 1인, 데이터 전처리 및 bbox처리 4인 전부
- EfficientDet 모델을 사용하여 위해물품을 분류할 수 있는지 시도

      

**성과**

- EfficientDet에 대한 지식과 시간 부족으로 객체 인식에 실패함
- 전이학습이 마냥 쉬운 것이 아니라 모델 구조에 대한 이해가 필수적이며 각 모델마다의 사용법을 익히는 것이 필수적이라는 것을 알았음

**시기**

- 프로젝트 진행 기간 ( 2025.4.21 ~ 2025.4.22) .
