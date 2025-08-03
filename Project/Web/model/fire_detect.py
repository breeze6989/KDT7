# 객체 탐지 모델 추론 후 return bbox좌표
# 성익씨
import random
import cv2
def fire_detect_f(img_path):
    # img=cv2.imread(img)
    # x,y=img.size()
    a=random.randint(0,1)
    if a:
        x_2=random.randint(0,560)
        y_2=random.randint(0,560)
        x_1=random.randint(0,x_2)
        y_1=random.randint(0,y_2)
        return x_1,y_1,x_2,y_2
    else: 
        None

print(fire_detect_f('a'))