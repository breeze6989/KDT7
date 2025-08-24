# quick_check.py
import cv2, sys
cap = cv2.VideoCapture(sys.argv[1], cv2.CAP_FFMPEG)
f = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    print('read frame', f); f += 1
cap.release()
print('총 프레임:', f)
