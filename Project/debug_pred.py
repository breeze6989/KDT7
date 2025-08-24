# debug_pred.py
import cv2, torch, numpy as np, sys
from ultralytics.nn.autobackend import AutoBackend
model = AutoBackend(sys.argv[2])
FIRE   = next((i for i,n in model.names.items() if 'fire' in n.lower() or 'flame' in n.lower()), 0)
cap = cv2.VideoCapture(sys.argv[1])
for _ in range(5):           # 앞쪽 5프레임만 체크
    ret,frame=cap.read()
    if not ret: break
    img=cv2.resize(frame,(640,640))[:, :, ::-1].transpose(2,0,1)
    img=np.ascontiguousarray(img,dtype=np.float32)/255.0
    img=torch.from_numpy(img).unsqueeze(0)
    pred=model(img)[0]
    print(pred[:,5], pred[:,4])   # class id, conf
cap.release()