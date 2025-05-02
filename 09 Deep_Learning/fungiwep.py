#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fungi_web.py
URL 예시: http://localhost:8080/cgi-bin/fungiweb.py
"""

import sys, codecs, cgi, base64, os, tempfile, pickle
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# ---------------- 모델 정의 ----------------
class FungiCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        self.dropout_rate = 0.25

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------------- 모델 & 인코더 로딩 ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"F:\KDT7\10_deep_learning\team_project\cgi-bin\fungi_cnn_best_trainBased.pth"
labelenc_path = r"F:\KDT7\10_deep_learning\team_project\cgi-bin\fungi_label_encoder.pkl"

try:
    with open(labelenc_path, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    model = FungiCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except:
    model = None
    le = None
# ---------------- 영어 라벨 → 한국어 변환 ----------------
label_translation = {
    "Neotropic": "신열대구",
    "Palearctic": "구북구",
    "Afrotropic": "아프리카열대구",
    "Indomalaya": "인도말라야구",
    "Australasia": "오스트랄라시아구",
    "Nearctic": "근북구",
    "Oceania": "오세아니아",
    "Antarctic": "남극",
    "알수없음": "알 수 없음"
}

# ---------------- 추론 함수 ----------------
def classify_fungi(img_data):
    if (model is None) or (le is None):
        return "❌ 모델이 로드되지 않았습니다."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_data)
        tmp_name = tmp.name

    img = cv2.imread(tmp_name)
    os.remove(tmp_name)
    if img is None:
        return "❌ 이미지 로딩 실패"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (128, 128))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tensor_img = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor_img)
        probs = torch.softmax(out, dim=1)[0]
        idx = probs.argmax().item()  # ✅ 예측 클래스 인덱스
        region_name_en = le.inverse_transform([idx])[0]
        region_name_ko = label_translation.get(region_name_en, region_name_en)
        confidence = probs[idx].item() * 100


    return f"이 곰팡이는 <b style='color: #e67e22'>{region_name_ko}</b> 환경에 분포합니다!<br>(신뢰도: <b>{confidence:.1f}%</b>)"


# ---------------- 웹 출력 함수 ----------------
def print_html_form(msg="", img_b64=None):
    print("Content-Type: text/html; charset=utf-8\n")
    img_html = ""
    if img_b64:
        img_html = f"""
        <div style='margin-top: 20px;'>
            <h3>업로드한 이미지</h3>
            <img src="data:image/png;base64,{img_b64}" style="max-width: 300px; max-height: 300px;">
        </div>
        """
    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>곰팡이 분류기</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                background-color: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 90%;
                max-width: 700px;
                margin: auto;
            }}
            .result {{
                margin-top: 20px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                font-size: 1.1em;
            }}
            input[type="file"] {{
                margin-top: 20px;
            }}
            input[type="submit"] {{
                background-color: #27ae60;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            input[type="submit"]:hover {{
                background-color: #219150;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🌿 곰팡이 분포환경 분류기</h2>
            <p>곰팡이 이미지를 업로드하면 어떤 지역 환경에 속하는지 추론합니다.</p>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="fungi_img" accept="image/*">
                <br>
                <input type="submit" value="분석하기">
            </form>
            {img_html}
            <div class="result">{msg}</div>
        </div>
    </body>
    </html>
    """
    print(html)

# ---------------- 메인 처리 ----------------
def main():
    form = cgi.FieldStorage()
    if "fungi_img" in form:
        fileitem = form["fungi_img"]
        if fileitem.filename:
            img_bytes = fileitem.file.read()
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            result_msg = classify_fungi(img_bytes)
            print_html_form(msg=result_msg, img_b64=b64_str)
            return
    print_html_form(msg="🖼 이미지를 업로드해주세요!")

if __name__ == "__main__":
    main()
