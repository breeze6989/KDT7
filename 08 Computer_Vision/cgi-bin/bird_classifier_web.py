"""
# URL : http://localhost:8080/cgi-bin/bird_classifier_web.py
"""
# 모듈 로딩 ---------------------------------------------------
import cgi, sys, codecs, os
import joblib
import cv2
import numpy as np
import base64
import tempfile
from skimage.feature import hog

# WEB 인코딩 설정 ---------------------------------------------
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

# 함수 선언 --------------------------------------------------
# 특징 추출 함수
def extract_features_from_image(img):
    try:
        # 바운딩 박스 설정 (이미지 중앙 부분 사용)
        h, w = img.shape[:2]
        center_margin = 0.1
        x = int(w * center_margin)
        y = int(h * center_margin)
        width = int(w * (1 - 2 * center_margin))
        height = int(h * (1 - 2 * center_margin))

        bbox = (x, y, width, height)

        # HSV 특징 추출
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_bins, s_bins, v_bins = 4, 4, 4  # 총 48차원

        h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])

        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)

        hsv_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()

        # HOG 특징 추출
        x, y, w, h = bbox
        cropped_img = img[y:y+h, x:x+w]
        resized_img = cv2.resize(cropped_img, (32, 32))
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        fd, _ = hog(
            gray,
            orientations=4,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=True,
            block_norm='L2-Hys'
        )

        combined_features = np.concatenate([hsv_features, fd])

        # 차원이 256보다 작으면 0 padding, 크면 자르기
        desired_length = 256
        if combined_features.shape[0] < desired_length:
            padding = np.zeros(desired_length - combined_features.shape[0])
            combined_features = np.concatenate([combined_features, padding])
        else:
            combined_features = combined_features[:desired_length]

        return combined_features

    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        return None
# WEB 페이지 출력 --------------------------------------------
def displayWEB(detect_msg="", img_data=None):
    print("Content-Type: text/html; charset=utf-8")
    print("")
    
    # 이미지 표시 부분
    img_display = ""
    if img_data:
        img_display = f"""
        <div style='margin-top: 20px;'>
            <h3>업로드한 이미지</h3>
            <img src="data:image/jpeg;base64,{img_data}" style="max-width: 300px; max-height: 300px;">
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>새 종류 분류기</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                text-align: center;
            }}
            h2 {{
                color: #333;
            }}
            .container {{
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 80%;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .result {{
                margin-top: 20px;
                padding: 10px;
                border-radius: 5px;
                background-color: #f0f0f0;
                font-weight: bold;
                color: #2c3e50;
            }}
            input[type="file"] {{
                margin: 20px 0;
            }}
            input[type="submit"] {{
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }}
            input[type="submit"]:hover {{
                background-color: #2980b9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>[ 새 종류 분류기 ]</h2>
            <p>사진을 업로드하면 어떤 과에 속해있는 새인지 분류합니다.</p>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="bird_image" accept="image/*">
                <br>
                <input type="submit" value="분류하기">
            </form>
            {img_display}
            <div class="result">{detect_msg}</div>
        </div>
    </body>
    </html>
    """
    print(html)

# 이미지 분류 함수 ---------------------------------------------
def classify_bird_image(img_data):
    try:
        # 이미지 데이터를 임시 파일에 저장
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(img_data)
        
        # 이미지 읽기
        img = cv2.imread(temp_filename)
        if img is None:
            return "이미지를 읽을 수 없습니다."
        
        # 임시 파일 삭제
        os.unlink(temp_filename)
        
        # 특징 추출
        features = extract_features_from_image(img)
        if features is None:
            return "이미지에서 특징을 추출할 수 없습니다."
        
        # 예측
        features = features.reshape(1, -1)  # 2D 배열로 변환
        prediction = clf.predict(features)[0]
        
        # 확률 구하기 (가능하다면)
        proba_msg = ""
        try:
            proba = clf.predict_proba(features)[0]
            class_names = clf.classes_
            # 상위 3개 확률 가져오기
            class_probas = [(class_names[i], proba[i] * 100) for i in range(len(class_names))]
            class_probas.sort(key=lambda x: x[1], reverse=True)
            
            top_three = class_probas[:3]
            proba_msg = "<br><br>다른 가능성:<br>"
            for bird_class, prob in top_three:
                if bird_class != prediction:
                    proba_msg += f"- {bird_class}: {prob:.1f}%<br>"
        except:
            # 확률을 제공하지 않는 모델인 경우
            pass
        
        return f"이 새는 <span style='color: #e74c3c; font-size: 1.2em;'>{prediction}</span> 종류입니다." + proba_msg
    
    except Exception as e:
        return f"분류 중 오류가 발생했습니다: {str(e)}"

# 기능 구현 -----------------------------------------------------
# (1) 학습 데이터 읽기
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = r'C:\Users\KDP-50\OneDrive\바탕 화면\KDT7\09_computer_vision\team_project\cgi-bin\Logistic.pkl'
clf = joblib.load(model_path)

# (2) WEB 페이지 폼 데이터 가져오기
form = cgi.FieldStorage()

# 이미지와 분류 결과 초기화
img_data_base64 = None
result_msg = ""

# (3) 이미지 업로드 처리 및 분류
if "bird_image" in form:
    fileitem = form["bird_image"]
    if fileitem.filename:
        # 이미지 데이터 읽기
        img_data = fileitem.file.read()
        
        # 이미지를 base64로 인코딩 (HTML에서 표시하기 위한 용도)
        img_data_base64 = base64.b64encode(img_data).decode()
        
        # 새 분류
        result_msg = classify_bird_image(img_data)
    else:
        result_msg = "이미지를 선택해주세요."
else:
    result_msg = "새 이미지를 업로드하면 과를 분류합니다."

# (4) WEB 출력하기
displayWEB(result_msg, img_data_base64)
