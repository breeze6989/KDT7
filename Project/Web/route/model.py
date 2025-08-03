"""AI 모델 파일 업로드(PT) & 단일 이미지 추론 테스트
- 모델 업로드 시 Detector.load(pt) 호출 → 실시간 추론 교체
- 테스트 엔드포인트는 이미지 URL 입력 → 화재 확률 반환(디버그용)
{사전인지 이벤트에 대한 위도, 경도를
당일 해당 현재 시간 이후 있으면 전달, 아닌 경우 전달하지 않음}
1. 실시간 영상을 화재 탐지모델에 전달
  - return bbox
2-1. if !bbox: continue
2-2. else:
위치특정모델에 bbox 및 사전 이벤트 위도 경도 전달
3-1. if ((lat,long) != (이벤트 위도 경도)) & (lat,long):
    return 위도, 경도
    -> 웹에 해당 내용 전달
3-2 elif ((lat,long) == (이벤트 위도 경도)) & (lat,long):
    return 위도, 경도, 경보해제 가능 이벤트 상수
    -> 웹에 해당 내용 전달
3-3 else:
    -> 웹에 위치 특정 못함 전달

"""
from flask import Blueprint, request, current_app
import os, uuid, requests, numpy as np, cv2
from Web.model.fire_detect import *
bp = Blueprint("ai", __name__)
# .pt 업로드 

@bp.post("/model")

def upload_model():

    """PyTorch .pt 파일 업로드 → Detector에 즉시 로드"""
    if "file" not in request.files:
        return {"msg": "file missing"}, 400
    
    f = request.files["file"]
    fname = secure_filename(f.filename) or f"model_{uuid.uuid4().hex}.pt"
    models_dir = os.path.join(current_app.config["MEDIA_ROOT"], "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, fname)
    f.save(path)

    # Detector에 모델 로드
    detector.load(path)  # run.py 에서 만든 전역 instance
    return {"msg": "model uploaded", "path": path}

@bp.post("/test")
def test_infer():
    """단일 이미지 URL → 화재 확률 반환 (디버깅용)"""
    url = (request.get_json() or {}).get("image_url")
    if not url:
        return {"msg": "image_url required"}, 400
    try:
        resp = requests.get(url, timeout=5)
        frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        score = detector.infer(frame)
        return {"probability": round(score, 4)}
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception("infer error")
        return {"msg": "infer failed", "err": str(e)}, 500