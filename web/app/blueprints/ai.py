"""AI 모델 파일 업로드(PT) & 단일 이미지 추론 테스트
- 모델 업로드 시 Detector.load(pt) 호출 → 실시간 추론 교체
- 테스트 엔드포인트는 이미지 URL 입력 → 화재 확률 반환(디버그용)
"""
from flask import Blueprint, request, current_app
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
import os, uuid, requests, numpy as np, cv2
from ..services.detector import Detector  # 타입힌트용 (실제 인스턴스는 run.py)

# run.py 에서 detector 변수를 앱 전역으로 노출했다는 가정
from app.services.detector import detector  # pylint: disable=import-error

bp = Blueprint("ai", __name__)

@bp.post("/model")
@jwt_required()
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
@jwt_required()
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