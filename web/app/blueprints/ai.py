from flask import Blueprint, request, current_app
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
import os, uuid, requests, numpy as np, cv2
from ..services.detector import Detector  # Detector 인스턴스 재사용

# Detector 인스턴스는 run.py 에서 생성됨 (import 사이클 주의)
from ..services import detector as detector_mod  # type: ignore

bp = Blueprint("ai", __name__)

@bp.post("/model")
@jwt_required()
def upload_model():
    """PyTorch .pt 업로드 → Detector.reload"""
    f = request.files.get("file")
    if not f:
        return {"msg": "file missing"}, 400
    fname = secure_filename(f.filename) or f"model_{uuid.uuid4().hex}.pt"
    models_dir = os.path.join(current_app.config["MEDIA_ROOT"], "models"); os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, fname)
    f.save(path)
    detector_mod._detector.load(path)  # pylint:disable=protected-access
    return {"msg": "model uploaded", "path": path}

@bp.post("/test")
@jwt_required()
def test_infer():
    """단일 이미지 URL 테스트"""
    url = (request.get_json() or {}).get("image_url")
    if not url:
        return {"msg": "image_url required"}, 400
    resp = requests.get(url, timeout=5); frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    score = detector_mod._detector.infer(frame)  # pylint:disable=protected-access
    return {"score": score}