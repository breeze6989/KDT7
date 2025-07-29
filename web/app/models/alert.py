"""Alert 모델
- Detector 또는 외부 AI 서버가 생성하는 이벤트 레코드
- level: suspicious / confirmed / false (오경보)
- model_score: 0~1 확률 값
"""
from datetime import datetime
from ..extensions import db

class Alert(db.Model):
    """모델/운영자에 의해 생성된 알림 레코드"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey("camera.id"))
    level = db.Column(db.String(20), default="suspicious")     # suspicious/confirmed/false
    message = db.Column(db.String(255))                         # 예: '연기 감지'
    model_score = db.Column(db.Float)                           # 0~1 확률값
    clip_path = db.Column(db.String(255))                       # ffmpeg 저장 경로
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    camera = db.relationship("Camera", backref=db.backref("alerts", lazy=True))