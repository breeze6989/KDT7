from datetime import datetime
from ..extensions import db

class Camera(db.Model):
    """CCTV 카메라 메타"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    stream_url = db.Column(db.String(500), nullable=False)  # RTSP/HLS URL
    status = db.Column(db.String(20), default="normal")    # normal/offline/disabled
    latitude = db.Column(db.Float)                          # 지도 좌표 (선택)
    longitude = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)