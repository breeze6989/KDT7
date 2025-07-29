from datetime import datetime
from ..extensions import db

class Incident(db.Model):
    """사고(화재) 단위 – 복수 알림 묶음"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey("camera.id"))
    status = db.Column(db.String(20), default="open")  # open/resolved
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)

    camera = db.relationship("Camera", backref=db.backref("incidents", lazy=True))