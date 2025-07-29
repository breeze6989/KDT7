from datetime import datetime
from ..extensions import db

class Log(db.Model):
    """추가 설명/운영자 메모 저장 (1:N Alert)"""
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey("alert.id"))
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    alert = db.relationship("Alert", backref=db.backref("logs", lazy=True))