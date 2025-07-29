from datetime import datetime; from ..extensions import db
class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer)
    status = db.Column(db.String(20), default="open")
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)