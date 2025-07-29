from datetime import datetime; from ..extensions import db
class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer)
    event_type = db.Column(db.String(20))  # fire/test
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.String(255))