from datetime import datetime; from ..extensions import db
class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer); text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)