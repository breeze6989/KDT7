from datetime import datetime; from ..extensions import db
class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100)); rtsp_url = db.Column(db.String(500))
    status = db.Column(db.String(20), default="normal")
    location_x = db.Column(db.Float); location_y = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)