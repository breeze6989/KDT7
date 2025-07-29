from datetime import datetime
from passlib.hash import bcrypt
from ..extensions import db
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password_hash = db.Column(db.String(255))
    role = db.Column(db.String(20), default="viewer")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def set_password(self, pw): self.password_hash = bcrypt.hash(pw)
    def verify_password(self, pw): return bcrypt.verify(pw, self.password_hash)