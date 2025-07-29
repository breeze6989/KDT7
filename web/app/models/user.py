# user.py ------------------------------------------------------
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from ..extensions import db

class User(db.Model):
    """사용자 (로그인용)"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    _pw = db.Column("password_hash", db.String(255), nullable=False)
    role = db.Column(db.String(20), default="viewer")  # admin/operator/viewer
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw: str):
        self._pw = generate_password_hash(pw)

    def check_password(self, pw: str) -> bool:
        return check_password_hash(self._pw, pw)