from ..extensions import db

class Layout(db.Model):
    """그리드 배열 (3×3 / 4×3) 설정"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), unique=True)
    rows = db.Column(db.Integer)
    cols = db.Column