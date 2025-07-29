from ..extensions import db
class Layout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20)); rows = db.Column(db.Integer); cols = db.Column(db.Integer)
    is_active = db.Column(db.Boolean, default=False)