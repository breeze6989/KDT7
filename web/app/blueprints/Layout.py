from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.Layout import Layout

bp = Blueprint("layout", __name__)

@bp.get("/")
@jwt_required()
def list_layout():
    return {"layouts": [{"id": l.id, "name": l.name, "active": l.is_active} for l in Layout.query.all()]}

@bp.put("/activate/<int:lid>")
@jwt_required()
def activate(lid):
    Layout.query.update({Layout.is_active: False}); l=Layout.query.get_or_404(lid); l.is_active=True; db.session.commit(); return {"msg": "activated"}