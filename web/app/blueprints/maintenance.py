from flask import Blueprint
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.camera import Camera
bp = Blueprint("maintenance", __name__)

@bp.put("/disable/<int:cid>")@jwt_required()
def dis(cid): c=Camera.query.get_or_404(cid); c.status="disabled"; db.session.commit(); return{"msg":"disabled"}

@bp.put("/enable/<int:cid>")@jwt_required()
def ena(cid): c=Camera.query.get_or_404(cid); c.status="normal"; db.session.commit(); return{"msg":"enabled"}