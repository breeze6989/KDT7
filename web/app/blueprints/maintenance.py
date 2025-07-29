from flask import Blueprint
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.camera import Camera

bp = Blueprint("maintenance", __name__)

@bp.put("/disable/<int:cid>")
@jwt_required()
def disable_cam(cid):
    cam = Camera.query.get_or_404(cid); cam.status = "disabled"; db.session.commit()
    return {"msg": "disabled"}

@bp.put("/enable/<int:cid>")
@jwt_required()
def enable_cam(cid):
    cam = Camera.query.get_or_404(cid); cam.status = "normal"; db.session.commit()
    return {"msg": "enabled"}