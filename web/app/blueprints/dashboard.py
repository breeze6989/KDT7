from flask import Blueprint
from flask_jwt_extended import jwt_required
from sqlalchemy import func
from ..extensions import db
from ..models.camera import Camera
from ..models.event import Event
bp = Blueprint("dashboard", __name__)

@bp.get("/summary")@jwt_required()
def summary():
    total_cam=db.session.query(func.count(Camera.id)).scalar(); total_evt=db.session.query(func.count(Event.id)).scalar();
    fires=db.session.query(func.count(Event.id)).filter(Event.event_type=="fire").scalar();
    return{"total_cameras":total_cam,"total_events":total_evt,"fire_events":fires}
