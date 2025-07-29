from flask import Blueprint
from flask_jwt_extended import jwt_required
from sqlalchemy import func
from ..extensions import db
from ..models.alert import Alert
from ..models.camera import Camera

bp = Blueprint("dashboard", __name__)

@bp.get("/summary")
@jwt_required()
def summary():
    """대시보드 KPI 요약 제공"""
    total = db.session.query(func.count(Alert.id)).scalar() or 0
    confirmed = db.session.query(func.count(Alert.id)).filter_by(level="confirmed").scalar() or 0
    false = db.session.query(func.count(Alert.id)).filter_by(level="false").scalar() or 0
    active = db.session.query(func.count(Camera.id)).filter(Camera.status != "offline").scalar()
    total_cam = db.session.query(func.count(Camera.id)).scalar()
    false_rate = round(false / total * 100, 2) if total else 0
    return {
        "total_alerts": total,
        "confirmed": confirmed,
        "false_rate": false_rate,
        "active_cameras": f"{active}/{total_cam}",
    }
