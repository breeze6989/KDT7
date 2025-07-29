from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db, socketio
from ..models.alert import Alert

bp = Blueprint("alerts", __name__)

def to_dict(a: Alert):
    return {"id": a.id, "cam": a.camera_id, "level": a.level, "score": a.model_score,
            "clip": a.clip_path, "created": a.created_at.isoformat()}

@bp.get("/")
@jwt_required()
def list_alerts():
    return {"alerts": [to_dict(a) for a in Alert.query.order_by(Alert.created_at.desc()).limit(100)]}

@bp.put("/<int:aid>")
@jwt_required()
def update_alert(aid):
    """운영자 수동 상태 변경 (confirmed/false)"""
    a=Alert.query.get_or_404(aid); a.level=request.json.get("level", a.level)
    db.session.commit(); socketio.emit("alert_update", to_dict(a), namespace="/alerts")
    return to_dict(a)