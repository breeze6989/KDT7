from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db, socketio
from ..models.alert import Alert
bp = Blueprint("alerts", __name__)

def d(a): return{"id":a.id,"cam":a.camera_id,"level":a.level,"created":a.created_at.isoformat()}

@bp.get("/")@jwt_required()
def list_(): return{"alerts":[d(a) for a in Alert.query.order_by(Alert.created_at.desc()).limit(100)]}

@bp.put("/<int:aid>")@jwt_required()
def upd(aid):
    a=Alert.query.get_or_404(aid); a.level=request.json.get("level",a.level); db.session.commit(); socketio.emit("alert_update",d(a),namespace="/alerts"); return d(a)