from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.event import Event
bp = Blueprint("events", __name__)

def ser(e):
    return{"id":e.id,"cam":e.camera_id,"type":e.event_type,"time":e.timestamp.isoformat(),"details":e.details}

@bp.get("/")@jwt_required()
def list_events():
    cam=request.args.get("camera_id"); q=db.session.query(Event).order_by(Event.timestamp.desc())
    if cam: q=q.filter_by(camera_id=cam)
    return{"events":[ser(e) for e in q.limit(int(request.args.get("limit",50)))]}