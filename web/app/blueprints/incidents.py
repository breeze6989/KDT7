from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.incident import Incident

bp = Blueprint("incidents", __name__)

def ser(i: Incident):
    return {"id": i.id, "cam": i.camera_id, "status": i.status,
            "start": i.started_at.isoformat(), "end": i.ended_at.isoformat() if i.ended_at else None}

@bp.get("/")
@jwt_required()
def all_inc():
    return {"incidents": [ser(i) for i in Incident.query.all()]}

@bp.post("/")
@jwt_required()
def add_inc():
    inc=Incident(**request.get_json()); db.session.add(inc); db.session.commit(); return ser(inc),201