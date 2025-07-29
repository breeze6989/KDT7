from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.camera import Camera

bp = Blueprint("cameras", __name__)

def ser(cam: Camera):
    return {"id": cam.id, "name": cam.name, "url": cam.stream_url, "status": cam.status,
            "lat": cam.latitude, "lng": cam.longitude}

@bp.get("/")
@jwt_required()
def list_cams():
    return {"cameras": [ser(c) for c in Camera.query.all()]}

@bp.post("/")
@jwt_required()
def add_cam():
    d=request.get_json(); c=Camera(**d); db.session.add(c); db.session.commit(); return ser(c),201

@bp.get("/geo")
@jwt_required()
def geo_json():
    feats=[{"type":"Feature","geometry":{"type":"Point","coordinates":[c.longitude,c.latitude]},"properties":ser(c)} for c in Camera.query.all() if c.latitude and c.longitude]
    return {"type":"FeatureCollection","features": feats}