from flask import Blueprint, request
from flask_jwt_extended import jwt_required
from ..extensions import db
from ..models.camera import Camera
bp = Blueprint("cameras", __name__)

def ser(c):
    return{"id":c.id,"name":c.name,"url":c.rtsp_url,"status":c.status,"loc":{"x":c.location_x,"y":c.location_y}}

@bp.get("/")@jwt_required()
def all():
    return{"cameras":[ser(c) for c in Camera.query.all()]}

@bp.post("/")@jwt_required()
def add():
    c=Camera(**(request.get_json() or {})); db.session.add(c); db.session.commit(); return ser(c),201

@bp.get("/geo")@jwt_required()
def geo():
    return{"features":[{"id":c.id,"x":c.location_x,"y":c.location_y} for c in Camera.query.all()]}