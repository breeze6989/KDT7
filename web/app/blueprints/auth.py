from flask import Blueprint, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from ..extensions import db
from ..models.user import User
bp = Blueprint("auth", __name__)

@bp.post("/register")
def register():
    d=request.get_json() or {}; u=User(username=d["username"]); u.set_password(d["password"])
    db.session.add(u); db.session.commit(); return{"msg":"created"},201

@bp.post("/login")
def login():
    d=request.get_json() or {}; u=User.query.filter_by(username=d["username"]).first()
    if not u or not u.verify_password(d["password"]):
        return{"msg":"bad"},401
    return{"access_token": create_access_token(identity={"id":u.id,"role":u.role})}

@bp.get("/me")
@jwt_required()
def me():
    return{"user": get_jwt_identity()}
