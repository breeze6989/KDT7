from flask import Blueprint, request
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from ..extensions import db
from ..models.user import User

# Blueprint 인스턴스
auth_bp = Blueprint("auth", __name__)

# ---------- 라우트 ----------

@auth_bp.post("/register")
def register():
    """회원가입 (개발용)"""
    data = request.get_json(force=True)
    # 이미 존재하는 유저 체크
    if User.query.filter_by(username=data.get("username")).first():
        return {"msg": "already exists"}, 409

    # User 생성
    u = User(username=data["username"], role=data.get("role", "viewer"))
    u.set_password(data["password"])
    db.session.add(u)
    db.session.commit()
    return {"msg": "created"}, 201


@auth_bp.post("/login")
def login():
    """로그인 → JWT 토큰 반환"""
    data = request.get_json(force=True)
    u = User.query.filter_by(username=data.get("username")).first()
    if not u or not u.check_password(data.get("password", "")):
        return {"msg": "bad creds"}, 401

    token = create_access_token(identity={"id": u.id, "role": u.role})
    return {"access_token": token}


@auth_bp.get("/me")
@jwt_required()
def me():
    """토큰 유저 정보 조회"""
    return {"user": get_jwt_identity()}

