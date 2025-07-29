"""Blueprint 패키지 초기화 – 필요 시 공용 데코레이터·에러핸들러 작성"""
from flask import Blueprint

# 공용 오류 처리 예시 (선택)
api_bp = Blueprint("api", __name__)

@api_bp.app_errorhandler(404)
def not_found(err):
    return {"msg": "not found"}, 40