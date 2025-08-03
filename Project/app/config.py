import os
class Config:
    SECRET_KEY = "dev-secret"
    MEDIA_ROOT = os.path.join(os.getcwd(), "media")   # 썸네일 캐시 저장
    # JWT 쿠키(개발용 간단 설정)
    JWT_SECRET_KEY = "dev-jwt"
    JWT_TOKEN_LOCATION = ["cookies"]
    JWT_ACCESS_COOKIE_NAME = "access_token"
    JWT_COOKIE_SECURE = False
    JWT_COOKIE_CSRF_PROTECT = False