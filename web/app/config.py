import os
from dotenv import load_dotenv
load_dotenv()

import os; from dotenv import load_dotenv; load_dotenv()
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt")
    SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///fire.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", "media")  # 클립·모델 저장 경로
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", "media")