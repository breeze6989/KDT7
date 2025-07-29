"""환경변수를 불러와 Flask Config 로 매핑"""
import os
from dotenv import load_dotenv

load_dotenv()  # .env → os.environ

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///fire.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MEDIA_ROOT = os.getenv("MEDIA_ROOT", "media")  # 클립·스냅샷 저장 경로