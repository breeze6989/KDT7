from flask import Flask
from flask_jwt_extended import JWTManager
from .config import Config

jwt = JWTManager()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 정적 경로: media → /media
    app.static_folder = app.config["MEDIA_ROOT"]
    app.static_url_path = "/media"

    jwt.init_app(app)

    # 블루프린트 등록
    from .blueprints.main import bp as main_bp
    from .blueprints.cameras import bp as cam_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(cam_bp, url_prefix="/cameras")

    return app