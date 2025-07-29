"""create_app 패턴
- 모든 확장(init_app) → 블루프린트 등록 → /ping 헬스체크
"""
from flask import Flask
from .config import Config
from .extensions import db, migrate, jwt, socketio

# 지연(import) ⇒ 순환참조 방지 + 코드 가독성
from .blueprints.auth import bp as auth_bp
from .blueprints.cameras import bp as cam_bp
from .blueprints.alerts import bp as alert_bp
from .blueprints.incidents import bp as inc_bp
from .blueprints.Layout import bp as layout_bp
from .blueprints.dashboard import bp as dash_bp
from .blueprints.maintenance import bp as mnt_bp
from .blueprints.ai import bp as ai_bp


def create_app() -> Flask:
    """Flask 객체 생성 & 설정"""
    app = Flask(__name__)
    app.config.from_object(Config())

    # ─── 확장 초기화 ───
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # ─── 블루프린트 등록 ───
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(cam_bp, url_prefix="/cameras")
    app.register_blueprint(alert_bp, url_prefix="/alerts")
    app.register_blueprint(inc_bp, url_prefix="/incidents")
    app.register_blueprint(layout_bp, url_prefix="/layout")
    app.register_blueprint(dash_bp, url_prefix="/dashboard")
    app.register_blueprint(mnt_bp, url_prefix="/maintenance")
    app.register_blueprint(ai_bp, url_prefix="/ai")

    @app.get("/ping")
    def ping():
        """헬스체크 (CI/CD probe 용)"""
        return {"msg": "pong"}

    return app

