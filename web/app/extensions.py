from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO

# 전역 인스턴스 (create_app 에서 init_app)

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
socketio = SocketIO()