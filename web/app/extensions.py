from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
socketio = SocketIO()
from flask_sqlalchemy import SQLAlchemy; db = SQLAlchemy()
from flask_migrate import Migrate; migrate = Migrate()
from flask_jwt_extended import JWTManager; jwt = JWTManager()
from flask_socketio import SocketIO; socketio = SocketIO()
