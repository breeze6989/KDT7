"""애플리케이션 부트스트랩
1) app = create_app()
2) Detector(app) 스레드 시작 (실시간 추론)
3) Flask‑SocketIO 개발 서버 실행
"""
from app import create_app            # 앱 팩토리
from app.extensions import socketio   # SocketIO 서버
from app.services.detector import Detector  # 실시간 추론 스레드

app = create_app()

# Detector: DB에 등록된 카메라 스트림을 주기적으로 모니터링
_detector = Detector(app)
_detector.start()   # 데몬 스레드 (백그라운드)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)