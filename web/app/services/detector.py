"""Detector 서비스
- 1) 모델(.pt) 로드 (Blueprint 업로드 API 통해 최신 경로 기록)
- 2) DB 에 등록된 모든 카메라의 RTSP/HLS 스트림 열기
- 3) 각 카메라별로 5프레임 단위로 추론 → 화재/연기 여부 판단
- 4) 결과를 Alert 테이블에 저장 + SocketIO 이벤트 전송
"""
import threading, time, os, cv2, torch
import numpy as np
from flask import current_app
from sqlalchemy.orm import scoped_session, sessionmaker
from app.extensions import db, socketio
from app.models.camera import Camera
from app.models.alert import Alert


class Detector(threading.Thread):
    """백그라운드 추론 스레드"""

    def __init__(self, app, interval=1.0):  # interval: 카메라 목록 리프레시 주기
        super().__init__(daemon=True)
        self.app = app
        self.interval = interval
        self.model = None  # torch 모델
        self.model_path = None
        # SQLAlchemy 세션은 스레드 로컬로 별도 생성
        self.Session = scoped_session(sessionmaker(bind=db.engine))

    # ---------- 모델 로드 ----------
    def load_model(self, pt_path: str):
        """PyTorch .pt 모델 로드 (GPU 우선, 없으면 CPU)"""
        # 1) 디바이스 선택
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 2) 모델 로드 & 디바이스 이동
        self.model = torch.jit.load(pt_path, map_location=self.device)
        self.model.eval().to(self.device)
        self.model_path = pt_path
        dname = "GPU" if self.device.type == "cuda" else "CPU"
        current_app.logger.info(
            f"[Detector] model loaded on {dname}: {pt_path}"
        )

    # ---------- 메인 루프 ----------
    def run(self):
        with self.app.app_context():
            while True:
                cams = self.Session().query(Camera).all()
                for cam in cams:
                    self.process_camera(cam)
                time.sleep(self.interval)

    # ---------- 카메라 단일 처리 ----------
    def process_camera(self, cam: Camera):
        """카메라 스트림 열어 5프레임 간격 추론"""
        cap = cv2.VideoCapture(cam.stream_url)
        if not cap.isOpened():
            current_app.logger.warning(f"[Detector] cannot open stream: {cam.name}")
            return

        frame_idx = 0
        while frame_idx < 150:  # 150프레임(≈5초)만 시연용, 실제 서비스는 while True
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:  # 5프레임당 추론
                fire_prob = self.infer(frame)
                if fire_prob >= 0.8:  # 스코어 threshold 예시
                    self.raise_alert(cam, fire_prob)
            frame_idx += 1
        cap.release()

    # ---------- 추론 ----------
        # ---------- 추론 ----------
    def infer(self, frame: np.ndarray) -> float:
        """OpenCV BGR Frame → Tensor 변환 → 모델 forward → 화재 score 반환"""
        if self.model is None:
            return 0.0
        # 전처리 (예: 224x224, BGR→RGB, 0-1 스케일)
        img = cv2.resize(frame, (224, 224))[:, :, ::-1] / 255.0
        tensor = (
            torch.from_numpy(img.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            out = self.model(tensor)
        # out: [batch, 1] → sigmoid 확률 가정
        prob = torch.sigmoid(out)[0].item()
        return prob

    # ---------- Alert 생성 ----------
    def raise_alert(self, cam: Camera, score: float):
        """DB 저장 & SocketIO 브로드캐스트"""
        session = self.Session()
        alert = Alert(camera_id=cam.id, level="suspicious", message="연기/화재 의심", model_score=score)
        session.add(alert)
        session.commit()

        data = {"camera_id": cam.id, "score": score, "alert_id": alert.id}
        socketio.emit("new_alert", data, namespace="/alerts")
        current_app.logger.info(f"[Detector] Alert raised cam:{cam.id} score:{score}")