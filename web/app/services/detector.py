import threading, time, cv2, torch, numpy as np, uuid
from flask import current_app
from sqlalchemy.orm import scoped_session, sessionmaker
from ..extensions import db, socketio
from ..models.camera import Camera
from ..models.event import Event
from ..utils.clip import save_clip

class Detector(threading.Thread):
    """RTSP → GPU Torch 추론 → Event 저장 + SocketIO"""
    def __init__(self, app, refresh=5):
        super().__init__(daemon=True)
        self.app = app
        self.refresh = refresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.Session = sessionmaker(bind=db.engine)

    def load(self, pt):
        """/ai/upload 에서 호출 – 모델 로드"""
        self.model = torch.jit.load(pt, map_location=self.device).eval().to(self.device)
        current_app.logger.info(f"[Detector] 모델 로드: {pt}")

    def infer(self, frame):
        if not self.model:
            return 0.0
        img = cv2.resize(frame,(224,224))[:, :, ::-1]/255.0
        t = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(t))[0].item()
        return p

    def run(self):
        with self.app.app_context():
            while True:
                sess = scoped_session(self.Session)
                cams = sess.query(Camera).filter(Camera.status != "disabled").all()
                for cam in cams:
                    self._process(cam, sess)
                sess.close(); time.sleep(self.refresh)

    def _process(self, cam, sess):
        cap = cv2.VideoCapture(cam.rtsp_url);
        if not cap.isOpened():
            return
        idx=0
        while idx<150:
            ret, frame = cap.read(); idx+=1
            if not ret: break
            if idx%5==0 and self.model:
                score = self.infer(frame)
                if score>=0.8:
                    clip=save_clip(cam.rtsp_url)
                    evt=Event(camera_id=cam.id,event_type="fire",details=f"confidence={score:.2f};clip={clip}")
                    sess.add(evt); sess.commit();
                    socketio.emit("new_event", {"id":evt.id,"cam":cam.id,"clip":clip,"score":score}, namespace="/events")
                    break
        cap.release()