"""
app.py – Flask + YOLOv8 화재감지 대시보드
"""

import os, cv2, time, datetime, sqlite3, threading
from pathlib import Path
import numpy as np, torch
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from ultralytics.nn.autobackend import AutoBackend

# ───── 환경 상수 ─────
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
BASE_DIR   = Path(__file__).parent
VIDEO_DIR  = BASE_DIR/'static'/'videos'
MODEL_PATH = BASE_DIR/'models'/'fire_detector.pt'
DB_PATH    = BASE_DIR/'fire_dash.db'
CONF_THRES = 0.20        # ↓ 필요하면 0.5 로 올리세요
COOLTIME   = 5           # 초

# ───── Flask ─────
app = Flask(__name__)
app.secret_key = 'CHANGE_THIS_KEY'

# ───── DB 헬퍼 ─────
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db(); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS cctvs(
        id INTEGER PRIMARY KEY,name TEXT,src TEXT,status TEXT,lat REAL,lng REAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS logs(
        id INTEGER PRIMARY KEY,cctv_id INTEGER,ts TEXT,msg TEXT,lat REAL,lng REAL,
        FOREIGN KEY(cctv_id) REFERENCES cctvs(id))""")
    if cur.execute("SELECT COUNT(*) FROM cctvs").fetchone()[0] == 0:
        for i, vid in enumerate(sorted(VIDEO_DIR.glob('*.mp4')), 1):
            lat, lng = 37.55 + i*0.0002, 126.97 + i*0.0002
            cur.execute("INSERT INTO cctvs VALUES(?,?,?,?,?,?)",
                        (i, f"CCTV-{i}", str(vid), '정상', lat, lng))
    conn.commit(); conn.close()

# ───── YOLO 모델 ─────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = AutoBackend(str(MODEL_PATH), device=device, fuse=False)

def detect_fire(frame):
    img = cv2.resize(frame, (640, 640))[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)/255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img)[0]
    return bool(((pred[:,5]==0) & (pred[:,4]>CONF_THRES)).any().item()) if pred.numel() else False

# ───── 모니터 스레드 ─────
last_log, muted = {}, set()
def save_log(cid):
    conn=db(); cur=conn.cursor()
    lat,lng = cur.execute("SELECT lat,lng FROM cctvs WHERE id=?", (cid,)).fetchone()
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute("INSERT INTO logs(cctv_id,ts,msg,lat,lng) VALUES(?,?,?,?,?)",
                (cid,ts,f"CCTV-{cid} 화재 감지",lat,lng))
    cur.execute("UPDATE cctvs SET status='화재감지' WHERE id=?", (cid,))
    conn.commit(); conn.close()

def monitor(cid, path):
    uri = 'file:///' + str(path).replace('\\','/')
    cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    last_log[cid] = 0
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        if f % 3 == 0 and detect_fire(frame):
            now=time.time()
            if cid not in muted and now - last_log[cid] > COOLTIME:
                save_log(cid); last_log[cid] = now
        f += 1

# ───── 라우트 ─────
@app.route('/')
def home(): return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('dashboard'))

@app.route('/show_logs')
def show_logs():
    page = max(int(request.args.get('page', 1)), 1); size = 10; off = (page-1)*size
    conn=db()
    rows = conn.execute("""SELECT l.*,c.src FROM logs l JOIN cctvs c ON l.cctv_id=c.id
                           ORDER BY l.id DESC LIMIT ? OFFSET ?""",(size,off)).fetchall()
    tot  = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]; conn.close()
    return render_template('logs.html', logs=rows, page=page, page_cnt=(tot+size-1)//size)

@app.route('/api/dashboard_data')
def api_dash():
    conn=db()
    cams=[dict(r) for r in conn.execute("SELECT * FROM cctvs").fetchall()]
    logs=[dict(r) for r in conn.execute(
        "SELECT id,ts,msg,cctv_id,lat,lng FROM logs ORDER BY id DESC LIMIT 5").fetchall()]
    conn.close()
    for c in cams:
        c['stream_url']=url_for('static',filename='videos/'+Path(c['src']).name)
    for l in logs:
        cam=next(x for x in cams if x['id']==l['cctv_id'])
        l['stream_url']=cam['stream_url']
    return jsonify({'cctv_list':cams,'log_list':logs})

@app.route('/api/ack_alert', methods=['POST'])
def ack_alert():
    cid=int(request.json['cctv_id']); muted.add(cid)
    db().execute("UPDATE cctvs SET status='정상' WHERE id=?", (cid,)).commit()
    return jsonify(ok=True)

# ───── 실행 ─────
if __name__ == '__main__':
    init_db()
    for cid, src in db().execute("SELECT id,src FROM cctvs"):
        threading.Thread(target=monitor, args=(cid, Path(src)), daemon=True).start()
    app.run(debug=True, threaded=True)
