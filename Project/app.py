"""
app.py – 화재 감지 대시보드 (3프레임마다 예측, 5초 쿨타임, 5초 간격 videos↔DB 동기화)
"""
import os, cv2, time, datetime, sqlite3, threading, json # json 모듈 추가
from pathlib import Path
import numpy as np, torch
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from ultralytics import YOLO
from collections import deque

# ───── 환경 상수 ─────
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
BASE_DIR   = Path(__file__).parent
VIDEO_DIR  = BASE_DIR / 'static' / 'videos'
MODEL_PATH = BASE_DIR / 'models' / 'fire_detector.pt'
DB_PATH    = BASE_DIR / 'fire_dash.db'
FIRE_IDX  = 0
SMOKE_IDX = 2
CONF_THR  = 0.3
COOLTIME   = 600
SYNC_INT   = 5
# ───── 튜닝 상수 ─────
SKIP_FRAMES  = 2     # 2프레임마다 예측 (기존 3)
VOTE_WINDOW  = 12    # 최근 12회 결과 저장
FIRE_VOTE    = 6     # 6회 이상이면 화재
SMOKE_VOTE   = 6     # 6회 이상이면 주의


# ───── Flask ─────
app = Flask(__name__)
app.secret_key = 'CHANGE_THIS_KEY'

# ───── DB 헬퍼 ─────
db_lock = threading.Lock()

def db():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        isolation_level=None
    )
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

def init_db():
    conn = db(); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS cctvs(
        id INTEGER PRIMARY KEY,name TEXT,src TEXT,status TEXT,lat REAL,lng REAL)""")
    # 좌표 저장 컬럼 추가: boxes
    cur.execute("""CREATE TABLE IF NOT EXISTS logs(
        id INTEGER PRIMARY KEY,cctv_id INTEGER,ts TEXT,msg TEXT,lat REAL,lng REAL,
        kind TEXT DEFAULT '정상', boxes TEXT,
        FOREIGN KEY(cctv_id) REFERENCES cctvs(id))""")
    conn.commit(); conn.close()

# ───── videos 폴더 ↔ DB 동기화 ─────
def sync_videos_with_db():
    conn=db(); cur=conn.cursor()
    existing={Path(r['src']).name for r in cur.execute("SELECT src FROM cctvs")}
    files   ={f.name for f in VIDEO_DIR.glob('*.mp4')}

    for name in files-existing:
        _id = cur.execute("SELECT COALESCE(MAX(id),0)+1 FROM cctvs").fetchone()[0]
        lat,lng = 37.55 + _id*0.0002, 126.97 + _id*0.0002
        cur.execute("INSERT INTO cctvs VALUES(?,?,?,?,?,?)",
                    (_id, f"CCTV-{_id}", str(VIDEO_DIR/name), '정상', lat, lng))
        print(f'[SYNC] 새 영상 등록 ▶ {name}')

    conn.commit(); conn.close()

def sync_scheduler():
    while True:
        sync_videos_with_db()
        time.sleep(SYNC_INT)

# ───── YOLOv8 모델 ─────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = YOLO(str(MODEL_PATH))
model.fuse()
model_lock = threading.Lock()

# ─── detect_status 함수 수정: 좌표값 반환 ─────────────────
def detect_status(frame):
    """BGR 이미지 → ('정상'|'주의'|'화재감지', [boxes])"""
    try:
        img = cv2.resize(frame, (640, 640))
        with model_lock:
            r = model.predict(img, verbose=False, device=device)[0]

        if r.boxes:
            cls   = np.array(r.boxes.cls.cpu())
            conf  = np.array(r.boxes.conf.cpu())
            boxes = r.boxes.xyxy.cpu().numpy().tolist()   # ← list 로 변환

            fire  = ((cls == FIRE_IDX)  & (conf > CONF_THR)).any()
            smoke = ((cls == SMOKE_IDX) & (conf > CONF_THR)).any()
            if fire:  return '화재감지', boxes
            if smoke: return '주의', boxes
    except Exception as e:
        print('[YOLO ERROR]', e)

    return '정상', []          # (상태, 빈 리스트)


# ───── 모니터 스레드 ─────
last_log, muted = {}, set()

# ── 로그 저장 함수 수정: 박스 좌표값 저장 ──────────────────
def save_log(cid, status, lat, lng, boxes):
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f"CCTV-{cid} {status}"
    with db_lock:
        conn = db(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO logs(cctv_id,ts,msg,lat,lng,kind,boxes)
            VALUES(?,?,?,?,?,?,?)
        """, (cid, ts, msg, lat, lng, status, json.dumps(boxes)))
        cur.execute("UPDATE cctvs SET status=? WHERE id=?", (status, cid))
        conn.commit(); conn.close()


# ── monitor 함수 수정: 좌표값을 save_log에 전달 ──────────
from collections import deque
...
def monitor(cid, path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f'[WARN] CCTV-{cid} 영상 열기 실패'); return

    votes = deque(maxlen=VOTE_WINDOW)   # ← 12 칸 순환버퍼
    cur_status_db = '정상'
    last_logged   = {'상태':'정상', 'time':0}
    lat,lng = db().execute("SELECT lat,lng FROM cctvs WHERE id=?", (cid,)).fetchone()

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        if f % SKIP_FRAMES == 0:        # ← 2프레임마다
            raw, boxes = detect_status(frame)    # ('정상'|… , boxes)
            votes.append(raw)

            fire_cnt  = votes.count('화재감지')
            smoke_cnt = votes.count('주의')
            new_status = ('화재감지' if fire_cnt  >= FIRE_VOTE else
                          '주의'     if smoke_cnt >= SMOKE_VOTE else
                          '정상')

            # DB 상태 즉시 반영
            if new_status != cur_status_db:
                with db_lock:
                    db().execute("UPDATE cctvs SET status=? WHERE id=?",
                                 (new_status, cid))
                cur_status_db = new_status

            # 쿨타임(10분) 조건으로 로그 저장
            now = time.time()
            if new_status != '정상' and (
                new_status != last_logged['상태'] or
                now - last_logged['time'] > COOLTIME):
                save_log(cid, new_status, lat, lng, boxes)
                last_logged = {'상태': new_status, 'time': now}

        f += 1



# ───── 인증/라우팅 ─────
@app.route('/')
def root(): return redirect(url_for('dashboard') if 'user' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form['username']=='admin' and request.form['password']=='admin':
            session['user']='admin'; return redirect(url_for('dashboard'))
        return render_template('login.html', error='로그인 실패')
    return render_template('login.html')

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('login'))

# ─── Jinja 필터 등록 ───
@app.template_filter('basename')
def _basename(p):
    from pathlib import Path
    return Path(p).name

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/show_logs')
def show_logs():
    page=max(int(request.args.get('page',1)),1); size=10; off=(page-1)*size
    conn=db()
    rows = conn.execute(
        """SELECT l.*,c.src FROM logs l JOIN cctvs c ON l.cctv_id=c.id
           ORDER BY l.id DESC LIMIT ? OFFSET ?""",(size, off)).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]; conn.close()
    page_cnt = (total + size - 1) // size
    return render_template('logs.html', logs=rows, page=page, page_cnt=page_cnt)

@app.route('/api/dashboard_data')
def api_dash():
    conn=db()
    cams=[dict(r) for r in conn.execute("SELECT * FROM cctvs").fetchall()]
    logs=[dict(r) for r in conn.execute(
        "SELECT id,ts,msg,cctv_id,lat,lng FROM logs ORDER BY id DESC LIMIT 5").fetchall()]
    conn.close()
    for c in cams:
        c['stream_url'] = url_for('static', filename='videos/'+Path(c['src']).name)
    for l in logs:
        cam = next(x for x in cams if x['id']==l['cctv_id'])
        l['stream_url'] = cam['stream_url']
    return jsonify({'cctv_list':cams,'log_list':logs})

@app.route('/api/ack_alert', methods=['POST'])
def ack_alert():
    cid = int(request.json['cctv_id'])
    muted.add(cid)
    with db_lock:
        conn = db()
        conn.execute("UPDATE cctvs SET status='정상' WHERE id=?", (cid,))
        conn.commit(); conn.close()
    return jsonify(ok=True)

# ───── 실행 흐름 ─────
def start_monitors():
    for cid, src in db().execute("SELECT id,src FROM cctvs"):
        threading.Thread(target=monitor, args=(cid, Path(src)), daemon=True).start()
        print(f'[START] CCTV-{cid} 모니터링 스레드')

if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    init_db(); sync_videos_with_db()
    start_monitors()
    threading.Thread(target=sync_scheduler, daemon=True).start()

if __name__ == '__main__':
    init_db(); sync_videos_with_db()
    start_monitors()
    threading.Thread(target=sync_scheduler, daemon=True).start()
    app.run(debug=True, threaded=True)