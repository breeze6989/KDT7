# app.py  — modal+logs names/coords ready
# -*- coding: utf-8 -*-

import os, time, threading, sqlite3, datetime, requests, json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, request, redirect, url_for, session

# ─────────────────────────────────────────────────────────────────────────
# 기본 경로/상수
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
VIDEO_DIR  = STATIC_DIR / "videos"
MODEL_DIR  = BASE_DIR / "models"
DB_PATH    = BASE_DIR / "fire_dash.db"

MODEL_PATH = MODEL_DIR / "fire_detector.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 클래스/임계값
FIRE_IDX   = 0
SMOKE_IDX  = 2
CONF_THR   = 0.5
IOU_THR    = 0.45

# 동작 파라미터
COOLDOWN   = 600   # 로그 쿨타임(초)
PRED_EVERY = 3     # N프레임마다 추론

# 노출 개수
LOCAL_LIMIT = 4
ITS_LIMIT   = 8

# ITS API (요청 영역과 키는 환경/파일에 보관 권장)
ITS_API_URL = (
    "https://openapi.its.go.kr:9443/cctvInfo"
    "?apiKey=d466347e68dc400c980b21840da42153"
    "&type=ex&cctvType=4"
    "&minX=128.231398&maxX=128.915359&minY=35.647893&maxY=36.053498"
    "&getType=json"
)

app = Flask(__name__)
app.secret_key = "CHANGE_THIS_KEY"

# ─────────────────────────────────────────────────────────────────────────
# DB helpers

db_lock = threading.Lock()

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn

def init_db():
    conn = db(); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cctvs(
            id      INTEGER PRIMARY KEY,
            name    TEXT,
            src     TEXT,
            status  TEXT,
            lat     REAL,
            lng     REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id       INTEGER PRIMARY KEY,
            cctv_id  INTEGER,
            ts       TEXT,
            msg      TEXT,
            lat      REAL,
            lng      REAL
        )
    """)
    conn.commit(); conn.close()

def ensure_column(conn, table: str, column: str, type_def: str):
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}")

def migrate_db():
    conn = db()
    ensure_column(conn, "cctvs", "is_hls", "INTEGER DEFAULT 0")
    ensure_column(conn, "logs",  "boxes_json", "TEXT")
    conn.execute("UPDATE cctvs SET is_hls=1 WHERE LOWER(src) LIKE '%%.m3u8%%'")
    conn.commit(); conn.close()

# ─────────────────────────────────────────────────────────────────────────
# ITS JSON 파서(중첩 구조 대응)

def _walk_find_items(o, bag: List[Dict[str, Any]]):
    if isinstance(o, dict):
        # cctvurl이 있으면 아이템으로 수집
        if any(k.lower() == "cctvurl" for k in o.keys()):
            bag.append(o)
        for v in o.values(): _walk_find_items(v, bag)
    elif isinstance(o, list):
        for e in o: _walk_find_items(e, bag)

def parse_its_items(limit: int = ITS_LIMIT) -> List[Dict[str, Any]]:
    try:
        r = requests.get(ITS_API_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ITS][ERR] {e}")
        return []

    raw: List[Dict[str, Any]] = []
    _walk_find_items(data, raw)

    out, seen = [], set()
    for it in raw:
        url = it.get("cctvurl") or it.get("CCTVURL") or ""
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if any(url.lower().endswith(ext) for ext in (".jpg",".jpeg",".png",".gif")):
            continue
        if url in seen:  # 중복 제거
            continue
        seen.add(url)

        name = it.get("cctvname") or it.get("CCTVNAME") or "ITS"
        lat  = it.get("coordy") or it.get("y") or it.get("latitude")
        lng  = it.get("coordx") or it.get("x") or it.get("longitude")
        try:
            lat = float(lat) if lat is not None else None
            lng = float(lng) if lng is not None else None
        except Exception:
            lat = lng = None

        out.append({"url": url.strip(), "name": str(name).strip(), "lat": lat, "lng": lng})
        if len(out) >= limit: break
    return out

# ─────────────────────────────────────────────────────────────────────────
# 시드(정확히 로컬 4 + ITS 8만 채움)

def seed_local_videos(limit=LOCAL_LIMIT) -> int:
    files = sorted(VIDEO_DIR.glob("*.mp4"))[:limit]
    if not files:
        print("[SEED][LOCAL] mp4 없음:", VIDEO_DIR)
        return 0
    conn = db(); cur = conn.cursor()
    existing = cur.execute("SELECT COUNT(*) FROM cctvs WHERE is_hls=0").fetchone()[0]
    need = max(0, limit - existing); added = 0
    for p in files:
        if need <= 0: break
        if cur.execute("SELECT 1 FROM cctvs WHERE src=?", (str(p),)).fetchone():
            continue
        idx = existing + added + 1
        lat = 37.55 + idx*0.0002; lng = 126.97 + idx*0.0002  # 임의 좌표
        cur.execute(
            "INSERT INTO cctvs (name, src, status, lat, lng, is_hls) VALUES (?,?,?,?,?,?)",
            (f"LOCAL-{idx}", str(p), "정상", lat, lng, 0)
        )
        added += 1; need -= 1
    conn.commit(); conn.close()
    print(f"[SEED][LOCAL] added: {added}")
    return added

def seed_its_cctv(limit=ITS_LIMIT) -> int:
    items = parse_its_items(limit=limit)
    if not items:
        print("[SEED][ITS] parse fail or empty")
        return 0
    conn = db(); cur = conn.cursor()
    existing = cur.execute("SELECT COUNT(*) FROM cctvs WHERE is_hls=1").fetchone()[0]
    need = max(0, limit - existing); added = 0
    for i, it in enumerate(items, 1):
        if need <= 0: break
        url, name = it["url"], it["name"]
        if cur.execute("SELECT 1 FROM cctvs WHERE src=?", (url,)).fetchone():
            continue
        idx = existing + added + 1
        lat = it["lat"] if it["lat"] is not None else 37.60 + idx*0.0003
        lng = it["lng"] if it["lng"] is not None else 126.99 + idx*0.0003
        cur.execute(
            "INSERT INTO cctvs (name, src, status, lat, lng, is_hls) VALUES (?,?,?,?,?,?)",
            (name if name else f"ITS-{idx}", url, "정상", lat, lng, 1)
        )
        added += 1; need -= 1
    conn.commit(); conn.close()
    print(f"[SEED][ITS] added: {added}")
    return added

def get_active_cams() -> List[Dict[str, Any]]:
    conn = db()
    locals_ = [dict(r) for r in conn.execute(
        "SELECT * FROM cctvs WHERE is_hls=0 ORDER BY id ASC LIMIT ?", (LOCAL_LIMIT,)
    ).fetchall()]
    its_ = [dict(r) for r in conn.execute(
        "SELECT * FROM cctvs WHERE is_hls=1 ORDER BY id ASC LIMIT ?", (ITS_LIMIT,)
    ).fetchall()]
    conn.close()
    seen, out = set(), []
    for row in locals_ + its_:
        if row["src"] in seen: continue
        seen.add(row["src"]); out.append(row)
    return out[:(LOCAL_LIMIT+ITS_LIMIT)]

# ─────────────────────────────────────────────────────────────────────────
# 상태/로그 저장

state_lock: threading.Lock = threading.Lock()
shared_state: Dict[int, Dict[str, Any]] = {}
last_log_time: Dict[int, float] = {}

def save_log(cctv_id:int, status:str):
    """카메라 이름으로 메시지 저장 + 쿨타임 적용"""
    if status == "정상":
        return
    now = time.time()
    if now - last_log_time.get(cctv_id, 0) < COOLDOWN:
        return

    with db_lock:
        conn = db()
        cam = conn.execute("SELECT name,lat,lng FROM cctvs WHERE id=?", (cctv_id,)).fetchone()
        if not cam:
            conn.close(); return
        name, lat, lng = cam["name"], cam["lat"], cam["lng"]
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{name} {status}"                # ← 이름으로 저장 (요청사항)
        conn.execute(
            "INSERT INTO logs (cctv_id, ts, msg, lat, lng) VALUES (?,?,?,?,?)",
            (cctv_id, ts, msg, lat, lng)
        )
        conn.execute("UPDATE cctvs SET status=? WHERE id=?", (status, cctv_id))
        conn.commit(); conn.close()
    last_log_time[cctv_id] = now

# ─────────────────────────────────────────────────────────────────────────
# 추론/스레드

def run_model_on_frame(model, frame):
    r = model.predict(
        frame, conf=CONF_THR, iou=IOU_THR,
        classes=[FIRE_IDX, SMOKE_IDX], device=DEVICE, verbose=False
    )[0]
    fire = smoke = False
    if r.boxes is not None and len(r.boxes) > 0:
        cls = r.boxes.cls.cpu().numpy()
        fire  = any(int(c)==FIRE_IDX  for c in cls)
        smoke = any(int(c)==SMOKE_IDX for c in cls)
    return "화재감지" if fire else ("주의" if smoke else "정상")

def camera_worker(cctv: Dict[str, Any]):
    cid  = cctv["id"]; src = cctv["src"]; name = cctv["name"]
    is_hls = int(cctv.get("is_hls",0))

    # 모델(각 소스별 1개)
    try:
        model = YOLO(str(MODEL_PATH)).to(DEVICE)
        print(f"[MODEL] {name} loaded on {DEVICE}")
    except Exception as e:
        print(f"[MODEL][ERR] {name}: {e}")
        return

    cap = cv2.VideoCapture(src if is_hls else str(src), cv2.CAP_FFMPEG if is_hls else 0)
    if not cap.isOpened():
        print(f"[CAP][ERR] open fail: {name} -> {src}")
        return

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            if not is_hls:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 로컬은 반복재생
            continue

        if f % PRED_EVERY == 0:
            try:
                status = run_model_on_frame(model, frame)
            except Exception as e:
                print(f"[PRED][ERR] {name}: {e}")
                status = "정상"

            # 상태 공유
            with state_lock:
                shared_state[cid] = {"status": status}

            # 로그 저장
            save_log(cid, status)

        f += 1

# ─────────────────────────────────────────────────────────────────────────
# 필터/페이지

@app.template_filter("basename")
def _basename(p): return Path(p).name

@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("username")=="admin" and request.form.get("password")=="admin":
            session["user"]="admin"; return redirect(url_for("dashboard"))
        return render_template("login.html", error="로그인 실패")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/show_logs")
def show_logs():
    if "user" not in session: return redirect(url_for("login"))
    page = max(int(request.args.get("page",1)),1)
    size = 10; off = (page-1)*size
    conn = db()
    rows = conn.execute("""
        SELECT l.*, c.name AS cctv_name, c.src, c.is_hls
        FROM logs l JOIN cctvs c ON l.cctv_id=c.id
        ORDER BY l.id DESC LIMIT ? OFFSET ?
    """,(size,off)).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    conn.close()
    page_cnt = (total + size - 1)//size
    return render_template("logs.html", logs=rows, page=page, page_cnt=page_cnt)

# ─────────────────────────────────────────────────────────────────────────
# 대시보드/모달 API

@app.route("/api/dashboard_data")
def api_dashboard_data():
    cams = get_active_cams()  # 로컬4 + ITS8
    id_map = {}
    for c in cams:
        if int(c.get("is_hls",0))==1:
            c["stream_url"] = c["src"]
        else:
            c["stream_url"] = url_for("static", filename=f"videos/{Path(c['src']).name}")
        id_map[c["id"]] = c

    # 사이드 로그: 이름/좌표/스트림 포함해서 프론트가 그대로 사용
    conn = db()
    logs = [dict(r) for r in conn.execute("""
        SELECT l.id, l.ts, l.cctv_id, l.msg, l.lat, l.lng,
               c.name AS cctv_name, c.src, c.is_hls
        FROM logs l JOIN cctvs c ON l.cctv_id=c.id
        ORDER BY l.id DESC LIMIT 5
    """).fetchall()]
    conn.close()
    for l in logs:
        cam = id_map.get(l["cctv_id"])
        if cam:
            l["stream_url"] = cam["stream_url"]
        else:
            # 백업 스트림 계산
            l["stream_url"] = l["src"] if int(l.get("is_hls",0))==1 \
                              else url_for("static", filename=f"videos/{Path(l['src']).name}")
        # 상태 텍스트(저장된 msg의 마지막 단어 기준)
        tail = str(l["msg"]).split()[-1] if l.get("msg") else ""
        l["status"] = tail if tail in ("화재감지","주의","정상") else "정상"
    return jsonify({"cctv_list": cams, "log_list": logs})

@app.route("/api/cctv_info")
def api_cctv_info():
    """모달용 단일 카메라 정보"""
    cid = int(request.args.get("id", 0))
    conn = db()
    row = conn.execute("SELECT * FROM cctvs WHERE id=?", (cid,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error":"not found"}), 404
    row = dict(row)
    if int(row.get("is_hls",0))==1:
        stream = row["src"]
    else:
        stream = url_for("static", filename=f"videos/{Path(row['src']).name}")
    with state_lock:
        st = shared_state.get(cid, {"status":"정상"})
    return jsonify({
        "id": row["id"], "name": row["name"], "stream_url": stream,
        "lat": row["lat"], "lng": row["lng"], "status": st["status"]
    })

@app.route("/api/boxes_all")
def api_boxes_all():
    """프론트가 상태 배지만 갱신할 수 있도록 status만 전달"""
    active_ids = [c["id"] for c in get_active_cams()]
    items = []
    with state_lock:
        for cid in active_ids:
            items.append({
                "id": cid,
                "status": shared_state.get(cid, {"status":"정상"})["status"],
                "boxes": []  # bbox는 화면에 그리지 않도록 항상 빈 배열
            })
    return jsonify({"items": items})

# ─────────────────────────────────────────────────────────────────────────
# 관리/부팅

@app.route("/admin/reset")
def admin_reset():
    if "user" not in session: return redirect(url_for("login"))
    with db_lock:
        conn = db()
        conn.execute("DELETE FROM logs")
        conn.execute("DELETE FROM cctvs")
        conn.commit(); conn.close()
    seed_local_videos(LOCAL_LIMIT)
    seed_its_cctv(ITS_LIMIT)
    return redirect(url_for("dashboard"))

def boot_once():
    init_db()
    migrate_db()
    seed_local_videos(LOCAL_LIMIT)
    seed_its_cctv(ITS_LIMIT)
    for c in get_active_cams():
        threading.Thread(target=camera_worker, args=(c,), daemon=True).start()
        print(f"[THREAD] start -> {c['name']} | {c['src']}")

if __name__ == "__main__":
    # 플라스크 리로더로 인한 중복 방지
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        boot_once()
    app.run(debug=True, threaded=True)

