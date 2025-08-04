# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
import sqlite3
import os
import datetime
import random
import threading
import time

app = Flask(__name__)
app.config.from_pyfile('config.py')

# 데이터베이스 파일 경로 설정
DATABASE = 'site.db'

# --- 데이터베이스 헬퍼 함수 ---
def get_db_connection():
    """데이터베이스 연결을 반환하거나 새로 생성합니다."""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row # 컬럼 이름으로 접근 가능하게 설정
    return g.db

@app.teardown_appcontext
def close_db_connection(exception):
    """요청 종료 시 데이터베이스 연결을 닫습니다."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """데이터베이스 스키마를 초기화하고 초기 데이터를 삽입합니다."""
    with app.app_context():
        db = get_db_connection()
        
        # users 테이블 생성 (변동 없음)
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        
        # cctvs 테이블 생성 (변동 없음)
        db.execute('''
            CREATE TABLE IF NOT EXISTS cctvs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                stream_url TEXT NOT NULL,
                status TEXT NOT NULL, -- 정상, 화재감지, 오탐
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                recent_event TEXT, -- 최근 발생한 이벤트 내용
                last_updated TEXT -- 마지막 상태 업데이트 시간
            )
        ''')
        
        # logs 테이블 생성 (latitude, longitude, video_url 컬럼 추가)
        # 기존 app.py에는 video_url이 없었으므로 추가합니다.
        db.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL,
                cctv_id INTEGER, -- 어떤 CCTV에서 발생한 로그인지
                latitude REAL,   -- 감지 당시의 CCTV 위도
                longitude REAL,  -- 감지 당시의 CCTV 경도
                video_url TEXT,  -- 감지 당시의 영상 URL (또는 파일 경로)
                FOREIGN KEY (cctv_id) REFERENCES cctvs(id)
            )
        ''')
        db.commit()

        # 초기 사용자 데이터 삽입 (예시)
        cursor = db.execute("SELECT COUNT(*) FROM users WHERE username = 'user1'")
        if cursor.fetchone()[0] == 0:
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('user1', 'pass123'))
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', 'adminpass'))
            print("Initial users added.")
            db.commit()

        # 초기 CCTV 데이터 삽입 (예시)
        cursor = db.execute("SELECT COUNT(*) FROM cctvs")
        if cursor.fetchone()[0] == 0:
            for i in range(1, 11): # 10개의 CCTV 예시
                status = "정상" # 초기에는 모두 정상으로 설정
                recent_event = "최근 이벤트 없음"
                # 예시 영상 URL (실제 스트림 URL로 대체해야 함)
                # 시뮬레이션이 HLS.js와 연동되려면 HLS 스트림 URL이어야 합니다.
                video_url_template = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8" # 예시 HLS 스트림
                db.execute(
                    "INSERT INTO cctvs (name, stream_url, status, latitude, longitude, recent_event, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (f"CCTV #{i} 구역", video_url_template, status,
                     37.5665 + (random.random() - 0.5) * 0.05, # 임시 위도
                     126.9780 + (random.random() - 0.5) * 0.05, # 임시 경도
                     recent_event,
                     datetime.datetime.now().isoformat())
                )
            print("Initial CCTV data added.")
            db.commit()

# 애플리케이션 시작 시 데이터베이스 초기화
with app.app_context():
    init_db()

# --- 사용자 인증 및 세션 관리 ---
@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()

        if user:
            session['logged_in'] = True
            session['username'] = username
            flash('로그인 성공!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('잘못된 사용자 이름 또는 비밀번호입니다.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('login'))

# --- 대시보드 관련 엔드포인트 ---

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        flash('로그인이 필요합니다.', 'info')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """
    대시보드에 필요한 동적 데이터를 SQLite에서 가져와 제공하는 API 엔드포인트.
    """
    conn = get_db_connection()

    # CCTV 데이터 가져오기
    cctv_rows = conn.execute("SELECT id, name, stream_url, status, latitude, longitude, recent_event FROM cctvs").fetchall()
    cctv_data = []
    for row in cctv_rows:
        cctv_item = dict(row)
        cctv_item['cctv_id'] = cctv_item.pop('id')
        cctv_item['location'] = {
            'latitude': cctv_item.pop('latitude'),
            'longitude': cctv_item.pop('longitude')
        }
        cctv_data.append(cctv_item)

    # 로그 데이터 가져오기 (최신 10개) - log_id와 video_url도 함께 가져오도록 수정
    log_rows = conn.execute("SELECT id, timestamp, message, video_url FROM logs ORDER BY id DESC LIMIT 10").fetchall()
    log_list = []
    for row in log_rows:
        log_list.append({
            'id': row['id'], # 로그 ID 추가
            'timestamp': row['timestamp'],
            'message': row['message'],
            'video_url': row['video_url'] # 비디오 URL 추가
        })

    # 통계 데이터 계산
    total_cctvs = len(cctv_data)
    fire_detections = sum(1 for c in cctv_data if c['status'] == '화재감지')
    false_positives = sum(1 for c in cctv_data if c['status'] == '오탐')
    false_positive_rate = (false_positives / total_cctvs * 100) if total_cctvs > 0 else 0

    return jsonify({
        "cctv_list": cctv_data,
        "log_list": log_list, # 이제 log_list는 객체 리스트를 포함
        "stats": {
            "total_cctvs": total_cctvs,
            "fire_detections": fire_detections,
            "false_positives_rate": f"{false_positive_rate:.1f}%"
        }
    })

@app.route('/api/detect_fire', methods=['POST'])
def detect_fire():
    """
    특정 CCTV에서 화재가 감지되었음을 알리고 DB에 로그를 저장하는 API.
    """
    data = request.get_json()
    cctv_id = data.get('cctv_id')
    
    if not cctv_id:
        return jsonify({"message": "CCTV ID is required."}), 400

    conn = get_db_connection()
    cctv = conn.execute("SELECT * FROM cctvs WHERE id = ?", (cctv_id,)).fetchone()

    if not cctv:
        return jsonify({"message": f"CCTV with ID {cctv_id} not found."}), 404

    # CCTV 상태를 '화재감지'로 업데이트
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"#{cctv_id} 구역 화재 감지 ({cctv['name']})"
    
    conn.execute(
        "UPDATE cctvs SET status = ?, recent_event = ?, last_updated = ? WHERE id = ?",
        ('화재감지', log_message, current_time, cctv_id)
    )
    
    # 로그 테이블에 기록 시 위치 정보 및 영상 URL 함께 저장
    conn.execute(
        "INSERT INTO logs (timestamp, message, cctv_id, latitude, longitude, video_url) VALUES (?, ?, ?, ?, ?, ?)",
        (f"[{current_time}]", log_message, cctv_id, cctv['latitude'], cctv['longitude'], cctv['stream_url']) # CCTV의 스트림 URL을 로그 영상으로 사용
    )
    conn.commit()

    print(f"Fire detected at CCTV ID {cctv_id}. Logged with location and video URL.")
    return jsonify({"message": "Fire detection logged and CCTV status updated successfully."}), 200

# 새로운 라우트: 모든 로그를 보여주는 페이지
@app.route('/logs')
def show_all_logs():
    if 'logged_in' not in session:
        flash('로그인이 필요합니다.', 'info')
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    # logs 테이블과 cctvs 테이블을 JOIN하여 관련 CCTV 정보를 함께 가져옵니다.
    # ORDER BY l.id DESC 를 사용하여 최신 로그부터 표시
    log_rows = conn.execute(
        """
        SELECT 
            l.id,               -- 로그 ID 추가 (클릭 식별용)
            l.timestamp, 
            l.message, 
            l.latitude, 
            l.longitude, 
            l.video_url,        -- 로그에 저장된 비디오 URL
            c.name AS cctv_name, 
            c.stream_url AS cctv_stream_url -- CCTV의 현재 스트림 URL
        FROM logs AS l
        LEFT JOIN cctvs AS c ON l.cctv_id = c.id
        ORDER BY l.id DESC
        """
    ).fetchall()
    
    all_logs = [dict(row) for row in log_rows]
    
    # query parameter를 통해 특정 로그의 상세 정보를 미리 로드
    selected_log_id = request.args.get('log_id', type=int)
    selected_log_detail = None
    if selected_log_id:
        for log in all_logs:
            if log['id'] == selected_log_id:
                selected_log_detail = log
                break

    return render_template('all_logs.html', all_logs=all_logs, selected_log=selected_log_detail)


# --- 임의 화재 감지 시뮬레이션 (개발 및 테스트용) ---
# 실제 운영 환경에서는 이 부분을 비활성화하거나 제거해야 합니다.
def simulate_fire_detection():
    with app.app_context(): # 별도의 스레드에서 DB 접근 시 필요
        while True:
            time.sleep(random.randint(10, 25)) # 10~25초마다 한 번씩
            conn = get_db_connection()
            cursor = conn.execute("SELECT id FROM cctvs")
            cctv_ids = [row['id'] for row in cursor.fetchall()]

            if cctv_ids:
                random_cctv_id = random.choice(cctv_ids)
                
                cctv_status_check = conn.execute("SELECT status FROM cctvs WHERE id = ?", (random_cctv_id,)).fetchone()
                
                if cctv_status_check and cctv_status_check['status'] == '화재감지':
                    # 이미 화재감지 상태인 경우, 20% 확률로 "정상"으로 되돌림 (오탐 시뮬레이션)
                    if random.random() < 0.2:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_message = f"#{random_cctv_id} 구역 오탐 발생 (정상으로 복귀)"
                        cctv_data = conn.execute("SELECT * FROM cctvs WHERE id = ?", (random_cctv_id,)).fetchone()
                        conn.execute(
                            "UPDATE cctvs SET status = ?, recent_event = ?, last_updated = ? WHERE id = ?",
                            ('정상', log_message, current_time, random_cctv_id)
                        )
                        conn.execute(
                            "INSERT INTO logs (timestamp, message, cctv_id, latitude, longitude, video_url) VALUES (?, ?, ?, ?, ?, ?)",
                            (f"[{current_time}]", log_message, random_cctv_id, cctv_data['latitude'], cctv_data['longitude'], cctv_data['stream_url'])
                        )
                        conn.commit()
                        print(f"Simulated false positive at CCTV ID {random_cctv_id}.")
                    continue # 다음 루프로

                cctv_data = conn.execute("SELECT * FROM cctvs WHERE id = ?", (random_cctv_id,)).fetchone()
                if cctv_data:
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = f"#{random_cctv_id} 구역 임의 화재 감지 시뮬레이션!"
                    
                    conn.execute(
                        "UPDATE cctvs SET status = ?, recent_event = ?, last_updated = ? WHERE id = ?",
                        ('화재감지', log_message, current_time, random_cctv_id)
                    )
                    conn.execute(
                        "INSERT INTO logs (timestamp, message, cctv_id, latitude, longitude, video_url) VALUES (?, ?, ?, ?, ?, ?)",
                        (f"[{current_time}]", log_message, random_cctv_id, cctv_data['latitude'], cctv_data['longitude'], cctv_data['stream_url'])
                    )
                    conn.commit()
                    print(f"Simulated fire detection at CCTV ID {random_cctv_id}.")
            close_db_connection(None) # 스레드에서 사용한 DB 연결 닫기


# --- 에러 핸들링 ---
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# --- 애플리케이션 실행 ---
if __name__ == '__main__':
    # 시뮬레이션 스레드 시작 (개발 및 테스트용)
    # 실제 운영 환경에서는 이 부분을 주석 처리하거나 제거해야 합니다.
    simulate_thread = threading.Thread(target=simulate_fire_detection)
    simulate_thread.daemon = True # 메인 스레드 종료 시 함께 종료
    simulate_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000)