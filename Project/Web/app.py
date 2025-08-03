# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # 세션을 사용하기 위해 반드시 설정해야 합니다.

# '/login' URL (메인 페이지)에 GET, POST 요청을 모두 허용합니다.
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 폼에서 제출된 아이디와 비밀번호를 가져옵니다.
        username = request.form.get('username')
        password = request.form.get('password')

        # 간단한 테스트 계정으로 인증 로직을 구현합니다.
        # 실제 서비스에서는 데이터베이스와 연동하여 확인해야 합니다.
        if username == 'admin' and password == 'adminpass':
            session['logged_in'] = True
            session['username'] = username
            flash('로그인에 성공했습니다.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'danger')
            return redirect(url_for('login'))
    
    # GET 요청 시 login.html 페이지를 보여줍니다.
    return render_template('login.html')

# 대시보드 페이지에 접근하기 전에 로그인 상태인지 확인합니다.
@app.route('/dashboard')
def dashboard():
    if 'logged_in' in session and session['logged_in']:
        return render_template('dashboard.html')
    else:
        flash('로그인이 필요합니다.', 'info')
        return redirect(url_for('login'))

# 로그아웃 기능
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('로그아웃 되었습니다.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)