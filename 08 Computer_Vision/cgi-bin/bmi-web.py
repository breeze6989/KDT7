"""
# URL : http://localhost:8080/cgi-bin/bmi_web.py
"""
# 모듈 로딩 ---------------------------------------------------
import cgi, sys, codecs, os
import joblib

# WEB 인코딩 설정 ---------------------------------------------
sys.stdout=codecs.getwriter('utf-8')(sys.stdout.detach())

# 함수 선언 --------------------------------------------------
# WEB 페이지 출력 --------------------------------------------
def displayWEB(detect_msg):
    print("Content-Type: text/html; charset=utf-8")
    print("")
    html="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>비만 여부 판별</title>
    </head>
    <body align="center">
    <h2>[ 비만 상태 체크 ]</h2>
    <form>
        <div style='text-align:center; background-color:#D5D5D5;border-radius:10px;width:60%; margin: auto;padding:50px;'>
            <input id="height" type="text" placeholder="키" name="height"> &nbsp&nbsp
            <input id="weight" type="text" placeholder="몸무게" name="weight">
            <input type="submit" value="판정"></br>
            <p><font color='blue'>{}</font></p>
        </div>
    </form></body></html>""".format(detect_msg)
    print(html)

# 판정 --------------------------------------------------------
def detect_bmi(w, h):
    w = int(w)
    h = int(h)
    # 비만도  예측하기
    res = clf.predict([[w / 100, h / 200]])
    return str(res[0])

# 기능 구현 -----------------------------------------------------
# (1) 학습 데이터 읽기
pklfile = os.path.dirname(__file__) + "/bmi.pkl"
clf = joblib.load(pklfile)

# (2) WEB 페이지 <Form> -> <INPUT> 리스트 가져오기
form = cgi.FieldStorage()
height_value = form.getvalue('height')
weight_value = form.getvalue('weight')

# (3) 판정 하기
if height_value is not None and weight_value is not None:
    bmi_dic = {"fat": "과체중", "normal": "정상체중", "thin": "저체중"}
    result = detect_bmi(weight_value, height_value)
    result = '키 {}, 몸무게 {} => {}입니다.'.format(height_value, weight_value, bmi_dic[result])
else:
    result ='측정된 결과가 없습니다.'

# (4) WEB 출력하기
displayWEB(result)

