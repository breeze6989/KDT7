import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import koreanize_matplotlib
import pandas as pd
import re

# 파일 로드
file_path = "합격자데이터.csv"
df = pd.read_csv(file_path)

# 학력 데이터 파이 차트 (각 항목 색상을 명확하게 구분)
education_counts = df['학력'].value_counts()
colors = ["#3399FF", "#33FF57", "#3357FF", "#FF33A1", "#A133FF", "#FFDD44","#FF4500"]  # 색상 구분 강화
plt.figure(figsize=(8, 8))
wedges, texts = plt.pie(
    education_counts, labels=None, startangle=140, colors=colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

# 범례 추가 (% 값 표시)
total = sum(education_counts)
plt.legend(wedges, [f"{label} {count / total * 100:.1f}%" for label, count in zip(education_counts.index, education_counts)], 
           loc="center left", bbox_to_anchor=(1, 0.5))

plt.title('학력 분포')
plt.show()

# 학과 데이터 파이 차트 (상위 5개만 표시)
department_counts = df['학과'].value_counts().nlargest(5)
plt.figure(figsize=(6, 6))
plt.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('학과 분포 (상위 5개)')
plt.show()

# 학과 데이터 파이 차트 (나머지 항목 표시)
other_departments = df['학과'].value_counts().iloc[5:]
plt.figure(figsize=(6, 12))
wedges, texts = plt.pie(other_departments, labels=None, startangle=140, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

# 범례 추가 (% 값 표시)
total_other = sum(other_departments)
plt.legend(wedges, [f"{label} {count / total_other * 100:.1f}%" for label, count in zip(other_departments.index, other_departments)], 
           loc="center left", bbox_to_anchor=(1, 0.5))

plt.title('학과 분포 (기타)')
plt.show()

# 학점 분포 (분자만 추출하여 숫자로 변환)
df['학점_분자'] = df['학점'].str.extract(r'(\d+\.\d+)').astype(float)
plt.figure(figsize=(8, 5))
plt.hist(df['학점_분자'].dropna(), bins=10, edgecolor='black')
plt.xlabel('학점')
plt.ylabel('빈도수')
plt.title('학점 분포')
plt.show()

# 어학 성적 워드 클라우드 생성 (윈도우 기준 한글 기본 폰트 설정, '토익' 제거, '오픽'과 '토스' 이후 값만 유지)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
language_scores = ' '.join(df['어학'].dropna().astype(str))
processed_scores = []

for word in language_scores.split():
    if '토익' in word:
        continue  # '토익' 포함 단어 제거
    elif '오픽' in word:
        processed_scores.append(word.replace('오픽', '').strip())
    elif '토스' in word:
        processed_scores.append(word.replace('토스', '').strip())
    else:
        processed_scores.append(word)

wordcloud_text = ' '.join(filter(None, processed_scores))

wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=400).generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('어학 성적 워드 클라우드')
plt.show()

# 어학 성적 (토익) 점수 분포 그래프 생성
toefl_scores = df['어학'].dropna().astype(str).str.extract(r'토익\s*(\d+)')[0].dropna().astype(int)
plt.figure(figsize=(8, 5))
plt.hist(toefl_scores, bins=10, edgecolor='black', color='skyblue')
plt.xlabel('토익 점수')
plt.ylabel('빈도수')
plt.title('토익 점수 분포')
plt.show()

certificate_list = []

# df에 포함된 모든 열 순회
for col in df.columns:
    # 해당 열에서 NaN 제거, 문자열 변환
    col_data = df[col].dropna().astype(str)
    
    # 각 셀에 대해 "자격증" 패턴 찾기
    for cell in col_data:
        # 예: "자격증 전기기사, 자격증 컴활1급" 내부에서 여러 번 추출 가능
        # 정규식에서, 자격증(\s*) 뒤의 공백을 포함하여,
        # 이후 공백/콤마/세미콜론 등 구분자를 만나기 전까지 한 덩어리씩 추출
        # 필요에 따라 패턴을 조정할 수 있습니다.
        matches = re.findall(r'자격증\s*([^\s,;]+)', cell)
        # 추출된 항목을 certificate_list에 추가
        certificate_list.extend(matches)

# 만약 "자격증" 뒤에 이어지는 문자열을 전부(공백 포함) 가져와야 한다면:
#   matches = re.findall(r'자격증\s*(.*)', cell)
# 처럼 사용해도 됩니다. 상황에 맞춰 수정하세요.

# --------------------------------------------------
# 자격증 이름/종류별 빈도수 계산 및 시각화
# --------------------------------------------------

if len(certificate_list) > 0:
    cert_series = pd.Series(certificate_list)
    cert_counts = cert_series.value_counts()

    # 막대 그래프
    plt.figure(figsize=(10, 6))
    cert_counts.plot(kind='bar', color='skyblue')
    plt.title('자격증 분포')
    plt.xlabel('자격증명')
    plt.ylabel('빈도수')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("자격증을 포함하는 데이터가 없습니다.")