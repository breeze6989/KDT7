import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib  # 한글 폰트 깨짐 문제 방지
import os

# CSV 파일 경로(예: job_data.csv 가정)
file_path = r"C:\Users\KDP-50\OneDrive\바탕 화면\KDT7\06_webcrawling\wordcloud\사람인_채용정보_최종_데이터엔지니어.csv"

# CSV 불러오기
df = pd.read_csv(file_path)

# =============================================================================
# 1. 근무지역 파이차트 (서울,경기 vs 지방), 결측치는 제외
# =============================================================================
def classify_region(loc_str):
    """
    loc_str에 '서울' 또는 '경기'가 들어있으면 '서울,경기',
    그 외는 '지방'. 결측치는 None -> 이후 dropna로 제외.
    """
    if pd.isnull(loc_str):
        return None
    loc_str = str(loc_str)
    if ("서울" in loc_str) or ("경기" in loc_str):
        return "서울,경기"
    else:
        return "지방"

df["근무지역_전처리"] = df["근무지역"].apply(classify_region)

# 결측치 제외
region_filtered = df.dropna(subset=["근무지역_전처리"])
region_counts = region_filtered["근무지역_전처리"].value_counts()

plt.figure(figsize=(6, 6))
region_counts.plot.pie(autopct="%.1f%%", startangle=140)
plt.title("근무지역 파이차트 (서울,경기 vs 지방, 결측치 제외)")
plt.ylabel("")
plt.show()


# =============================================================================
# 2. 근무형태 막대그래프 (계약직 포함되면 '계약직'으로 통합), 
#    범례(legend)에 x축 라벨 표시
# =============================================================================
def classify_worktype(work_str):
    """
    '계약직'이라는 텍스트가 들어 있으면 '계약직'으로 통일,
    결측치는 '기타'.
    """
    if pd.isnull(work_str):
        return "기타"
    work_str = str(work_str).strip()
    if "계약직" in work_str:
        return "계약직"
    return work_str

df["근무형태_전처리"] = df["근무형태"].apply(classify_worktype)
worktype_counts = df["근무형태_전처리"].value_counts()

plt.figure(figsize=(8, 5))
x_positions = range(len(worktype_counts))
# 각각의 항목에 대해 bar를 그리면서 label=카테고리 지정 -> legend로 표시
for i, (category, count) in enumerate(worktype_counts.items()):
    plt.bar(i, count, label=category)

plt.title("근무형태 막대그래프 (계약직 통합) - Legend로 카테고리 표시")
plt.ylabel("빈도 수")
plt.xticks([])  # x축 숫자 숨김
plt.legend()

# 막대 위에 데이터 표시
sum_y = worktype_counts.sum()
for i, count in enumerate(worktype_counts):
    perc = count / sum_y * 100
    plt.text(i, count + 0.3, f"{count}건 ({perc:.1f}%)", ha='center')

plt.ylim(0, max(worktype_counts)*1.2)
plt.tight_layout()
plt.show()


# =============================================================================
# 3. 급여정보 막대그래프 (수정): 
#    '면접 후 결정', '회사 내규에 따름' 그대로, 나머지는 '기타'로 묶어 3개 막대 표시
# =============================================================================
def classify_pay(pay_str):
    """
    '면접 후 결정', '회사 내규에 따름'이면 그대로,
    그 외(결측치 포함)는 '기타'로 분류
    """
    if pd.isnull(pay_str):
        return "기타"
    pay_str = str(pay_str).strip()
    if pay_str in ["면접 후 결정", "회사내규에 따름"]:
        return pay_str
    else:
        return "기타"

df["급여정보_전처리"] = df["급여정보"].apply(classify_pay)
pay_counts = df["급여정보_전처리"].value_counts()
# 결과는 최대 3개 (면접 후 결정 / 회사 내규에 따름 / 기타)

plt.figure(figsize=(8, 5))
x = range(len(pay_counts))
y = pay_counts.values
labels = pay_counts.index

plt.bar(x, y, tick_label=labels)
plt.title("급여정보 막대그래프 (면접 후 결정/회사 내규에 따름/기타)")
plt.xlabel("급여 유형")
plt.ylabel("빈도 수")

# 막대 위에 개수 및 퍼센트
sum_y = sum(y)
for i, v in enumerate(y):
    perc = v / sum_y * 100
    plt.text(i, v + 0.3, f"{v}건 ({perc:.1f}%)", ha='center', fontsize=9)

plt.ylim(0, max(y)*1.2)
plt.tight_layout()
plt.show()


# =============================================================================
# 4. 근무형태 파이차트 (수정): 범례(legend)에 데이터 수치(개수, %) 표시
#    파이조각 내부에는 라벨이 없음
# =============================================================================
worktype_counts_for_pie = df["근무형태_전처리"].value_counts()

plt.figure(figsize=(6,6))
# 파이 조각에 라벨 대신 None 설정
wedges, _ = plt.pie(worktype_counts_for_pie, startangle=140)

# 범례(legend)에 표시할 내용: 카테고리 + (개수 + 퍼센트)
total = worktype_counts_for_pie.sum()
legend_labels = []
for cat, val in zip(worktype_counts_for_pie.index, worktype_counts_for_pie):
    pct = val / total * 100
    legend_labels.append(f"{cat} ({val}건, {pct:.1f}%)")

plt.legend(wedges, legend_labels, title="근무형태", loc="center left", 
           bbox_to_anchor=(1, 0.5))

plt.title("근무형태 파이차트 (범례에 수치 표시)")
plt.tight_layout()
plt.show()


# =============================================================================
# 5. 언어 열 파이차트 (수정사항 없음)
# =============================================================================
lang_counts = df["언어"].value_counts(dropna=True)  # 필요시 dropna=False
plt.figure(figsize=(6,6))
lang_counts.plot.pie(autopct='%.1f%%', startangle=140)
plt.title("언어 파이차트")
plt.ylabel("")
plt.show()