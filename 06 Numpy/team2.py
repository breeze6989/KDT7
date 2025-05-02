import pandas as pd
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# =====================================================================================
# 1. 데이터 불러오기 및 전처리
# =====================================================================================
file_path = './학생데이터.csv'  # csv 파일 경로 지정
df = pd.read_csv(file_path, encoding='utf-8')

# 필요 없는 열들(예: 인덱스 등) 제거
df = df.drop(df.columns[[1,2,3]], axis=1)

# 열 이름 재정의
df.columns = [
    'ID','성별','나이','전공','참석률','중간고사','기말고사','과제','퀴즈','참여점수',
    '프로젝트','종합점수','등급','1주 공부시간','과외활동','인터넷가능여부',
    '부모교육수준','가족소득수준','스트레스레벨','수면시간'
]

# (1) 결측치 처리 (간단 예시)
df['참석률'] = df['참석률'].fillna(df['참석률'].mean())
df['과제'] = df['과제'].fillna(df['과제'].mean())
df['참여점수'] = df['참여점수'].fillna(df['참여점수'].mean())

# (2) 부모교육수준 변경: No→무학력, High School→고졸, Bachelor's→학사, Master's→석사, PhD→박사
df['부모교육수준'] = df['부모교육수준'].map({
    'No': '무학력',
    'High School': '고졸',
    "Bachelor's": '학사',
    "Master's": '석사',
    'PhD': '박사'
})
# 혹시 남은 결측치 있으면 무학력으로 처리
df['부모교육수준'] = df['부모교육수준'].fillna('무학력')

# (3) 가족소득수준 변경: Low→낮음, Medium→보통, High→높음
df['가족소득수준'] = df['가족소득수준'].map({
    'Low': '낮음',
    'Medium': '보통',
    'High': '높음'
})
# 혹시 남은 결측치 있으면 낮음으로 처리
df['가족소득수준'] = df['가족소득수준'].fillna('낮음')

# (4) 과외활동, 인터넷가능여부 → 숫자(0,1) 변환
df['과외활동'] = df['과외활동'].map({'No': 0, 'Yes': 1})
df['인터넷가능여부'] = df['인터넷가능여부'].map({'No': 0, 'Yes': 1})

# =====================================================================================
# 2. (수정) 파이그래프로 "과외활동 여부에 따른 부모교육수준 분포"와 "가족소득수준 분포"
# =====================================================================================

# -------------------------------------------------------------------------------------
# (A) 과외활동(0) vs 부모교육수준
# -------------------------------------------------------------------------------------
# 순서 지정: 무학력, 고졸, 학사, 석사, 박사
edu_order = ['무학력','고졸','학사','석사','박사']

fig, axes = plt.subplots(1,2, figsize=(12,6))

for i, activity_val in enumerate([0,1]):
    subset = df[df['과외활동'] == activity_val]['부모교육수준'].value_counts()
    # reindex로 순서 강제
    subset = subset.reindex(edu_order, fill_value=0)
    
    # 파이차트
    axes[i].pie(subset, labels=subset.index, autopct='%1.1f%%', startangle=140)
    axes[i].set_title(f"과외활동={activity_val} (부모교육수준)")

plt.suptitle("과외활동 여부(0=No,1=Yes)에 따른 부모교육수준 분포", fontsize=14)
plt.savefig("과외활동_부모교육수준_파이차트.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------------------
# (B) 과외활동(0) vs 가족소득수준
# -------------------------------------------------------------------------------------
# 순서 지정: 낮음, 보통, 높음
income_order = ['낮음','보통','높음']

fig, axes = plt.subplots(1,2, figsize=(12,6))

for i, activity_val in enumerate([0,1]):
    subset = df[df['과외활동'] == activity_val]['가족소득수준'].value_counts()
    # reindex로 순서 강제
    subset = subset.reindex(income_order, fill_value=0)
    
    # 파이차트
    axes[i].pie(subset, labels=subset.index, autopct='%1.1f%%', startangle=140)
    axes[i].set_title(f"과외활동={activity_val} (가족소득수준)")

plt.suptitle("과외활동 여부(0=No,1=Yes)에 따른 가족소득수준 분포", fontsize=14)
plt.savefig("과외활동_가족소득수준_파이차트.png", dpi=300)
plt.show()

# =====================================================================================
# (나머지는 동일) 과외활동에 따른 연속형 변수 Boxplot, 참여점수 vs 종합점수, 전공별 시각화 등
# =====================================================================================

# 2-(추가). 연속형 변수 Boxplot (스트레스레벨, 수면시간, 1주 공부시간, 종합점수)
fig, axes = plt.subplots(1, 4, figsize=(20,5))

sns.boxplot(x='과외활동', y='스트레스레벨', data=df, ax=axes[0])
axes[0].set_title('과외활동 여부에 따른 스트레스레벨')

sns.boxplot(x='과외활동', y='수면시간', data=df, ax=axes[1])
axes[1].set_title('과외활동 여부에 따른 수면시간')

sns.boxplot(x='과외활동', y='1주 공부시간', data=df, ax=axes[2])
axes[2].set_title('과외활동 여부에 따른 1주 공부시간')

sns.boxplot(x='과외활동', y='종합점수', data=df, ax=axes[3])
axes[3].set_title('과외활동 여부에 따른 종합점수')

plt.tight_layout()
plt.savefig("과외활동_연속형변수_비교.png", dpi=300)
plt.show()

# =====================================================================================
# 3. 참여점수에 따른 종합점수 시각화 (산점도) & png 저장
# =====================================================================================
plt.figure(figsize=(6,5))
plt.scatter(df['참여점수'], df['종합점수'], alpha=0.7)
plt.title('참여점수에 따른 종합점수')
plt.xlabel('참여점수')
plt.ylabel('종합점수')
plt.savefig("참여점수_vs_종합점수.png", dpi=300)
plt.show()

# =====================================================================================
# 4. 전공별(Department)로 그룹 나누어 동일 항목 시각화
# =====================================================================================
departments = df['전공'].unique()

for dept in departments:
    sub_df = df[df['전공'] == dept]
    print(f"\n========== 전공: {dept} ==========")
    
    # 예) 과외활동 여부 vs 스트레스 레벨 박스플롯
    plt.figure()
    sns.boxplot(x='과외활동', y='스트레스레벨', data=sub_df)
    plt.title(f'{dept} 전공 학생 - 과외활동(0/1) vs 스트레스레벨')
    plt.savefig(f"{dept}_과외활동_vs_스트레스레벨.png", dpi=300)
    plt.show()

    # 예) 참여점수 vs 종합점수 산점도
    plt.figure()
    plt.scatter(sub_df['참여점수'], sub_df['종합점수'])
    plt.title(f'{dept} 전공 학생 - 참여점수 vs 종합점수')
    plt.xlabel('참여점수')
    plt.ylabel('종합점수')
    plt.savefig(f"{dept}_참여점수_vs_종합점수.png", dpi=300)
    plt.show()

# =====================================================================================
# 5. ANOVA (또는 t-test), 회귀분석 - (부모교육수준→과외활동, 가족소득수준→참여점수)
# =====================================================================================

# ----------------------------------------------------------------------------
# (a) t-test: 과외활동(0 vs 1) -> 종합점수
# ----------------------------------------------------------------------------
group_yes = df[df['과외활동'] == 1]['종합점수']
group_no  = df[df['과외활동'] == 0]['종합점수']

ttest_result = stats.ttest_ind(group_yes, group_no, equal_var=False)
print("\n[과외활동 여부에 따른 종합점수] t-test 결과")
print("t-statistic:", ttest_result.statistic, "p-value:", ttest_result.pvalue)

# ----------------------------------------------------------------------------
# (b) ANOVA: 참여점수를 구간화하여 종합점수 차이
# ----------------------------------------------------------------------------
bins = [0, 30, 70, 100]
labels = ['Low', 'Medium', 'High']
df['참여점수_구간'] = pd.cut(df['참여점수'], bins=bins, labels=labels, include_lowest=True)

anova_result = stats.f_oneway(
    df[df['참여점수_구간'] == 'Low']['종합점수'],
    df[df['참여점수_구간'] == 'Medium']['종합점수'],
    df[df['참여점수_구간'] == 'High']['종합점수']
)
print("\n[참여점수 구간에 따른 종합점수] ANOVA 결과")
print("F-statistic:", anova_result.statistic, "p-value:", anova_result.pvalue)

# ----------------------------------------------------------------------------
# (c) 회귀분석: 종속변수=종합점수, 독립변수=과외활동(0/1), 참여점수(연속형), 인터넷가능여부(0/1)
# ----------------------------------------------------------------------------
# 혹시나 남아 있는 결측치 제거 + 전체를 float로 변환
df['종합점수'] = pd.to_numeric(df['종합점수'], errors='coerce')
df['참여점수'] = pd.to_numeric(df['참여점수'], errors='coerce')

X = df[['과외활동','인터넷가능여부','참여점수']]
y = df['종합점수']

# 결측치가 있다면 제거
Xy = pd.concat([X, y], axis=1).dropna()
X = Xy.drop(columns='종합점수')
y = Xy['종합점수']

# 회귀분석 (상수항 추가)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("\n[회귀분석 결과: 종합점수 ~ 과외활동 + 참여점수 + 인터넷가능여부]")
print(model.summary())

# ----------------------------------------------------------------------------
# (d) 예측 vs 실제 시각화 & png 저장
# ----------------------------------------------------------------------------
predictions = model.predict(X)
conf_int = model.get_prediction(X).conf_int()

plt.figure(figsize=(6,6))
plt.scatter(y, predictions, alpha=0.5, label='실제 vs 예측')
plt.fill_between(
    y, conf_int[:,0], conf_int[:,1], 
    color='gray', alpha=0.2, label='95% 신뢰구간'
)
line = np.linspace(min(y), max(y), 100)
plt.plot(line, line, 'r--', label='y=예측값(완벽예측)')

plt.xlabel('실제 종합점수')
plt.ylabel('예측 종합점수')
plt.title('OLS 회귀 결과: 실제 vs 예측')
plt.legend()
plt.savefig("OLS_회귀_실제_예측비교.png", dpi=300)
plt.show()

# =====================================================================================
# (e) 랜덤 포레스트 예시
# =====================================================================================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
y_pred_rf = rf_model.predict(X)
mse_rf = mean_squared_error(y, y_pred_rf)
print(f"[랜덤 포레스트 회귀] MSE: {mse_rf:.4f}")
