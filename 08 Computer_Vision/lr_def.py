import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.neighbors import *
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 1.3버전 - 한번에 다해주는 함수 및 이미지 처리 함수 추가, 추후에 다른 기능도 업데이트 예정


# 1. 데이터 전처리 및 모델 학습


# 함수 목차

# 1) 단일 피쳐의 상관계수 확인 및 산점도 그래프화
# 2) 데이터 전처리
#    - 2-1) PowerTransformer
#    - 2-2) Modified Z-score로 이상치 처리 (이상치는 중앙값으로 대체)
#    - 2-3) MinMaxScaler
#    - 2-4) RobustScaler
#    - 2-5) StandardScaler              
#    - 2-6) OneHotEncoder               
#    - 2-7) LabelEncoder                
#    - 2-8) OrdinalEncoder              
# 3) 피쳐와 타겟 입력받아 학습용과 테스트용으로 데이터 분리
# 4) 학습 데이터와 테스트 데이터 입력받아 회귀분석(릿지,라쏘,엘라스틱,선형) & 로지스틱 / KNN  !!앙상블 추가
#    - 4-1) regression_kfold_analysis(Ridge,Lasso,ElasticNet)
#    - 4-2) LogisticRegression_kfold_analysis  
#    - 4-3) LinearRegression_kfold_analysis
#    - 4-4) KNeighborsClassifier_kfold_analysis 
# 5) (4)에서 반환받은 결과 DataFrame으로 score와 loss 시각화
# 6) 모델과 테스트 데이터를 받아 예측 후 지표출력
#    -6-1) 회귀 모델 지표 출력 (r2, RMSE)
#    -6-2) 분류 모델 지표 출력 (accuracy, precision, recall, f1 등)
# 7) 사용자 입력 -> float 변환 -> DataFrame 구성 -> 예측 -> 결과 출력
# 8) DecisionTreeClassifier를 통해 시각화 (graphviz) 
# 9) predict_with_dataframe (입력 DF -> 예측값 출력)
# 10) one_click_machinelearning (통합 실행)

###############################################################################
# 1) feature와 target을 입력받아 subplot으로 시각화 + 각 그래프에 상관계수 출력
def plot_features_and_target(df, feature_cols, target_col):
    """
    df           : pandas DataFrame
    feature_cols : 시각화할 feature(컬럼)들의 리스트
    target_col   : 단일 target 컬럼 이름
    """
    nrows = math.ceil(len(feature_cols) / 3)
    ncols = 3
    X, Y = 0, 0

    # 단순 예시 로직 (사용자 정의 조건)
    if len(feature_cols)/3 < len(feature_cols)/nrows:
        X, Y = 10, 3
    elif len(feature_cols)/3 > len(feature_cols)*1.5/nrows:
        X = len(feature_cols)
        Y = len(feature_cols)**2 / 13.55

    # 서브플롯 생성
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(X, Y))

    # axes를 1차원 배열로 만듦(예: shape=(nrows*ncols,))
    # nrows*ncols가 1인 경우도 일관되게 처리하기 위해 리스트로 감싸는 로직 추가
    import numpy as np
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # feature별로 반복
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        ax.scatter(df[feature], df[target_col])
        
        # 상관계수
        corr_value = df[feature].corr(df[target_col])
        ax.set_title(f'{feature} vs {target_col} (corr={corr_value:.2f})')
        ax.set_xlabel(feature)
        ax.set_ylabel(target_col)

     # 만약 feature 개수보다 subplot 수가 많은 경우, 남는 축은 숨김
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.savefig("features_vs_target.png")
    plt.close() 


###############################################################################
# 2) 데이터 전처리
# 2-1) DataFrame을 입력받아 PowerTransformer로 데이터를 정규화 후 반환
def power_transform_dataframe(df):
    """
    df: pandas DataFrame
    """
    # PowerTransformer 객체 생성
    pt = PowerTransformer()
    
    # fit_transform 수행
    transformed_data = pt.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed

# 2-2) DataFrame을 입력받아 modified Z-score로 이상치 처리
##     modified Z-score는 이상치를 감지하는 용도지만 추가로 이상치를 중앙값으로 대체하는 기능을 넣음
def fill_outliers_with_median_modified_zscore(df, threshold=3.5):
    """
    1) 숫자형 컬럼(each numeric column)에 대해:
       - 중앙값(median) 계산
       - MAD(median absolute deviation) 계산
       - Modified Z-Score(M_i) = 0.6745 * (x_i - median) / MAD
       - abs(M_i) > threshold 인 경우 → “이상치”로 간주
       - 이상치인 해당 셀의 값을 “그 컬럼의 중앙값”으로 대체
    2) 반환: outlier가 처리된 새로운 DataFrame
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_data = df_copy[col]
        
        # 중앙값
        median_val = col_data.median()
        # MAD: median of absolute deviations
        mad_val = np.median(np.abs(col_data - median_val))
        if mad_val == 0:
            # 모든 값이 동일하거나 MAD가 0이면 이상치 식별 불가능
            continue
        
        # Modified Z-Score
        M_i = 0.6745 * (col_data - median_val) / mad_val
        
        # threshold 초과시 이상치로 간주
        outlier_mask = np.abs(M_i) > threshold
        
        # 이상치 → 해당 컬럼의 중앙값으로 대체
        df_copy.loc[outlier_mask, col] = median_val
    
    return df_copy

# 2-3) DataFrame을 입력받아 MinMaxScaler로 데이터를 정규화 후 반환
def MinMaxScaler_dataframe(df):
    """
    df: pandas DataFrame
    """
    # MinMaxScaler 객체 생성
    mm = MinMaxScaler()
    
    # fit_transform 수행
    transformed_data = mm.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed

# 2-4) DataFrame을 입력받아 RobustScaler로 데이터를 정규화 후 반환
##     modified Z-score와 다르게 이상치를 따로 처리하는 것이 아니라 이상치에 영향을 덜 받으며 데이터 정규화
def RobustScaler_dataframe(df):
    """
    df: pandas DataFrame
    """
    # RobustScaler 객체 생성
    rb = RobustScaler()
    
    # fit_transform 수행
    transformed_data = rb.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed

# 2-5) DataFrame을 입력받아 StandardScaler 로 데이터를 정규화 후 반환
def StandardScaler_dataframe(df):
    """
    df: pandas DataFrame
    """
    # StandardScaler 객체 생성
    ss = StandardScaler()
    
    # fit_transform 수행
    transformed_data = ss.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed

# 2-6) OneHotEncoder 
def OneHotEncoder_dataframe(df, columns=None):
    """
    특정 컬럼(범주형 컬럼)에 대해 OneHotEncoder를 적용하고,
    인코딩된 결과를 DataFrame으로 반환.
    
    - df        : 원본 DataFrame
    - columns   : OneHotEncoder 적용할 컬럼 리스트 (None이면 모든 object형 등 자동)
    """
    df_copy = df.copy()

    if columns is None:
        # object 타입이나 category 타입만 골라 자동 지정
        columns = df_copy.select_dtypes(include=['object','category']).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df_copy[columns])
    
    # 인코딩된 컬럼명
    encoded_cols = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_copy.index)
    
    # 기존 컬럼 제거 후 인코딩된 컬럼 합침
    df_copy.drop(columns=columns, inplace=True)
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    
    return df_copy

# 2-7) LabelEncoder 
def LabelEncoder_dataframe(df, columns=None):
    """
    여러 개 컬럼에 LabelEncoder를 적용.
    - df       : DataFrame
    - columns  : LabelEncoder 적용할 컬럼 리스트 (None이면 object/category 전체)
    
    각 컬럼에 대해 별도로 LabelEncoder를 적용하며, 신규 레이블이 나타나면 오류 발생 가능.
    """
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include=['object','category']).columns.tolist()
    
    for col in columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    return df_copy

# 2-8) OrdinalEncoder 
def OrdinalEncoder_dataframe(df, columns=None, categories='auto'):
    """
    여러 컬럼에 OrdinalEncoder를 적용.
    - df         : DataFrame
    - columns    : OrdinalEncoder 적용할 컬럼 리스트 (None이면 object/category 전체)
    - categories : 순서를 명시적으로 주려면 리스트로 전달, 기본 'auto'는 데이터 순서대로
    
    단, 명시적 카테고리 순서가 필요한 경우에는 categories에 예: [['low','medium','high'], ...] 형태
    """
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include=['object','category']).columns.tolist()
    
    encoder = OrdinalEncoder(categories=categories)
    df_copy[columns] = encoder.fit_transform(df_copy[columns])
    return df_copy


###############################################################################
# 3) feature와 target 변수를 입력받아 train/test로 분리
def split_train_test(features, target, test_size=0.2, random_state=42):
    """
    features : 2차원(여러 feature) 데이터 (DataFrame 또는 2D array)
    target   : 1차원(단일 target) 데이터 (Series 또는 1D array)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, 
        test_size=test_size, 
        random_state=random_state
    )
    return x_train, x_test, y_train, y_test


###############################################################################
# 4) 학습/테스트 -> 결과 DataFrame 반환
# 4-1) x_train, x_test, y_train, y_test를 받아서
#    Ridge, Lasso, ElasticNet 각각에 대해 KFold 학습/테스트 후 결과 DataFrame과 모델 반환(회귀)
def regression_kfold_analysis(x_train, x_test, y_train, y_test, alpha_list=None):
    """
    x_train, x_test, y_train, y_test : 학습/테스트용 데이터
    alpha_list                       : 하이퍼파라미터 alpha 후보 리스트(예: [0.01, 0.1, 1, 10])
    
    반환값: (ridge_df, lasso_df, elastic_df)
            -> 각 모델별로 train_score/test_score, train_loss/test_loss가 담긴 DataFrame
    """
    if alpha_list is None:
        alpha_list = [0.01,0.05,0.1,0.5,1,2,5,10]
    
    # 결과 저장용 DataFrame 생성
    ridge_df = pd.DataFrame(columns=['alpha','train_score','test_score','train_loss','test_loss'])
    lasso_df = pd.DataFrame(columns=['alpha','train_score','test_score','train_loss','test_loss'])
    elastic_df = pd.DataFrame(columns=['alpha','train_score','test_score','train_loss','test_loss'])
    
    # KFold 객체 생성
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Ridge
    for alpha in alpha_list:
        ridge_model = Ridge(alpha=alpha)
        
        train_scores = []
        train_losses = []
        for train_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            ridge_model.fit(X_tr, Y_tr)
            
            pred_tr = ridge_model.predict(X_tr)
            train_r2 = r2_score(Y_tr, pred_tr)
            train_mse = root_mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
        
        # cross-validation 평균
        avg_train_score = np.mean(train_scores)
        avg_train_loss = np.mean(train_losses)
        
        # 전체 train으로 다시 학습 후 test 데이터 평가
        ridge_model.fit(x_train, y_train)
        pred_test = ridge_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss = root_mean_squared_error(y_test, pred_test)
        
        # 결과 저장
        ridge_df.loc[len(ridge_df)] = [alpha, avg_train_score, test_score, avg_train_loss, test_loss]

    # Lasso
    for alpha in alpha_list:
        lasso_model = Lasso(alpha=alpha)
        
        train_scores = []
        train_losses = []
        for train_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            lasso_model.fit(X_tr, Y_tr)
            
            pred_tr = lasso_model.predict(X_tr)
            train_r2 = r2_score(Y_tr, pred_tr)
            train_mse = root_mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
        
        avg_train_score = np.mean(train_scores)
        avg_train_loss = np.mean(train_losses)
        
        lasso_model.fit(x_train, y_train)
        pred_test = lasso_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss = root_mean_squared_error(y_test, pred_test)
        
        lasso_df.loc[len(lasso_df)] = [alpha, avg_train_score, test_score, avg_train_loss, test_loss]

    # ElasticNet
    for alpha in alpha_list:
        elastic_model = ElasticNet(alpha=alpha)
        
        train_scores = []
        train_losses = []
        for train_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            elastic_model.fit(X_tr, Y_tr)
            
            pred_tr = elastic_model.predict(X_tr)
            train_r2 = r2_score(Y_tr, pred_tr)
            train_mse = root_mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
        
        avg_train_score = np.mean(train_scores)
        avg_train_loss = np.mean(train_losses)
        
        elastic_model.fit(x_train, y_train)
        pred_test = elastic_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss = root_mean_squared_error(y_test, pred_test)
        
        elastic_df.loc[len(elastic_df)] = [alpha, avg_train_score, test_score, avg_train_loss, test_loss]

    # 결과 DataFrame 출력(필요시)
    print("=== Ridge 결과 ===")
    print(ridge_df, "\n")
    print("=== Lasso 결과 ===")
    print(lasso_df, "\n")
    print("=== ElasticNet 결과 ===")
    print(elastic_df, "\n")
    
    return ridge_df, lasso_df, elastic_df,ridge_model,lasso_model,elastic_model


# 4-2) # 4-2) x_train, x_test, y_train, y_test를 받아서
#    LogisticRegression에 대해 KFold 학습/테스트 후 결과 DataFrame과 모델 반환(분류)

def LogisticRegression_kfold_analysis(x_train, x_test, y_train, y_test, alpha_list=None):
    """
    Logistic Regression 교차검증 분석
    alpha_list는 규제 강도( alpha ) 리스트로, 실제 LogisticRegression에선 C=1/alpha 사용
    반환: (logistic_df, final_logistic_model)
    """
    if alpha_list is None:
        alpha_list = [0.01,0.05,0.1,0.5,1,2,5,10]
    
    # DataFrame: alpha, train_score, test_score, train_loss, test_loss
    # -> train_score / test_score : 분류정확도
    # -> train_loss / test_loss   : log_loss
    logistic_df = pd.DataFrame(columns=['alpha','train_score','test_score','train_loss','test_loss'])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    

    for alpha in alpha_list:
        # alpha -> C=1/alpha
        c_val = 1.0 / alpha if alpha != 0 else 1e6
        
        logistic_model = LogisticRegression(C=c_val, solver='lbfgs', max_iter=1000)

        acc_list = []
        loss_list = []

        for tr_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            
            logistic_model.fit(X_tr, Y_tr)
            
            # train accuracy
            train_acc = logistic_model.score(X_tr, Y_tr)
            # train log_loss
            pred_proba_tr = logistic_model.predict_proba(X_tr)
            train_ll = log_loss(Y_tr, pred_proba_tr)

            acc_list.append(train_acc)
            loss_list.append(train_ll)

        avg_train_acc  = np.mean(acc_list)
        avg_train_loss = np.mean(loss_list)

        # 최종 학습 & test 평가는 아래에서
        logistic_model.fit(x_train, y_train)
        test_acc = logistic_model.score(x_test, y_test)
        pred_proba_test = logistic_model.predict_proba(x_test)
        test_ll = log_loss(y_test, pred_proba_test)

        # 결과 저장
        logistic_df.loc[len(logistic_df)] = [
            alpha,
            avg_train_acc,
            test_acc,
            avg_train_loss,
            test_ll
        ]

        

    print("=== LogisticRegression 결과 ===")
    print(logistic_df, "\n")

    return logistic_df, logistic_model 

## 4-3) x_train, x_test, y_train, y_test를 받아서
#    LinearRegression에 대해 KFold 학습/테스트 후 결과 DataFrame과 모델 반환(회귀)

def LinearRegression_kfold_analysis(x_train, x_test, y_train, y_test):

    alpha_list=[0,0]
    
    # 결과 저장용 DataFrame 생성
    linear_df = pd.DataFrame(columns=['alpha','train_score','test_score','train_loss','test_loss'])
    
    # KFold 객체 생성
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    
    # linear regression
    for alpha in alpha_list:
        # alpha는 의미 없으므로 그냥 무시
        linear_model = LinearRegression()
        
        train_scores = []
        train_losses = []
        for train_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            linear_model.fit(X_tr, Y_tr)
            pred_tr = linear_model.predict(X_tr)
            
            train_r2 = r2_score(Y_tr, pred_tr)
            train_mse = root_mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
            
        # cross-validation 평균
        avg_train_score = np.mean(train_scores)
        avg_train_loss  = np.mean(train_losses)
        
        # 전체 train으로 다시 학습 후 test 데이터 평가
        linear_model.fit(x_train, y_train)
        pred_test = linear_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss  = root_mean_squared_error(y_test, pred_test)
       
        # 결과 저장
        linear_df.loc[len(linear_df)] = [alpha, avg_train_score, test_score, avg_train_loss, test_loss]
        
    # 결과 DataFrame 출력(필요시)
    print("=== LinearRegression 결과 ===")
    print(linear_df, "\n")
    return linear_df, linear_model

## 4-4) x_train, x_test, y_train, y_test를 받아서
#    KNeighborsClassifier에 대해 KFold 학습/테스트 후 결과 DataFrame과 모델 반환(분류류)

def KNeighborsClassifier_kfold_analysis(x_train, x_test, y_train, y_test, k_list=None):
    """
    KFold 기반 KNN 분류
    k_list: [3,5,7,9] 등
    결과: (knn_df, knn_model)
    """
    if k_list is None:
        k_list = [1, 3, 5, 7, 9]
        
     # 결과 저장용 DataFrame 생성
    knn_df = pd.DataFrame(columns=['n_neighbors','train_score','test_score','train_loss','test_loss'])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # kneighborsclassifier
    for k in k_list:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        
        acc_list = []
        loss_list = []
        for tr_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            
            knn_model.fit(X_tr, Y_tr)
            train_acc = knn_model.score(X_tr, Y_tr)
            pred_proba_tr = knn_model.predict_proba(X_tr)
            train_ll = log_loss(Y_tr, pred_proba_tr)  # log_loss

            acc_list.append(train_acc)
            loss_list.append(train_ll)

        # cross-validation 평균
        avg_train_acc  = np.mean(acc_list)
        avg_train_loss = np.mean(loss_list)

        # 전체 train으로 다시 학습 후 test 데이터 평가
        knn_model.fit(x_train, y_train)
        test_acc = knn_model.score(x_test, y_test)
        test_ll = log_loss(y_test, knn_model.predict_proba(x_test))

        # 결과 저장
        knn_df.loc[len(knn_df)] = [k, avg_train_acc, test_acc, avg_train_loss, test_ll]
        
    # 결과 DataFrame 출력(필요시)
    print("=== KNeighborsClassifier 결과 ===")
    print(knn_df, "\n")
    return knn_df, knn_model


###############################################################################
# 5) DataFrame을 입력받아 x축=alpha(또는 k), 첫 그래프=(train_score, test_score),
#                      두 번째=(train_loss, test_loss) 시각화
def plot_regression_results(df):
    """
    df: 위에서 만든 result DataFrame (예: ridge_df, lasso_df, elastic_df 등)
    """
    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))
    
    axes[0].plot(df.iloc[:,0], df['train_score'],'o-', label='train_score')
    axes[0].plot(df.iloc[:,0], df['test_score'], 'o-', label='test_score')
    axes[0].set_xlabel(df.columns[0])  # alpha or n_neighbors 등
    axes[0].set_ylabel('score')
    axes[0].legend()
    axes[0].set_title('Score')
    
    axes[1].plot(df.iloc[:,0], df['train_loss'], 'o-', label='train_loss')
    axes[1].plot(df.iloc[:,0], df['test_loss'], 'o-', label='test_loss')
    axes[1].set_xlabel(df.columns[0])
    axes[1].set_ylabel('loss')
    axes[1].legend()
    axes[1].set_title('Loss')
    
    plt.tight_layout()
    plt.show()


###############################################################################
# 6-1) 모델과 테스트 데이터를 받아 예측 후 r2_score와 RMSE를 계산·출력(회귀용)
def predict_and_evaluate_model(model, X_test, y_test):
    """
    model  : 학습된 모델 (예: Ridge, Lasso, ElasticNet, etc.)
    X_test : 테스트용 feature 데이터
    y_test : 테스트용 target 데이터
    
    반환값: (score, rmse)
            score = r2_score
            rmse  = Root Mean Squared Error
    회귀 모델 전용. 분류 모델이면 사용 X
    """
    from sklearn.metrics import root_mean_squared_error
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"R2 Score: {score:.4f}, RMSE: {rmse:.4f}")
    return score, rmse

# 6-2) 분류 모델 지표 (Accuracy, Precision, Recall, F1)
def classification_evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='macro', zero_division=0)
    rec = recall_score(y_test, pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, pred, average='macro', zero_division=0)
    print("[분류평가]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision(macro): {prec:.4f}")
    print(f"Recall(macro): {rec:.4f}")
    print(f"F1(macro): {f1:.4f}")
    return acc, prec, rec, f1

###############################################################################
# 7) 사용자 입력 -> float 변환 -> DataFrame 구성 -> 예측 -> 결과 출력
def predict_with_user_input(model, feature_cols, df):
    """
    model        : 학습 완료된 모델 (Ridge, Lasso, ElasticNet, etc.)
    feature_cols : 모델 학습에 사용한 feature 컬럼명 리스트
    df           : 원본 데이터프레임 (또는 feature_cols를 포함한 데이터프레임)
                   -> 각 컬럼의 중앙값 계산용
    
    동작 요약:
    1) df에서 feature_cols의 중앙값 리스트를 구함
    2) 예: 3,4,5 형태로 표시하여 입력 예시를 보여줌
    3) 사용자의 실제 입력값을 받아 DataFrame 생성 후 예측
    """
    print('[예측값 출력]')

    medians = df[feature_cols].median()
    example_str = ",".join([f"{medians[col]:.3g}" for col in feature_cols])
    
    user_input_str = input(f'입력 (예: {example_str}) : ').split(',')
    new_data = [float(data.strip()) for data in user_input_str]
    print(new_data)  # 입력값 확인

    dataDF = pd.DataFrame([new_data], columns=feature_cols)
    prediction = model.predict(dataDF)
    
    print("예측값:", prediction[0])
    return prediction


###############################################################################
# 8-1) graphviz + DecisionTreeClassifier + export_graphviz 
def decision_tree_graphviz(train_x, train_y, max_depth=3, feature_names=None, class_names=None):
    """
    (스케일된) train_x와 (스케일 안된) train_y를 입력받아
    DecisionTreeClassifier 객체 학습 후 graphviz로 시각화
    
    - train_x      : 독립변수 DataFrame (전처리, 스케일링 등 완료된 상태)
    - train_y      : 종속변수 Series/ndarray (스케일X, 범주형이라면 문자열 등)
    - max_depth    : 트리 깊이 제한
    - feature_names: 각 feature 이름 리스트(트리 시각화 라벨용)
    - class_names  : 분류 문제일 때 클래스 이름 리스트
    
    사용 예시:
      decision_tree_graphviz(scaled_X, y, max_depth=4,
                             feature_names=X.columns,
                             class_names=['setosa','versicolor','virginica'])
    """
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_model.fit(train_x, train_y)

    # dot_data 생성
    dot_data = export_graphviz(
        dt_model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    # graphviz 시각화
    graph = graphviz.Source(dot_data)
    return dt_model, graph

# 8-1) RandomForestClassifier KFold
def random_forest_kfold_analysis(x_train, x_test, y_train, y_test, n_estimators_list=None):
    """
    분류용 RandomForest, KFold CV로 accuracy/log_loss 확인
    """
    if n_estimators_list is None:
        n_estimators_list = [50, 100, 200]

    rf_df = pd.DataFrame(columns=['n_estimators','train_score','test_score','train_loss','test_loss'])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_rf = None

    for n_estimators in n_estimators_list:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        acc_list, loss_list = [], []
        for tr_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            rf_model.fit(X_tr, Y_tr)
            train_acc = rf_model.score(X_tr, Y_tr)
            pred_proba_tr = rf_model.predict_proba(X_tr)
            train_ll = log_loss(Y_tr, pred_proba_tr)
            acc_list.append(train_acc)
            loss_list.append(train_ll)

        avg_acc = np.mean(acc_list)
        avg_loss= np.mean(loss_list)

        rf_model.fit(x_train, y_train)
        test_acc = rf_model.score(x_test, y_test)
        test_ll  = log_loss(y_test, rf_model.predict_proba(x_test))

        rf_df.loc[len(rf_df)] = [n_estimators, avg_acc, test_acc, avg_loss, test_ll]
        final_rf = rf_model

    print("=== RandomForest 결과 ===")
    print(rf_df, "\n")
    return rf_df, final_rf

# 8-2) GradientBoostingClassifier KFold
def gradientboost_kfold_analysis(x_train, x_test, y_train, y_test, n_estimators_list=None):
    if n_estimators_list is None:
        n_estimators_list = [50, 100, 200]

    gb_df = pd.DataFrame(columns=['n_estimators','train_score','test_score','train_loss','test_loss'])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_gb = None

    for n_estimators in n_estimators_list:
        gb_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        acc_list, loss_list = [], []
        for tr_idx, val_idx in kf.split(x_train):
            X_tr, X_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
            Y_tr, Y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx] 
            # Correction:
            Y_tr, Y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            gb_model.fit(X_tr, Y_tr)
            train_acc = gb_model.score(X_tr, Y_tr)
            train_ll  = log_loss(Y_tr, gb_model.predict_proba(X_tr))
            acc_list.append(train_acc)
            loss_list.append(train_ll)

        avg_acc  = np.mean(acc_list)
        avg_loss = np.mean(loss_list)

        gb_model.fit(x_train, y_train)
        test_acc = gb_model.score(x_test, y_test)
        test_ll  = log_loss(y_test, gb_model.predict_proba(x_test))

        gb_df.loc[len(gb_df)] = [n_estimators, avg_acc, test_acc, avg_loss, test_ll]
        final_gb = gb_model

    print("=== GradientBoosting 결과 ===")
    print(gb_df, "\n")
    return gb_df, final_gb

###############################################################################
# 9) “유저 입력” 대신, 이미 구성된 (피처만 있는) DataFrame을 입력받아 예측값 출력
def predict_with_dataframe(model, feature_df):
    """
    feature_df : 이미 모델에 맞는 피처 열을 가지고 있는 DF
    -> model.predict(feature_df) 후, 각 행에 대한 예측 결과를 print
    """
    preds = model.predict(feature_df)
    print("[Predict With DataFrame]")
    for i, p in enumerate(preds):
        print(f"Row {i} => {p}")
    return preds


###############################################################################
# 10) one_click_machinelearning
def one_click_machinelearning(df, feature_cols, target_col,feature_df_for_pre):
    """
    feature_df_for_pre : 예측을 위해 사용할 학습에 사용하지 않은 피쳐df
    1) plot_features_and_target(...) + 산점도 저장
    2) 전처리 단계 -> 2)항목 함수 중 하나를 선택
       - 1) PowerTransformer
       - 2) Modified Z-score
       - 3) MinMaxScaler
       - 4) RobustScaler
       - 5) StandardScaler
       - 6) OneHotEncoder
       - 7) LabelEncoder
       - 8) OrdinalEncoder
       - 9) 아무것도 안 하고 다음 단계
       - 선택 후 0이면 다시 선택, 1이면 적용 -> 다시 전처리 단계로 돌아가거나(계속 전처리 가능),
         최종 9를 고르면 전처리 종료
    3) split_train_test
    4) 모델 선택
       - 1 입력 => 4)항목 함수 중 하나(1~4)
         (1: regression_kfold_analysis, 2:Logistic, 3:Linear, 4:KNN 등)
       - 2 입력 => 8)항목 앙상블 함수(#8-1,#8-2) 중 하나
         (1: random_forest_kfold_analysis, 2: gradientboost_kfold_analysis)
       - 모델 선택시 0이면 재선택, 1이면 실행
    5) 4단계 반환 DF -> plot_regression_results
    6) 1=> 회귀지표, 2=> 분류지표
    7) predict_with_dataframe (피처 값이 들어있는 DF 입력받아 예측)
    """
    # 1) 산점도 그래프
    plot_features_and_target(df, feature_cols, target_col)
    print("[1단계] 산점도 그래프(features_vs_target.png) 저장 완료")

    # 준비
    working_df = df.copy()

    # 2) 전처리 단계
    while True:
        print("\n[2단계: 데이터 전처리]")
        print("1) PowerTransformer")
        print("2) ModifiedZ-score(이상치 대체)")
        print("3) MinMaxScaler")
        print("4) RobustScaler")
        print("5) StandardScaler")
        print("6) OneHotEncoder")
        print("7) LabelEncoder")
        print("8) OrdinalEncoder")
        print("9) 전처리하지 않고 다음 단계로 이동")
        choice = input("-> 전처리 번호 선택: ")

        if choice not in [str(i) for i in range(1,10)]:
            print(" 잘못된 입력입니다.")
            continue

        choice = int(choice)
        if choice == 9:
            # 전처리 안함
            print(" 전처리를 종료하고 다음 단계로 진행합니다.")
            break
        # 해당 함수 설명 + 적용 여부(0=재선택, 1=적용)
        if choice == 1:
            print("[PowerTransformer] : Yeo-Johnson 방식 등으로 데이터 분포 변환")
        elif choice == 2:
            print("[ModifiedZ-score] : 이상치(M_i>threshold)를 컬럼 중앙값으로 대체")
        elif choice == 3:
            print("[MinMaxScaler] : 모든 컬럼을 0~1 범위로 스케일")
        elif choice == 4:
            print("[RobustScaler] : 중앙값/사분위수 기반 스케일 (이상치 영향 적음)")
        elif choice == 5:
            print("[StandardScaler] : 평균0, 표준편차1로 스케일")
        elif choice == 6:
            print("[OneHotEncoder] : 범주형 -> 여러 열(더미변수)")
        elif choice == 7:
            print("[LabelEncoder] : 범주 -> 숫자 라벨")
        elif choice == 8:
            print("[OrdinalEncoder] : 범주 -> 순서화 숫자")

        confirm = input("이 함수를 적용하려면 1, 아니면 0을 입력하세요: ")
        if confirm == '0':
            print(" 전처리 함수를 다시 선택합니다.")
            continue
        else:
            # 실제 적용
            if choice == 1:
                numeric_cols = working_df.select_dtypes(include=[np.number]).columns
                working_df[numeric_cols] = power_transform_dataframe(working_df[numeric_cols])
            elif choice == 2:
                working_df = fill_outliers_with_median_modified_zscore(working_df)
            elif choice == 3:
                numeric_cols = working_df.select_dtypes(include=[np.number]).columns
                working_df[numeric_cols] = MinMaxScaler_dataframe(working_df[numeric_cols])
            elif choice == 4:
                numeric_cols = working_df.select_dtypes(include=[np.number]).columns
                working_df[numeric_cols] = RobustScaler_dataframe(working_df[numeric_cols])
            elif choice == 5:
                numeric_cols = working_df.select_dtypes(include=[np.number]).columns
                working_df[numeric_cols] = StandardScaler_dataframe(working_df[numeric_cols])
            elif choice == 6:
                working_df = OneHotEncoder_dataframe(working_df)
            elif choice == 7:
                working_df = LabelEncoder_dataframe(working_df)
            elif choice == 8:
                working_df = OrdinalEncoder_dataframe(working_df)
            
            # 전처리 적용 후 다시 전처리 반복할지, 9번 눌러서 종료할지
            print(" 전처리가 적용되었습니다. 계속 전처리를 진행할 수 있습니다. (9를 누르면 종료)")

    # 3) split train test
    X = working_df[feature_cols]
    y = working_df[target_col]
    x_train, x_test, y_train, y_test = split_train_test(X, y)
    print("[3단계] 데이터 분할 완료")

    # 4) 모델 선택
    while True:
        print("\n[4단계: 모델 선택]")
        print("1. (4)항목의 모델들 (1: regression, 2: logistic, 3: linear, 4:knn)")
        print("2. (8)항목의 앙상블 함수들 (1: random_forest, 2: gradient_boost)")
        sel_main = input("-> 1 or 2? ")

        if sel_main == '1':
            # 4)항목 선택
            while True:
                print(" [모델 학습] 1: regression_kfold, 2: logistic_kfold, 3: linear_kfold, 4: knn_kfold")
                sub_choice = input(" 모델 번호 선택: ")
                if sub_choice not in ['1','2','3','4']:
                    print(" 잘못된 입력")
                    continue
                # 설명
                if sub_choice=='1':
                    print("[regression_kfold_analysis] Ridge, Lasso, ElasticNet")
                elif sub_choice=='2':
                    print("[LogisticRegression_kfold_analysis] 로지스틱 분류")
                elif sub_choice=='3':
                    print("[LinearRegression_kfold_analysis] 단순 선형회귀")
                else:
                    print("[KNeighborsClassifier_kfold_analysis] KNN 분류")

                cfm = input("이 함수를 실행하려면 1, 취소->재선택 0: ")
                if cfm=='0':
                    continue
                else:
                    # 실행
                    if sub_choice=='1':
                        ridge_df, lasso_df, elastic_df, final_ridge, final_lasso, final_elastic = regression_kfold_analysis(x_train, x_test, y_train, y_test)
                        # 임의로 ridge_df 반환
                        result_df = ridge_df  # 시각화용
                        final_model = final_ridge
                    elif sub_choice=='2':
                        logistic_df, final_model = LogisticRegression_kfold_analysis(x_train, x_test, y_train, y_test)
                        result_df = logistic_df
                    elif sub_choice=='3':
                        linear_df, final_model = LinearRegression_kfold_analysis(x_train, x_test, y_train, y_test)
                        result_df = linear_df
                    else:
                        knn_df, final_model = KNeighborsClassifier_kfold_analysis(x_train, x_test, y_train, y_test)
                        result_df = knn_df
                    break
            # 모델 결정되었으니 탈출
            break

        elif sel_main == '2':
            # 8)항목 앙상블
            while True:
                print(" [앙상블 선택] 1: random_forest_kfold, 2: gradientboost_kfold")
                sub_choice = input("-> ")
                if sub_choice not in ['1','2']:
                    print(" 잘못된 입력")
                    continue

                if sub_choice=='1':
                    print("[RandomForestClassifier] 교차검증")
                else:
                    print("[GradientBoostingClassifier] 교차검증")

                cfm = input("이 함수를 실행하려면 1, 취소->재선택 0: ")
                if cfm=='0':
                    continue
                else:
                    if sub_choice=='1':
                        rf_df, final_model = random_forest_kfold_analysis(x_train, x_test, y_train, y_test)
                        result_df = rf_df
                    else:
                        gb_df, final_model = gradientboost_kfold_analysis(x_train, x_test, y_train, y_test)
                        result_df = gb_df
                    break
            break
        else:
            print(" 잘못된 입력")

    # 5) 결과 DF 시각화
    plot_regression_results(result_df)
    print("[5단계] 결과 DF 그래프 시각화 완료")

    # 6) 지표 출력
    while True:
        print("\n[6단계: 지표 출력]")
        print("1. 회귀 지표 (r2, RMSE)")
        print("2. 분류 지표 (accuracy, precision, recall, f1)")
        c = input("-> ")
        if c not in ['1','2']:
            print(" 잘못된 입력")
            continue
        if c=='1':
            predict_and_evaluate_model(final_model, x_test, y_test)
        else:
            classification_evaluate_model(final_model, x_test, y_test)
        break

    # 7) predict_with_dataframe 유사 기능
    print("\n[7단계: DataFrame을 입력받아 예측]")
    preds = predict_with_dataframe(final_model, feature_df_for_pre)
    print("예측 값:\n",  preds)
    print("\n=== 완료 ===")
    
    
    
    
    
    

# 2. 이미지 전처리 & 모델 학습


# 함수목차

# 1) 이미지 로드와 전처리, 증강
# 2) 이진화된 이미지 -> Feature 추출
# 3) 폴더 단위 -> DF 생성 (+회전/뒤집기 옵션) - 매개변수 잘 확인하고 추가할것
# 4) 여러 DF 결합
# 5) 이미지 분류 모델 학습 - 매개변수 및 주석 확인
# 6) 모델 평가
# 7) one_click_imagetraining
# 8) 이미지 입력받아 예측
########################################
# 1) 유틸성 함수: 이미지 읽기, 전처리, 증강
########################################

def load_image(train_filepath):
    """
    train_filepath 경로의 이미지를 OpenCV로 읽어오기 (BGR 형태).
    """
    img = cv2.imread(train_filepath)
    if img is None:
        raise FileNotFoundError(f"[load_image] '{train_filepath}' 를 불러올 수 없습니다.")
    return img

def convert_to_gray(img_bgr):
    """
    BGR 이미지를 Grayscale로 변환.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize_image(img, width=64, height=64):
    """
    이미지를 (width x height)로 리사이즈.
    """
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def threshold_image(img_gray, thresh_val=127):
    """
    명시적 thresh_val로 이진화
    """
    _, bin_img = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)
    return bin_img

def auto_threshold_image(img_gray):
    """
    Otsu 방식을 사용해 이진화 (자동 임계값 결정).
    """
    val, bin_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bin_img, val

def rotate_image(img, angle=90):
    """
    회전 각도(angle)에 따라 이미지를 회전
    angle = 90, 180, 270 등
    """
    # 이미지 중심 구하기
    h, w = img.shape[:2]
    center = (w/2, h/2)
    # 회전 행렬 계산
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 회전 적용
    rotated = cv2.warpAffine(img, mat, (w, h))
    return rotated

def flip_image(img, flip_code=1):
    """
    flip_code=1 : 좌우 뒤집기
    flip_code=0 : 상하 뒤집기
    flip_code=-1: 상하좌우 뒤집기
    """
    flipped = cv2.flip(img, flip_code)
    return flipped


########################################
# 2) 이진화된 이미지 -> Feature 추출
########################################

def extract_features(img_bin):
    """
    이진화 이미지(bin_img)에서
    mean, std, white_ratio(255비율) 추출.
    """
    arr = img_bin.flatten()
    mean_val = np.mean(arr)
    std_val  = np.std(arr)
    white_ratio = np.count_nonzero(arr == 255) / len(arr)
    return {
        "mean": mean_val,
        "std":  std_val,
        "white_ratio": white_ratio
    }


########################################
# 3) 폴더 단위 -> DF 생성 (+회전/뒤집기 옵션) - 이미지 확장자 다를 시 코드 수정
########################################

def build_dataset_from_images(
    folder_path,      # 이미지 폴더 경로
    label_name,       # 타겟 라벨
    width=64, height=64,
    use_auto_thresh=False,
    do_rotation=False,
    do_flip=False
):
    """
    폴더 내의 모든 .jpg/.png 이미지를:
    1) 로드 -> Grayscale -> resize(64x64)
    2) auto_thresh? -> Otsu로 이진화 or 수동threshold=127
    3) Feature 추출(mean,std,white_ratio)
    4) 'target' = label_name
    5) do_rotation=True면, 90/180/270도 회전 이미지도 추가
    6) do_flip=True면, 좌우/상하 뒤집기도 추가
    반환: DataFrame (각 row가 1개 이미지(증강된 것도 각각 1row))
    """
    rows = []
    all_files = sorted(os.listdir(folder_path))

    for fn in all_files:
        if not(fn.lower().endswith('.jpg') or fn.lower().endswith('.png')):
            continue
        train_filepath = os.path.join(folder_path, fn)
        # 1) 로드/전처리
        bin_img = load_image(train_filepath)
        # bin_img = convert_to_gray(bin_img)
        # bin_img = resize_image(bin_img, width, height)

        # if use_auto_thresh:
        #     bin_img, otsu_val = auto_threshold_image(img_resized)
        # else:
        #     bin_img = threshold_image(img_resized, 127)

        # 2) feature
        base_feats = extract_features(bin_img)
        base_feats['filename'] = fn
        base_feats['target']   = label_name
        # rows에 추가
        rows.append(base_feats)

        # (회전)
        if do_rotation:
            for angle in [90, 180, 270]:
                rot_img = rotate_image(img_resized, angle)
                # 이진화
                if use_auto_thresh:
                    bin2, _ = auto_threshold_image(rot_img)
                else:
                    bin2 = threshold_image(rot_img, 127)

                feats2 = extract_features(bin2)
                feats2['filename'] = f"{fn}_rot{angle}"
                feats2['target']   = label_name
                rows.append(feats2)

        # (뒤집기)
        if do_flip:
            # 좌우, 상하, 상하좌우 총 3개 추가
            # 필요 시 원하는 것만
            for flip_code in [1, 0, -1]:
                flip_img = flip_image(img_resized, flip_code=flip_code)
                if use_auto_thresh:
                    bin3, _ = auto_threshold_image(flip_img)
                else:
                    bin3 = threshold_image(flip_img, 127)

                feats3 = extract_features(bin3)
                feats3['filename'] = f"{fn}_flip{flip_code}"
                feats3['target']   = label_name
                rows.append(feats3)

    df = pd.DataFrame(rows)
    return df

#############################
# 4) 여러 DF 결합
#############################

def combine_image_datasets(dfs_list):
    """
    여러 DataFrame을 하나로 concat.
    """
    combined_df = pd.concat(dfs_list, ignore_index=True)
    return combined_df

#############################
# 5) 이미지 분류 모델 학습 - 매개변수 및 주석 확인
#############################

def train_image_classifier(df, feature_cols, target_col, model_name="logistic"):
    """
    df : 이미지 feature DataFrame
    feature_cols : 예 ['mean','std','white_ratio']
    target_col   : 'target'
    model_name   : 'logistic','svc','knn','random_forest','gboost' 등

    반환: (model)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    X = df[feature_cols].values
    y = df[target_col].values

    model_name = model_name.lower()
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'svc':
        model = SVC(probability=True, kernel='rbf', gamma='scale')
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'gboost':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        print(f"[train_image_classifier] 알 수 없는 모델명 '{model_name}', logistic으로 진행")
        model = LogisticRegression(max_iter=1000)

    model.fit(X, y)
    return model


#############################
# 6) 모델 평가
#############################

def evaluate_image_classifier(model, df, feature_cols, target_col):
    """
    df: 이미지 DataFrame
    feature_cols: 예 ['mean','std','white_ratio']
    target_col: 'target'

    - Accuracy, Precision(macro), Recall(macro), F1(macro) 출력
    """
    X = df[feature_cols].values
    y_true = df[target_col].values
    y_pred = model.predict(X)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("[이미지 분류 모델 평가]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision(macro): {prec:.4f}")
    print(f"Recall(macro): {rec:.4f}")
    print(f"F1(macro): {f1:.4f}")




#############################
# 7) one_click_imagetraining
#############################

def one_click_imagetraining():
    """
    1) 사용자에게 '몇 개 폴더?'를 물어보고,
    2) 각 폴더 경로 + 라벨 + (resize, auto thresh, rotation, flip) 여부 입력,
    3) build_dataset_from_images 로 DF 생성 (증강 포함),
    4) 여러 DF를 combine,
    5) 최종 CSV로 저장
    6) (원한다면) 이미지 분류 모델 학습 -> 평가

    * rotation/flip 은 사용자가 y/n 입력, y면 적용
    * csv파일 이름도 사용자에게 입력받아 저장
    * 모델 학습도 y/n
      - y -> 모델 선택(1~5) -> train -> evaluate
    """
    print("\n[one_click_imagetraining] 여러 폴더 이미지 -> 하나의 DF -> CSV 저장 / 모델학습")

    # 1) 폴더 개수
    n_str = input("몇 개 폴더를 사용하시겠습니까? (기본=1): ")
    try:
        n_folders = int(n_str.strip())
    except:
        n_folders = 1

    all_dfs = []
    # 2) 각 폴더 처리
    for i in range(n_folders):
        print(f"\n[{i+1}/{n_folders}]번째 폴더 정보를 입력하세요.")
        folder_path = input("폴더 경로: ")
        label_name  = input("라벨명(예: 개, 고양이): ")
        w_str = input("리사이즈 width(기본=64): ")
        h_str = input("리사이즈 height(기본=64): ")
        auto_str = input("자동 임계값(Otsu)? (y/n,기본=y): ")
        rot_str  = input("회전 증강 추가?(y/n,기본=n): ")
        flip_str = input("뒤집기 증강 추가?(y/n,기본=n): ")

        try:
            w_val = int(w_str.strip())
        except:
            w_val = 64
        try:
            h_val = int(h_str.strip())
        except:
            h_val = 64

        use_auto_thresh = (auto_str.strip().lower() != 'n')  # 기본 y
        do_rotation     = (rot_str.strip().lower() == 'y')
        do_flip         = (flip_str.strip().lower() == 'y')

        # 3) DF 생성
        df_img = build_dataset_from_images(
            folder_path, label_name,
            width=w_val, height=h_val,
            use_auto_thresh=use_auto_thresh,
            do_rotation=do_rotation,
            do_flip=do_flip
        )
        print(f"[{i+1}] => {len(df_img)}개 이미지(row)가 생성됨 shape={df_img.shape}")
        all_dfs.append(df_img)

    # 4) DF 결합
    final_df = combine_image_datasets(all_dfs)
    print(f"\n[combine_image_datasets] 총 {len(final_df)}개 row. shape={final_df.shape}")

    # 5) CSV 저장
    save_name = input("CSV로 저장할 파일 이름(예: myimages.csv, 기본=myimages.csv): ")
    if not save_name.strip():
        save_name = "myimages.csv"

    final_df.to_csv(save_name, index=False, encoding='utf-8-sig')
    print(f"[CSV 저장] {save_name} (shape={final_df.shape})")

    # 6) 모델 학습 의사
    learn_str = input("\n이미지 모델 학습하시겠습니까? (y/n, 기본=n): ")
    if learn_str.strip().lower() == 'y':
        # 모델 선택
        print("\n[사용 가능한 모델]\n1: logistic\n2: svc\n3: knn\n4: random_forest\n5: gboost")
        m_str = input("모델 번호 선택(기본=1): ").strip()
        model_map = {
            '1':'logistic','2':'svc','3':'knn','4':'random_forest','5':'gboost'
        }
        chosen_model = model_map.get(m_str, 'logistic')
        print(f" -> '{chosen_model}' 모델을 사용합니다.")

        # 학습
        feature_cols = ['mean','std','white_ratio']  # 가정
        model = train_image_classifier(final_df, feature_cols, 'target', chosen_model)
        print("[모델 학습 완료]")

        # 평가
        evaluate_image_classifier(model, final_df, feature_cols, 'target')
        print("[모델 평가 완료]")

        print("\none_click_imagetraining 완료. (model, final_df) 반환합니다.")
        return model, final_df

    else:
        print("모델 학습 없이 종료합니다.")
        return None, final_df
    

#############################
# 8) 이미지 입력받아 예측  - 이미지 확장자 다를 시 코드 수정
#############################
    
def predict_with_user_imagefolder(model, feature_cols=['mean','std','white_ratio'], width=64, height=64, use_auto_thresh=False):
    """
    사용자에게 이미지 폴더 경로를 입력받아,
    1) 폴더 내 .jpg/.png 이미지를 전처리(Gray->Resize->이진화) 후,
    2) (mean, std, white_ratio) 등의 feature를 추출,
    3) model.predict(...)를 실행하여 예측값을 print.

    model       : 이미 학습된 분류/회귀 모델
    feature_cols: 모델이 기대하는 피처 목록 (예: ['mean','std','white_ratio'])
    width,height: 리사이즈 크기 (기본 64x64)
    use_auto_thresh: True면 Otsu 자동 이진화, False면 임계값=127로 이진화
    """
    folder_path = input("예측할 이미지가 들어있는 폴더 경로를 입력하세요: ")
    if not os.path.isdir(folder_path):
        print(f"[오류] 폴더 경로가 올바르지 않습니다: {folder_path}")
        return

    # 폴더 내 이미지를 읽어와 피처 추출
    rows = []
    all_files = sorted(os.listdir(folder_path))
    for fn in all_files:
        if not(fn.lower().endswith('.jpg') or fn.lower().endswith('.png')):
            continue
        train_filepath = os.path.join(folder_path, fn)
        print(f"[로딩] {train_filepath}")
        img_bgr = cv2.imread(train_filepath)
        if img_bgr is None:
            print(f"[오류] 이미지를 불러올 수 없음: {train_filepath}")
            continue
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # img_resize = cv2.resize(img_gray, (width, height), interpolation=cv2.INTER_AREA)

        # if use_auto_thresh:
        #     _, bin_img = cv2.threshold(img_resize, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # else:
        #     _, bin_img = cv2.threshold(img_resize, 127, 255, cv2.THRESH_BINARY)

        # mean, std, white_ratio 추출
        arr = img_bgr.flatten()
        mean_val = np.mean(arr)
        std_val  = np.std(arr)
        white_ratio = np.count_nonzero(arr==255) / len(arr)

        row_dict = {
            'filename': fn,
            'mean': mean_val,
            'std' : std_val,
            'white_ratio': white_ratio
        }
        rows.append(row_dict)

    if not rows:
        print(f"[결과] 폴더 내 이미지가 없거나 처리할 수 없습니다: {folder_path}")
        return

    # DataFrame 생성
    df_pred = pd.DataFrame(rows)
    print(f"[전처리 완료] 총 {len(df_pred)}장 이미지의 feature 생성됨.")
    
    # 모델에 넣기
    if not set(feature_cols).issubset(df_pred.columns):
        print("[오류] feature_cols가 df_pred에 없습니다. 실제 컬럼:", df_pred.columns.tolist())
        return

    X_pred = df_pred[feature_cols].values
    predictions = model.predict(X_pred)

    # 예측값 출력
    print("\n=== 예측 결과 ===")
    for i, p in enumerate(predictions):
        fn = df_pred.loc[i, 'filename']
        print(f"{fn} => {p}")
    return df_pred, predictions

def train_all_models_from_images_color(
    train_csv_path,         # 학습용 CSV (열: [img_path, 과])
    test_csv_path,          # 테스트용 CSV (열: [img_path, 과])
    target_label='과',      # 타겟 열 (기본= '과')
    n_splits=5,            # KFold 분할 개수
    bins=(8,8,8),          # HSV 컬러 히스토그램 bin 크기 (H,S,V)
    max_iter_logistic=2000 # LogisticRegression 반복 횟수
):
    """
    1) 학습 CSV와 테스트 CSV에서 [img_path, 과] 열을 읽어온다.
    2) 각 이미지(컬러)를 HSV 변환 -> 3D 컬러 히스토그램(8x8x8=512차원) -> flatten -> feature
    3) train_df, test_df 생성 후, 
       - train_df에 대해 KFold(n_splits)로 교차검증 (모델별 Accuracy, F1)
       - train_test_split(hold-out)으로 최종 모델 학습 + 검증
       - test_df로 최종 평가 -> 가장 좋은 모델(F1기준) 찾기
    4) 모든 모델을 dict로 반환
    """

    import pandas as pd
    import numpy as np
    import cv2
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    # 분류 모델 5종
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    ############################
    # (A) CSV 로드
    ############################
    df_train_csv = pd.read_csv(train_csv_path)
    df_test_csv  = pd.read_csv(test_csv_path)

    for col in ['img_path', target_label]:
        if col not in df_train_csv.columns:
            raise KeyError(f"[오류] 학습 CSV에 '{col}' 열이 없습니다.")
        if col not in df_test_csv.columns:
            raise KeyError(f"[오류] 테스트 CSV에 '{col}' 열이 없습니다.")

    ############################
    # (B) 이미지 -> HSV 컬러히스토그램 생성 함수
    ############################
    def build_feature_df(csv_df):
        rows = []
        for i, row in csv_df.iterrows():
            fn = row['img_path']
            label = row[target_label]

            img_bgr = cv2.imread(fn)
            if img_bgr is None:
                print(f"[경고] 이미지 로드 실패: {fn}")
                continue

            # BGR -> HSV 변환
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # 3D 컬러히스토그램 (bins=(8,8,8) 디폴트)
            hist = cv2.calcHist([img_hsv], [0,1,2], None, bins, [0,180, 0,256, 0,256])
            # 정규화
            hist = cv2.normalize(hist, hist).flatten()  # shape=(bins[0]*bins[1]*bins[2],)

            # 행 데이터 구성
            row_dict = {'img_path': fn, target_label: label}
            # hist 512차원 -> 열 이름 만들기
            for idx, val in enumerate(hist):
                row_dict[f'hist_{idx}'] = float(val)

            rows.append(row_dict)
        return pd.DataFrame(rows)

    ############################
    # (C) train_df, test_df 생성
    ############################
    train_df = build_feature_df(df_train_csv)
    test_df  = build_feature_df(df_test_csv)

    print(f"[학습셋] {len(train_df)} 개, [테스트셋] {len(test_df)} 개")
    if len(train_df)==0 or len(test_df)==0:
        print("[오류] DF가 비어있습니다. 중단.")
        return {}

    # feature_cols: 'hist_0' ~ 'hist_511' (bins=8,8,8 일 때 512개)
    feature_cols = sorted([c for c in train_df.columns if c.startswith('hist_')])

    # X, y
    X_full = train_df[feature_cols]
    y_full = train_df[target_label]

    X_test_all = test_df[feature_cols]
    y_test_all = test_df[target_label]

    ############################
    # (D) 스케일링 (StandardScaler)
    ############################
    scaler = StandardScaler()
    # 전체 train_df(=X_full)에 맞춰 fit
    scaler.fit(X_full)

    # transform
    X_full_sc = scaler.transform(X_full)
    X_test_sc = scaler.transform(X_test_all)

    ############################
    # (E) KFold 교차검증
    ############################
    print(f"\n=== KFold 교차검증 (n_splits={n_splits}) ===")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_map = {
        'Logistic':  LogisticRegression(max_iter=max_iter_logistic),
        'SVC':       SVC(probability=True, kernel='rbf', gamma='scale'),
        'KNN':       KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoost': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for model_name, model_obj in model_map.items():
        print(f"\n[모델: {model_name}]")
        acc_list, f1_list = [], []
        for tr_idx, val_idx in kf.split(X_full_sc):
            X_tr, X_val = X_full_sc[tr_idx], X_full_sc[val_idx]
            y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]

            model_obj.fit(X_tr, y_tr)
            pred_val = model_obj.predict(X_val)
            acc = accuracy_score(y_val, pred_val)
            f1_ = f1_score(y_val, pred_val, average='macro', zero_division=0)
            acc_list.append(acc)
            f1_list.append(f1_)

        print(f"- KFold Mean Acc={np.mean(acc_list):.4f}, F1={np.mean(f1_list):.4f}")

    ############################
    # (F) hold-out (train_test_split)
    ############################
    print("\n=== hold-out 학습 ===")
    from sklearn.model_selection import train_test_split
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_full_sc, y_full,
        test_size=0.2,
        random_state=42
    )

    all_models = {}
    for model_name, model_obj in model_map.items():
        model = model_obj
        model.fit(X_tr2, y_tr2)

        pred_val2 = model.predict(X_val2)
        acc2 = accuracy_score(y_val2, pred_val2)
        f12 = f1_score(y_val2, pred_val2, average='macro', zero_division=0)
        print(f"[{model_name}] hold-out => Acc={acc2:.4f}, F1={f12:.4f}")

        all_models[model_name] = model

    ############################
    # (G) 최종 테스트셋 평가
    ############################
    print("\n=== 테스트셋 평가 ===")
    best_model_name, best_f1 = None, -1.0
    for m_name, m_obj in all_models.items():
        pred_test = m_obj.predict(X_test_sc)
        acc_t = accuracy_score(y_test_all, pred_test)
        f_t   = f1_score(y_test_all, pred_test, average='macro', zero_division=0)
        print(f"- {m_name} => Acc={acc_t:.4f}, F1={f_t:.4f}")

        if f_t> best_f1:
            best_f1 = f_t
            best_model_name = m_name

    print(f"\n[테스트셋] 가장 좋은 모델 => {best_model_name} (F1={best_f1:.4f})")
    print("=== 모든 모델 학습 및 평가 완료 ===")

    return all_models

