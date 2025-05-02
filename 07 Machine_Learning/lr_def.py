# lr_def.py 

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
# 6) 모델과 테스트 데이터를 받아 예측 후 r2_score와 RMSE를 계산·출력하고 반환
# 7) 사용자 입력 -> float 변환 -> DataFrame 구성 -> 예측 -> 결과 출력
# 8) DecisionTreeClassifier를 통해 시각화 (graphviz) 


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
            train_mse = mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
        
        # cross-validation 평균
        avg_train_score = np.mean(train_scores)
        avg_train_loss = np.mean(train_losses)
        
        # 전체 train으로 다시 학습 후 test 데이터 평가
        ridge_model.fit(x_train, y_train)
        pred_test = ridge_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss = mean_squared_error(y_test, pred_test)
        
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
            train_mse = mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
        
        avg_train_score = np.mean(train_scores)
        avg_train_loss = np.mean(train_losses)
        
        lasso_model.fit(x_train, y_train)
        pred_test = lasso_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss = mean_squared_error(y_test, pred_test)
        
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
            train_mse = mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
        
        avg_train_score = np.mean(train_scores)
        avg_train_loss = np.mean(train_losses)
        
        elastic_model.fit(x_train, y_train)
        pred_test = elastic_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss = mean_squared_error(y_test, pred_test)
        
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
            train_mse = mean_squared_error(Y_tr, pred_tr)
            
            train_scores.append(train_r2)
            train_losses.append(train_mse)
            
        # cross-validation 평균
        avg_train_score = np.mean(train_scores)
        avg_train_loss  = np.mean(train_losses)
        
        # 전체 train으로 다시 학습 후 test 데이터 평가
        linear_model.fit(x_train, y_train)
        pred_test = linear_model.predict(x_test)
        test_score = r2_score(y_test, pred_test)
        test_loss  = mean_squared_error(y_test, pred_test)
       
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
    from sklearn.metrics import mean_squared_error
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"R2 Score: {score:.4f}, RMSE: {rmse:.4f}")
    return score, rmse


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
# 8) graphviz + DecisionTreeClassifier + export_graphviz 
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
