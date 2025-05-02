##----------------------------------------------
## pandas 데이터 관련 사용자 정의 함수들
##----------------------------------------------
def print_attribute(df,dfname):
    ## - 데이터프레임의 대표속성
    print(f'---[{dfname}]---')
    print(f'df.dtypes => {df.dtypes}')
    print(f'df.index => {df.index}')
    print(f'df.columns => {df.columns}')
    print(f'df.ndim => {df.ndim}')
    print(f'df.shape => {df.shape}')
   
    
def summary(df,dfname):
    print(f'---[{dfname}]---')
    
    # 요약정보 출력
    df.info()
    
    # 실제 데이터 일부 출력
    print(df.head(2),df.tail(2),sep='\n\n')
    
    # 통계치 출력
    print(df.describe())