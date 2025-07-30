import pandas as pd
from scipy.stats import zscore

def detect_outliers(df, columns, threshold=3):
    """
    여러 컬럼에 대해 Z-score 기반 이상치 탐지를 수행하고, 
    이상치 여부를 컬럼별로 별도 컬럼으로 추가합니다.

    Parameters:
    - df: 입력 DataFrame
    - columns: 이상치 탐지를 적용할 컬럼 리스트
    - threshold: Z-score 기준값 (기본값 3)

    Returns:
    - 이상치 여부 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue  # 없는 컬럼은 건너뜀
        z_col = f'{col}_zscore'
        outlier_col = f'{col}_is_outlier'
        
        df[z_col] = zscore(df[col].dropna())
        df[outlier_col] = df[z_col].abs() > threshold
    
    return df

# all 전체를 구현하기 위한 함수
def all_detect_outliers(df, columns, threshold=3):
    """
    여러 컬럼에 대해 Z-score 기반 이상치 탐지를 수행하고, 
    이상치 여부를 컬럼별로 별도 컬럼으로 추가합니다.

    Parameters:
    - df: 입력 DataFrame
    - columns: 이상치 탐지를 적용할 컬럼 리스트
    - threshold: Z-score 기준값 (기본값 3)

    Returns:
    - 이상치 여부 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue  # 없는 컬럼은 건너뜀
        z_col = f'{col}_all_zscore'
        outlier_col = f'{col}_all_is_outlier'
        
        df[z_col] = zscore(df[col].dropna())
        df[outlier_col] = df[z_col].abs() > threshold
    
    return df



## 범위에 따른 이상치 탐색을 위한 함수


# 전체 
def all_detect_outliers_range_based(df, columns, lower=68, upper=73):
    """
    여러 컬럼에 대해 지정된 범위를 벗어나는 값을 이상치로 판단하고,
    이상치 여부를 컬럼별로 별도 컬럼에 저장합니다.

    Parameters:
    - df: 입력 DataFrame
    - columns: 이상치 탐지를 적용할 컬럼 리스트
    - lower: 허용 최소값 (기본 68)
    - upper: 허용 최대값 (기본 73)

    Returns:
    - 이상치 여부 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        outlier_col = f'{col}_all_is_outlier'
        df[outlier_col] = (df[col] < lower) | (df[col] > upper)
    
    return df