# ============================================================================================================
# 모델 재학습 파이프라인 (최종 수정본, 상세 주석 포함)
#
# 기능 개요:
# 1. 원본 공정 데이터와 새로 수집된 운영 데이터를 통합합니다.
# 2. 사용자 입력에 따라 특정 기간(1개월, 3개월 등)의 데이터만 필터링합니다.
# 3. 클리닝을 통해 이상값, 결측값 제거 후 학습에 적합한 데이터셋 생성합니다.
# 4. 기준 날짜를 기반으로 전처리된 데이터를 CSV로 저장합니다.
# 5. 선택적 하이퍼파라미터 튜닝(--tune 옵션 제공)
# 6. 모델 학습 후 성능 로그 및 결과 모델을 저장합니다.
# ============================================================================================================

# 필요한 외부 라이브러리들을 임포트
import pandas as pd
import numpy as np
import os
import yaml
import joblib
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import traceback
from utils.model_utils import append_model_log

# ----------------------------------------
# 📂 디렉토리 및 경로 설정
# ----------------------------------------
os.makedirs("config", exist_ok=True)  # 설정파일 저장 폴더
os.makedirs("models", exist_ok=True)  # 모델 파일 저장 폴더
os.makedirs("data", exist_ok=True)    # 전처리된 데이터 저장 폴더

# 파일 경로 정의
RAW_DATA_PATH = "./static/data/imputed_filtered_train_data_2025.csv"
NEW_LOG_PATH = "./static/data/second_1016.csv"
BEST_PARAMS_CONFIG_PATH = "config/best_params.yaml"
ACTIVE_MODEL_CONFIG_PATH = "config/active_model.yaml"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "Khold_scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "Khold_feature_names.pkl")

# ----------------------------------------
# 🛠️ 원본 + 신규 데이터 통합 함수
# ----------------------------------------
def prepare_raw_data():
    """
    원본 데이터와 신규 로그 데이터를 통합하고, 시간 기준 중복 제거
    """
    print("="*50)
    print("1. Raw 데이터 통합 시작...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"원본 Raw 데이터 파일이 없습니다: {RAW_DATA_PATH}")
    base_df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    print(f"  - 원본 데이터 로드: {len(base_df)}건")

    if os.path.exists(NEW_LOG_PATH):
        new_df = pd.read_csv(NEW_LOG_PATH, encoding="utf-8-sig")
        print(f"  - 신규 데이터 로드: {len(new_df)}건")
        combined_df = pd.concat([base_df, new_df], ignore_index=True).drop_duplicates(subset=['time'])
    else:
        print("  - 신규 데이터 없음. 원본 데이터만 사용")
        combined_df = base_df

    print(f"  - 통합된 총 데이터: {len(combined_df)}건")
    print("="*50)
    return combined_df

# ----------------------------------------
# 🧼 데이터 전처리 및 필터링 함수
# ----------------------------------------
def preprocess_data(df, period, target_date_str):
    """
    필터링 및 클리닝을 수행하고 결과를 CSV로 저장
    """
    print("2. 데이터 전처리 시작...")

    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
    df.dropna(subset=['time'], inplace=True)

    if period != 'all':
        period_map = {'1m': 1, '3m': 3, '6m': 6, '1y': 12}
        months_to_subtract = period_map.get(period)
        end_date = pd.to_datetime(target_date_str, format='%Y%m%d').tz_localize('UTC')
        start_date = end_date - DateOffset(months=months_to_subtract)

        print(f"  - 기간 필터링 적용: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
        print(f"  - 필터링 후 데이터 수: {len(df)}건")
    else:
        print("  - 전체 기간 사용")

    df.loc[df["n_temp_sv"] == 0, "n_temp_sv"] = 70
    invalid_mask = (
        (df["n_temp_sv"] == 0) | (df["k_rpm_pv"] == 0) | (df["E_scr_pv"] == 0) |
        (df["k_rpm_pv"] <= 100) | (df["k_rpm_sv"] <= 100)
    )
    df_clean = df[~invalid_mask].copy()
    df_final = df_clean[(df_clean["scale_pv"] >= 2.90) & (df_clean["scale_pv"] <= 3.30)].copy()

    print(f"  - 유효 샘플 수: {len(df_final)} (제거된 샘플 수: {len(df) - len(df_final)})")

    output_filename = f"{target_date_str}_combined_preprocessed_data.csv"
    output_path = os.path.join("data", output_filename)

    try:
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  - CSV 저장 완료: {output_path}")
    except Exception as e:
        print(f"  - ❌ 저장 실패: {e}")

    print("="*50)
    return df_final

# ----------------------------------------
# 🔍 하이퍼파라미터 튜닝 함수
# ----------------------------------------
def tune_hyperparameters_and_save(df):
    """
    GridSearchCV를 활용한 랜덤 포레스트 모델의 하이퍼파라미터 탐색 및 저장
    """
    print("3. 하이퍼파라미터 튜닝 시작...")
    drop_cols = ["time", "scale_pv", "k_rpm_sv", "s_temp_sv", "E_scr_sv", "E_scr_pv", "n_temp_sv", "c_temp_sv", "is_imputed"]
    final_features = df.drop(columns=drop_cols, errors='ignore').columns
    X = df[final_features].astype(float)
    y = df["scale_pv"].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring="r2", cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_scaled, y)

    best_params = grid_search.best_params_
    print("  - 최적 파라미터:", best_params)
    with open(BEST_PARAMS_CONFIG_PATH, "w") as f:
        yaml.dump(best_params, f, allow_unicode=True)
    print(f"  - 저장 완료: {BEST_PARAMS_CONFIG_PATH}")
    print("="*50)

# ----------------------------------------
# 🤖 모델 학습 및 저장 함수
# ----------------------------------------
def train_and_save_model(df, target_date_str):
    """
    RandomForestRegressor 학습 후 모델 및 성능 메트릭 저장
    """
    print("4. 모델 학습 및 저장 시작...")
    drop_cols = ["time", "scale_pv", "k_rpm_sv", "s_temp_sv", "E_scr_sv", "E_scr_pv", "n_temp_sv", "c_temp_sv", "is_imputed"]
    final_features = df.drop(columns=drop_cols, errors='ignore').columns
    X = df[final_features].astype(float)
    y = df["scale_pv"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(final_features.tolist(), FEATURES_PATH)

    print("  - 스케일러 및 피처 저장 완료")

    if not os.path.exists(BEST_PARAMS_CONFIG_PATH):
        raise FileNotFoundError("하이퍼파라미터 파일이 없습니다. --tune 옵션으로 먼저 실행하세요.")

    with open(BEST_PARAMS_CONFIG_PATH, "r") as f:
        best_params = yaml.safe_load(f)

    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    r2 = model.score(X_scaled, y)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)

    print(f"  - R²: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE : {mae:.4f}")

    filename = f"{target_date_str}_RF_r2_{r2:.4f}_mae_{mae:.4f}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"  - 모델 저장 완료: {filepath}")

    log_path = os.path.join(MODELS_DIR, "model_performance_log.csv")
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_filename": filename,
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "data_size": len(df)
    }])
    if not os.path.exists(log_path):
        log_entry.to_csv(log_path, index=False, encoding='utf-8-sig')
    else:
        log_entry.to_csv(log_path, mode='a', header=False, index=False, encoding='utf-8-sig')

    with open(ACTIVE_MODEL_CONFIG_PATH, "w") as f:
        yaml.dump({
            "model_path": filepath,
            "scaler_path": SCALER_PATH,
            "feature_names_path": FEATURES_PATH,
            "performance": {"r2": r2, "mse": mse, "rmse": rmse}
        }, f, allow_unicode=True, sort_keys=False)


    append_model_log(
    date=datetime.now().strftime('%Y-%m-%d %H:%M'),
    mae=mae,
    loss=np.sum(np.abs(y_pred - 3.00)),
    loss_rate=np.sum(np.abs(y_pred - 3.00)) / np.sum(y),
    model_name=filename.replace('.pkl', '')
)


print("="*50)



# ----------------------------------------
# 🚀 메인 함수: 인자 파싱 및 전체 파이프라인 실행
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 재학습 파이프라인")
    parser.add_argument('--tune', action='store_true', help="하이퍼파라미터 튜닝 실행 여부")
    parser.add_argument('--period', type=str, default='all', choices=['1m', '3m', '6m', '1y', 'all'], help="재학습 기간 선택")
    parser.add_argument('--date', type=str, default=None, help="기준 날짜 지정 (YYYYMMDD)")
    args = parser.parse_args()

    # 날짜 검증 및 설정

    if args.date:
        try:
            datetime.strptime(args.date, '%Y%m%d')
            target_date_str = args.date
        except ValueError:
            print("❌ 잘못된 날짜 형식입니다. YYYYMMDD 형식으로 입력해주세요.")
            exit()
    else:
        target_date_str = datetime.now().strftime('%Y%m%d')

    print(f"🚀 기준 날짜: {target_date_str} / 데이터 범위: {args.period}")

    try:
        raw_combined_data = prepare_raw_data()
        clean_data = preprocess_data(raw_combined_data, args.period, target_date_str)

        if clean_data.empty:
            print("❌ 전처리 후 사용할 수 있는 데이터가 없습니다.")
        else:
            if args.tune:
                tune_hyperparameters_and_save(clean_data)
            else:
                print("3. 하이퍼파라미터 튜닝 생략 (기존 값 사용)")
            train_and_save_model(clean_data, target_date_str)
            print("🎉 모델 재학습 완료")

    except Exception as e:
        print("❌ 오류 발생:", e)
        traceback.print_exc()
