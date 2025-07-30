# 📦 주요 라이브러리 불러오기
import pandas as pd                       # 데이터프레임 사용
import joblib                             # 모델 및 스케일러 로드
import os                                 # 파일 경로 및 디렉토리 생성
import matplotlib.pyplot as plt           # 시각화
import matplotlib                         # GUI 없이 저장만 가능하게 설정
from datetime import datetime             # 현재 시간 가져오기

matplotlib.use("Agg")  # 서버 환경에서도 시각화 저장 가능하게 설정

# 🔤 한글 폰트 설정 (Windows 환경 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지

# ----------------------------------------
# ✅ 경로 설정 및 파일 존재 확인
# ----------------------------------------
model_path = "./static/models/20250602_RandomForest_r2_0.8527_mse_0.000172.pkl"
scaler_path = "./static/models/Khold_scaler.pkl"
feature_names_path = "./static/models/Khold_feature_names.pkl"
# data_path = "./static/data/clean_final_Test_data.csv"

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("output", exist_ok=True)
output_path = f"output/{now_str}_recommended_k_rpm_corrected.csv"

# 파일이 실제로 존재하는지 확인
assert os.path.exists(model_path), f"❌ 모델 파일 없음: {model_path}"
assert os.path.exists(scaler_path), f"❌ 스케일러 파일 없음: {scaler_path}"
assert os.path.exists(feature_names_path), f"❌ 피처 파일 없음: {feature_names_path}"
# assert os.path.exists(data_path), f"❌ 데이터 파일 없음: {data_path}"


# def predict_weight_batch(k_rpm_list, fixed_inputs):
#     df_list = []
#     for rpm in k_rpm_list:
#         row = fixed_inputs.copy()
#         row["k_rpm_pv"] = rpm
#         df_list.append(row)
#     df = pd.DataFrame(df_list)[feature_names]
#     X_scaled = scaler.transform(df)
#     return model.predict(X_scaled)

# ----------------------------------------
# 📥 모델, 스케일러, 피처 목록 불러오기
# ----------------------------------------
print("📦 모델 및 스케일러 로드 중...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_names_path)
print('완료')

# ----------------------------------------
# 🔁 여러 회전수로 중량 예측 (배치)
# ----------------------------------------
def predict_weight_batch(k_rpm_list, fixed_inputs):
    """
    회전수 리스트를 받아 해당 조건의 예측 중량을 반환
    fixed_inputs: k_rpm 외 모든 공정 값들
    """
    df_list = []
    for rpm in k_rpm_list:
        row = fixed_inputs.copy()
        row["k_rpm_pv"] = rpm
        df_list.append(row)
    df = pd.DataFrame(df_list)[feature_names]
    X_scaled = scaler.transform(df)
    return model.predict(X_scaled)

# ----------------------------------------
# 🎯 최적 k_rpm 찾기 (보정 대상 중량에 가장 가까운 rpm 탐색)
# ----------------------------------------
def find_best_k_rpm(fixed_inputs, corrected_target, current_k_rpm):
    for delta in [10, 20, 30]:  # 탐색 범위를 점점 확장
        search_range = list(range(max(100, current_k_rpm - delta), min(201, current_k_rpm + delta + 1)))
        try:
            preds = predict_weight_batch(search_range, fixed_inputs)
        except:
            continue

        diff_list = [abs(p - corrected_target) for p in preds]
        best_idx = diff_list.index(min(diff_list))
        best_k_rpm = search_range[best_idx]
        predicted_map = dict(zip(search_range, preds))

        # 예측 중량이 목표와 충분히 가까우면 조기 종료
        if abs(preds[best_idx] - corrected_target) <= 0.01:
            return best_k_rpm, round(preds[best_idx], 2), predicted_map

    # 실패해도 가장 근접한 회전수 반환
    best_idx = diff_list.index(min(diff_list))
    return search_range[best_idx], round(preds[best_idx], 2), dict(zip(search_range, preds))

# ----------------------------------------
# ✅ 단일 회전수로 중량 예측
# ----------------------------------------
def predict_weight(k_rpm, fixed_inputs):
    return predict_weight_batch([k_rpm], fixed_inputs)[0]

# ----------------------------------------
# ⚙️ 보정 알고리즘: 실측값 기준으로 k_rpm 조정 추천
# ----------------------------------------
def recommend_k_rpm_with_residual_correction(fixed_inputs, real_weight, current_k_rpm):
    predicted = round(predict_weight(current_k_rpm, fixed_inputs), 2)

    # 목표 값이 3.00g ±0.01 이내면 조정 불필요
    if abs(real_weight - 3.00) <= 0.01:
        print(f"✅ 실측 중량이 3.00g ±0.01 이내입니다. 현재 k_rpm 유지: {current_k_rpm}")
        return current_k_rpm, predicted, round(real_weight - predicted, 2), 3.00, predicted

    # 잔차 기반 보정 목표 계산
    residual = round(real_weight - predicted, 2)
    corrected_target = round(3.00 - residual, 2)

    print(f"  • 실제 중량 = {real_weight}g")
    print(f"  • 예측 중량 = {predicted}g")
    print(f"  • 예측 오차 = {residual:+.2f}g")
    print(f"  • 보정된 목표 중량 = {corrected_target}g")

    best_k_rpm, predicted_corrected, predicted_map = find_best_k_rpm(fixed_inputs, corrected_target, current_k_rpm)

    if best_k_rpm == current_k_rpm:
        print(f"✅ 추천된 k_rpm({best_k_rpm})이 현재 설정과 동일합니다. 유지 가능합니다.")

    print(f"  ⚙️ 현재 설정된 k_rpm: {current_k_rpm}")
    print(f"  ✅ 추천 회전수(k_rpm): {best_k_rpm} → 보정된 예측 중량: {predicted_corrected}g")

    return best_k_rpm, predicted, residual, corrected_target, predicted_corrected