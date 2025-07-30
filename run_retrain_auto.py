# run_retrain_auto.py
import subprocess
from datetime import datetime

# update_cycle.txt 경로
update_cycle_path = "models/update_cycle.txt"

# 기본값 매핑
period_map = {
    '1개월': '1m',
    '3개월': '3m',
    '6개월': '6m'
}

try:
    # 1️⃣ update_cycle.txt 읽기 → 먼저 period_arg 를 준비
    with open(update_cycle_path, 'r') as f:
        cycle_text = f.read().strip()

    period_arg = period_map.get(cycle_text, '3m')  # fallback 기본값 3m

    print(f"📅 현재 설정된 자동 업데이트 주기: {cycle_text} → period 인자: {period_arg}")

    # 2️⃣ 재학습 START 로그 기록
    with open("models/retrain_log.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [START] Retraining started with period: {period_arg}\n")

    # 3️⃣ 재학습 실행
    subprocess.run([
        "python",
        "retraion_pipline.py",
        "--period", period_arg
    ], check=True)

    # 4️⃣ 재학습 DONE 로그 기록
    with open("models/retrain_log.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [DONE] Retraining finished\n")

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    # 실패도 로그에 남기면 좋음:
    with open("models/retrain_log.txt", "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [FAIL] Retraining failed: {e}\n")
