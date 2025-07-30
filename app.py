import eventlet
eventlet.monkey_patch()

from functools import wraps
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
import pandas as pd
from utils.analysis import detect_outliers, all_detect_outliers, all_detect_outliers_range_based
import os
import utils.model_utils as mu
from datetime import datetime

from flask_socketio import SocketIO, emit
import subprocess

period_map = {'1개월': '1m', '3개월': '3m', '6개월': '6m'}
app = Flask(__name__)
app.secret_key = 'your-secret-key'
socketio = SocketIO(app, async_mode='eventlet')

# ▶ 메인 페이지
@app.route('/')
def main():
    return render_template('main.html')

#=========================================================================

# ▶ 로그인 화면

# 관리자 로그인 화면 
@app.route('/login/admin', methods=['GET', 'POST'])
def login_admin():

        # ── (1) GET 요청 시 남아있는 플래시 메시지 모두 비우기 ──
    if request.method == 'GET':
        # 방법 A: 직접 세션에서 빼내기
        session.pop('_flashes', None)
        # 방법 B: get_flashed_messages() 호출해서 가져오기(자동으로 세션에서 제거됨)
        # _ = get_flashed_messages()
        
    if request.method == 'POST':
        admin_id = request.form.get('admin_id') # HTML에서 입력한 값 가져옴
        admin_pw = request.form.get('admin_pw')  # HTML에서 입력한 값 가져옴


        if admin_id == '1234' and admin_pw == '1234':

            return redirect(url_for('admin_dashboard'))  # 성공 → 다음 페이지
        else:
            flash('해당 관리자 번호가 없습니다. 다시 입력해 주세요.', 'error')  # 실패 → 다시 로그인 화면
    return render_template('login_admin.html')

# 작업자 로그인 화면
@app.route('/login/worker', methods = ['GET', 'POST'])
def login_worker():

            # ── (1) GET 요청 시 남아있는 플래시 메시지 모두 비우기 ──
    if request.method == 'GET':
        # 방법 A: 직접 세션에서 빼내기
        session.pop('_flashes', None)
        # 방법 B: get_flashed_messages() 호출해서 가져오기(자동으로 세션에서 제거됨)
        # _ = get_flashed_messages()

    # POST 요청이 들어온 경우 ( 즉, 로그인 폼 제출 시 )
    if request.method == 'POST':
        # HTML <form>에서 'pw' 이름을 가진 입력 필드의 값을 가져옴
        worker_pw = request.form.get('pw') 

        # 하드코딩된 비밀번호 검사 ( 임시 방식/실제로는 DB 사용 )
        if worker_pw == '1234':
            # 로그인 성공 시 작업자 대시보드로 이동
            return redirect(url_for('worker_dashboard')) 
        else:
            # 로그인 실패 시 flash 메시지로 사용자에게 알림
            flash('해당 사원 번호가 없습니다. 다시 입력해 주세요.', 'error')

    # 처음 페이지 접속이거나 실패 시 로그인 페이지 렌더링        
    return render_template('login_worker.html')

#=========================================================================

### ▶ 관리자 관련 화면

# 관리자 대시보드 
@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')

# 관리자 AI 모델 관리
@app.route('/admin/aimodel', methods=['GET', 'POST'])
def admin_ai_model():
    models = mu.load_models()
    selected_model = request.form.get('selected_model')
    action = request.form.get('action')

    # period_map dict 추가 (안전하게 재정의)
    period_map = {'1개월': '1m', '3개월': '3m', '6개월': '6m'}

    update_cycle_path = 'models/update_cycle.txt'
    if os.path.exists(update_cycle_path):
        with open(update_cycle_path, 'r', encoding='utf-8') as f:
            current_cycle = f.read().strip()
    else:
        current_cycle = '1개월'  # 기본값 fallback

    if request.method == 'POST':
        # ── 수동 재학습 실행 ──
        if action == 'manual_retrain':
            # a) Save chosen cycle
            cycle = request.form.get('update_cycle')
            if not cycle or cycle not in period_map:
                cycle = current_cycle  # fallback 안전 처리

            os.makedirs('models', exist_ok=True)
            with open(update_cycle_path, 'w', encoding='utf-8') as f:
                f.write(cycle)

            # b) Prevent double retrain
            open('models/is_retraining.txt', 'w').close()

            try:
                # c) Run retrain synchronously
                subprocess.run([
                    "python", "retraion_pipline.py", "--period", period_map[cycle]
                ], check=True)

                # d) Compare MAE and auto-apply
                all_models = mu.load_models()
                old_model = next((m for m in all_models if m.get('is_current')), None)
                new_model = all_models[-1]
                old_mae = float(old_model['mae']) if old_model else float('inf')
                new_mae = float(new_model['mae'])

                if new_mae < old_mae:
                    mu.set_current_model(new_model['name'])
                    flash(f'새 모델 {new_model["name"]} MAE {new_mae:.4f} < 이전 MAE {old_mae:.4f} → 교체되었습니다.', 'success')
                else:
                    flash(f'재학습된 모델 MAE({new_mae:.4f}) > 이전 MAE({old_mae:.4f}) → 교체되지 않았습니다.', 'warning')

            except subprocess.CalledProcessError as e:
                flash(f'재학습 중 오류 발생: {str(e)}', 'error')

            finally:
                # e) Clean up flag file
                if os.path.exists('models/is_retraining.txt'):
                    os.remove('models/is_retraining.txt')

            return redirect(url_for('admin_ai_model'))

        # ── 모델 선택 적용 ──
        elif action == 'apply':
            if not selected_model:
                flash('모델을 선택해주세요.', 'error')
            else:
                with open('current_model.txt', 'w', encoding='utf-8') as f:
                    f.write(selected_model)
                flash(f'{selected_model} 모델이 적용되었습니다.', 'success')
            return redirect(url_for('admin_ai_model'))


        elif action and action.startswith('delete'):
            parts = action.split(':', 1)
            if len(parts) == 2:
                selected_model = parts[1]
                print(f"[삭제 요청 수신] selected_model={selected_model}")

                csv_path = 'static/data/models.csv'
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df = df[df['name'] != selected_model]
                    df.to_csv(csv_path, index=False)

                    pkl_file = os.path.join('models', f"{selected_model}.pkl")
                    if os.path.exists(pkl_file):
                        os.remove(pkl_file)

                    flash(f'{selected_model} 모델과 파일이 삭제되었습니다.', 'success')
                else:
                    flash('CSV 파일이 존재하지 않습니다.', 'error')
            else:
                flash('모델 삭제 요청이 올바르지 않습니다.', 'error')
                
    # ── GET 요청 처리: 현재 적용 모델 정보 조회 ──
    try:
        with open('current_model.txt', 'r', encoding='utf-8') as f:
            current_name = f.read().strip().replace('.pkl', '')
        current_model = next((m for m in models if m['name'] == current_name), models[-1])
    except Exception:
        current_model = models[-1] if models else {
            'name': '-', 'mae': '-', 'loss': '-', 'loss_rate': '-', 'date': '-'
        }

    # ── 템플릿 렌더링 ──
    return render_template(
        'admin_ai_model.html',
        model_list=models,
        current_model=current_model,
        current_cycle=current_cycle
    )

# 관리자 현활 관리
@app.route('/admin/status')
def admin_status():
    today = datetime.today().strftime('%Y-%m-%d')
    return render_template("admin_status.html", selected_date=today)

# 관리자 시스템 관리
@app.route('/admin/system')
def admin_system():
    return render_template('admin_system.html')

#=========================================================================

### ▶ 작업자 화면

# 작업자 메인 대시보드 페이지
@app.route('/worker/dashboard')
def worker_dashboard():
    # 작업자 메인 대시보드 페이지
    return render_template('worker_dashboard.html')
#=========================================================================

# ▶ API: scale_pv 값 전체 및 누적합


@app.route('/api/production')
def get_production():

    # 1. 파일 경로 및 방어 코드
    # path = 'static/data/second_1016.csv'
    # path = 'static/data/recommended_k_rpm_corrected.csv'
    path = 'static/data/recommended_k_rpm_corrected_1.csv'

    if not os.path.exists(path):
        return jsonify({'error': 'CSV 파일이 존재하지 않습니다.'}), 404

    # 1-1 파일 로드
    df = pd.read_csv(path)
    
    # 2. 필수 컬럼의 존재 여부 확인 
    required_cols = ['E_scr_pv', 'c_temp_pv', 'n_temp_pv', 'scale_pv', 's_temp_pv', 'k_rpm_pv']
    
    for col in required_cols:
        if col not in df.columns:
            return jsonify({'error': '필요한 컬럼이 존재하지 않습니다.'}), 400

    # 3. 유효 데이터만 필터링 (중량, RPM이 0이 아닌 경우만)
    # df = df[(df['scale_pv'] != 0) & (df['k_rpm_pv'] != 0)]

    df = detect_outliers(df, ['E_scr_pv','c_temp_pv', 'k_rpm_pv', 'n_temp_pv','s_temp_pv'])
        
    # 총 생산량을 구하기 위한 변수
    values = df['scale_pv'].dropna().tolist()

    # 4. 각 데이터 추출
    labels = list(range(len(df)))
    rpm_values = df['k_rpm_pv'].dropna().tolist()
    chamber_temps = df['c_temp_pv'].dropna().tolist()
    output_temps = df['n_temp_pv'].dropna().tolist()
    screw_temps = df['s_temp_pv'].dropna().tolist()
    
    # worker_log 탭3
    log_rows = df[['scale_pv', 'k_rpm_pv', 'k_rpm_sv', 'c_temp_pv', 'c_temp_sv', 's_temp_pv', 's_temp_sv', 'n_temp_pv', 'n_temp_sv']].values.tolist()

    # 손실율을 계산하기 위한 리스트
    loss_values = [abs(v - 3.0) for v in values if v != 3.0]

    # 6. 중량 이상치 색상 분류 (기준: A=3.0, B=2.8, C=3.2, D=3.4)
    a, b, c, d = 3.30, 3.15, 2.97, 2.95
   
    # 점 색상 분류 (빨간/노란/정상)
    point_colors = []
    for val in values:
        if (val >= a) or (val <= d):
            point_colors.append('red')
        elif (b <= val < a) or (d < val <= c):
            point_colors.append('yellow')
        else:
            point_colors.append('green')  # 정상

    # 7. 최근 빨간 점 개수 체크 (팝업 조건용 1분에 20개개)
    red_count_recent = point_colors.count('red')
    trigger_popup = red_count_recent >= 20

    # 8. 불량품 갯수 계산
    defect_values = []

    for i in values:
        if i == 0:
            defect_values.append(1)
        else:
            defect_values.append(0)
    
    # 9. 이상치 탐지를 위한 변수
    E_scr_pv_is_outlier = df['E_scr_pv_is_outlier'].dropna().tolist()
    c_temp_pv_is_outlier = df['c_temp_pv_is_outlier'].dropna().tolist()
    k_rpm_pv_is_outlier = df['k_rpm_pv_is_outlier'].dropna().tolist()
    n_temp_pv_is_outlier = df['n_temp_pv_is_outlier'].dropna().tolist()
    s_temp_pv_is_outlier = df['s_temp_pv_is_outlier'].dropna().tolist()

    # 10. 모델로 인한 중량 변화 및 RPM 변화 지점에 관련된 것
    real_weight = df['real_weight'].dropna().tolist() if 'real_weight' in df.columns else []
    predicted_reco = df['predicted_weight_with_recommended'].dropna().tolist() if 'predicted_weight_with_recommended' in df.columns else []

    # recommended_k_rpm 변화 지점 감지
    change_points = []
    prev_rpm = None
    if 'recommended_k_rpm' in df.columns:
        for i, rpm in enumerate(df['recommended_k_rpm']):
            if prev_rpm is not None and rpm != prev_rpm:
                change_points.append(i)
            prev_rpm = rpm

    # 11. 최근 현재RPM 관 변경된 RPM
    current_rpm = df['current_k_rpm'].dropna().tolist() if 'current_k_rpm' in df.columns else []
    recommended_rpm = df['recommended_k_rpm'].dropna().tolist() if 'recommended_k_rpm' in df.columns else []

    # 12. 단위별 오차 계산
    error_real = [abs(w - 3.0) for w in real_weight]
    error_pred = [abs(p - 3.0) for p in predicted_reco]

    # 13. 누적 오차 계산
    cum_real_err = []
    cum_pred_err = []

    for i in range(len(error_real)):
        cum_real_err.append(error_real[i] + (cum_real_err[i-1] if i > 0 else 0))
        cum_pred_err.append(error_pred[i] + (cum_pred_err[i-1] if i > 0 else 0))

    # key와 value의 이름 통일(다들 편하게 해여여)
    return jsonify({
        'labels' : labels,  # 그래프 상 x축을 계산하기 위한 리스트의 길이
        'values': values,  # 10월 16일 scale_pv 리스트
        'loss_values' : loss_values, # 10월 16일 로스값 계산한 리스트
        
        
        'k_rpm_pv' : rpm_values, # rpm을 계속적으로 추출한 리스트
        'screw' : screw_temps,
        'chamber' : chamber_temps,
        'output' : output_temps,
        'log_rows': log_rows,
        
        'defect_values' : defect_values,  # 불량률

        'E_scr_pv_is_outlier' : E_scr_pv_is_outlier,    # 스크류 이상치 범위 기준
        'c_temp_pv_is_outlier' : c_temp_pv_is_outlier,  # 챔버 이상치 범위 기준
        'k_rpm_pv_is_outlier' : k_rpm_pv_is_outlier,    # RPM 이상치 범위 기준
        'n_temp_pv_is_outlier' : n_temp_pv_is_outlier,  # 노줄 이상치 범위 기준
        's_temp_pv_is_outlier' : s_temp_pv_is_outlier,  # 스크류 온도 이상치 범위위 기준

        'point_colors': point_colors,  # 점 색상 리스트
        # 'trigger_popup': trigger_popup  # 빨간 점 5개 이상인지 여부

        'real_weight': real_weight,         # 실제 데이터 기반 무게
        'predicted_reco': predicted_reco,   # RPM 변환 예측 기반 무게
        'change_points': change_points,     # RPM 변환 지점 표시
        'error_real': error_real,           # 실제 데이터의 3g기준의 오차
        'error_pred': error_pred,           # RPM 변환 후 데이터의 3g 기준의 오차 
        'cum_real_err': cum_real_err,       # 실제 데이터의 3g 기준의 누적 오차
        'cum_pred_err': cum_pred_err,       # PRM 변환 후 데이터의 3g 기준의 누적 데이터 오차

        'current_rpm' : current_rpm,
        'recommended_rpm' : recommended_rpm
    })

#===========================================================================================
# _all -> 우리가 받은 전체 데이터 이는 주간 통계와 월간 통계를 함께 보기 위함

@app.route('/api/production_all')
def get_production_all():

    path_all = 'static/data/Second.csv'
    if not os.path.exists(path_all):
        return jsonify({'error': 'CSV 파일이 존재하지 않습니다.'}), 404

    df_all = pd.read_csv(path_all)

    if 'time' not in df_all.columns:
        return jsonify({'error': 'time 컬럼이 필요합니다.'}), 400

    df_all['time'] = pd.to_datetime(df_all['time'])
    num_cols = ['scale_pv', 'E_scr_pv', 'c_temp_pv', 'k_rpm_pv', 'n_temp_pv', 's_temp_pv']
    
    
    for col in num_cols:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    df_all = df_all.dropna(subset=num_cols)

    df_all = all_detect_outliers_range_based(df_all, ['E_scr_pv','c_temp_pv', 'k_rpm_pv', 'n_temp_pv','s_temp_pv'])

    labels_all = df_all['time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    values_all = df_all['scale_pv'].tolist()
    loss_values_all = [abs(v - 3.0) for v in values_all]
    defect_values_all = [1 if abs(v - 3.0) > 1.5 else 0 for v in values_all]

    E_scr_pv_all_is_outlier = df_all['E_scr_pv_all_is_outlier'].fillna(False).astype(int).tolist()
    c_temp_pv_all_is_outlier = df_all['c_temp_pv_all_is_outlier'].fillna(False).astype(int).tolist()
    k_rpm_pv_all_is_outlier = df_all['k_rpm_pv_all_is_outlier'].fillna(False).astype(int).tolist()
    n_temp_pv_all_is_outlier = df_all['n_temp_pv_all_is_outlier'].fillna(False).astype(int).tolist()
    s_temp_pv_all_is_outlier = df_all['s_temp_pv_all_is_outlier'].fillna(False).astype(int).tolist()

    return jsonify({
    'labels_all': labels_all,
    'values_all': values_all,
    'loss_values_all': loss_values_all,
    'defect_values_all': defect_values_all,

    'scale_pv_all': df_all['scale_pv'].tolist(),

    'E_scr_pv_all_is_outlier' : E_scr_pv_all_is_outlier,
    'c_temp_pv_all_is_outlier' : c_temp_pv_all_is_outlier,
    'k_rpm_pv_all_is_outlier' : k_rpm_pv_all_is_outlier, 
    'n_temp_pv_all_is_outlier' : n_temp_pv_all_is_outlier,
    's_temp_pv_all_is_outlier' : s_temp_pv_all_is_outlier
    })

#===========================================================================================
@app.route('/api/production_day')
def get_day_production():
    selected_date = request.args.get('date')

    if not selected_date or selected_date == '1':
        return jsonify({'error': '날짜가 지정되지 않았습니다.'}), 400

    admin_status = pd.read_csv('static/data/Second.csv')
    admin_status['time'] = pd.to_datetime(admin_status['time'])

    # 🔧 ISO 형식 대비 .date()로 필터
    admin_status = admin_status[admin_status['time'].dt.date == pd.to_datetime(selected_date).date()]
    
    if admin_status.empty:
        return jsonify({
            'production_by_hour': [0]*24,
            'loss_by_hour': [0]*24,
            'defect_by_hour': [0]*24,
            'revenue_by_hour': [0]*24
        })

    admin_status['hour'] = admin_status['time'].dt.hour
    grouped = admin_status.groupby('hour')

    
    prod = grouped['scale_pv'].sum().reindex(range(24), fill_value=0).tolist()
    loss = grouped['scale_pv'].apply(lambda x: abs(x - 3.0).sum()).reindex(range(24), fill_value=0).tolist()
    defect = grouped['scale_pv'].apply(lambda x: (abs(x - 3.0) > 0.3).sum()).reindex(range(24), fill_value=0).tolist()
    
    # 소수점 올림 함수 
    revenue = [round(p * 3, 2) for p in prod]
    prod = [round(val,2) for val in prod]
    loss = [round(val, 2) for val in loss]
    defect = [round(val, 2) for val in defect]

    return jsonify({
        'production_by_hour': prod,
        'loss_by_hour': loss,
        'defect_by_hour': defect,
        'revenue_by_hour': revenue
    })

#===========================================================================================
@app.route('/api/production_week')
def get_week_production():
    selected_date = request.args.get('date')

    if not selected_date or selected_date == '1':
        return jsonify({'error': '날짜가 지정되지 않았습니다.'}), 400

    admin_status = pd.read_csv('static/data/Second.csv')
    admin_status['time'] = pd.to_datetime(admin_status['time'])

    # 해당 날짜 포함 주의 시작~종료 구간 설정 (월~일)
    selected = pd.to_datetime(selected_date)
    week_start = selected - pd.Timedelta(days=selected.weekday())
    week_end = week_start + pd.Timedelta(days=6)

    # 주간 필터링
    week_data = admin_status[
        (admin_status['time'].dt.date >= week_start.date()) &
        (admin_status['time'].dt.date <= week_end.date())
    ].copy()

    if week_data.empty:
        return jsonify({
            'production_by_day': [0]*7,
            'loss_by_day': [0]*7,
            'defect_by_day': [0]*7,
            'revenue_by_day': [0]*7
        })

    # 날짜별 그룹핑
    week_data['date'] = week_data['time'].dt.date
    grouped = week_data.groupby('date')

    # 일별 집계
    prod = grouped['scale_pv'].sum().reindex(pd.date_range(week_start, week_end).date, fill_value=0).tolist()
    loss = grouped['scale_pv'].apply(lambda x: abs(x - 3.0).sum()).reindex(pd.date_range(week_start, week_end).date, fill_value=0).tolist()
    defect = grouped['scale_pv'].apply(lambda x: (abs(x - 3.0) > 0.3).sum()).reindex(pd.date_range(week_start, week_end).date, fill_value=0).tolist()

    # 매출 및 반올림
    revenue = [round(p * 3, 2) for p in prod]
    prod = [round(val, 2) for val in prod]
    loss = [round(val, 2) for val in loss]
    defect = [round(val, 2) for val in defect]

    return jsonify({
        'production_by_day': prod,
        'loss_by_day': loss,
        'defect_by_day': defect,
        'revenue_by_day': revenue
    })

#=========================================================================
@app.route('/api/production_month')
def get_month_production():
    selected_date = request.args.get('date')

    if not selected_date or selected_date == '1':
        return jsonify({'error': '날짜가 지정되지 않았습니다.'}), 400

    admin_status = pd.read_csv('static/data/Second.csv')
    admin_status['time'] = pd.to_datetime(admin_status['time']).dt.tz_localize(None)

    selected = pd.to_datetime(selected_date)
    month_start = selected.replace(day=1)
    next_month = (month_start + pd.DateOffset(months=1)).replace(day=1)
    
    # 월간 데이터 필터링
    month_data = admin_status[
        (admin_status['time'] >= month_start) &
        (admin_status['time'] < next_month)
    ].copy()

    if month_data.empty:
        return jsonify({
            'production_by_week': [0]*6,
            'loss_by_week': [0]*6,
            'defect_by_week': [0]*6,
            'revenue_by_week': [0]*6
        })

    # 주차 정보 컬럼 추가 (1주차 ~)
    month_data['week'] = ((month_data['time'].dt.day - 1) // 7) + 1
    grouped = month_data.groupby('week')

    prod = grouped['scale_pv'].sum().reindex(range(1, 7), fill_value=0).tolist()
    loss = grouped['scale_pv'].apply(lambda x: abs(x - 3.0).sum()).reindex(range(1, 7), fill_value=0).tolist()
    defect = grouped['scale_pv'].apply(lambda x: (abs(x - 3.0) > 0.3).sum()).reindex(range(1, 7), fill_value=0).tolist()
    revenue = [round(p * 3, 2) for p in prod]

    prod = [round(val, 2) for val in prod]
    loss = [round(val, 2) for val in loss]
    defect = [round(val, 2) for val in defect]
    
    return jsonify({
        'production_by_week': prod,
        'loss_by_week': loss,
        'defect_by_week': defect,
        'revenue_by_week': revenue
    })
#=========================================================================
if __name__ == '__main__':
    app.run(debug=True)