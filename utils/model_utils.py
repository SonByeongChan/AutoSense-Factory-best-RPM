import csv

CSV_PATH = 'static/data/models.csv'

def load_models():
    models = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append({
                'date': row['date'],
                'mae': row['mae'],
                'loss': row['loss'],
                'loss_rate': row['loss_rate'],
                'name': row['name']
            })
    return models

def save_models(models):
    filenames = ['date', 'mae', 'loss', 'loss_rate', 'name', 'is_current']
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=filenames)
        writer.writeheader()
        for m in models:
            writer.writerow(m)

def set_current_model(select_name):
    models = load_models()
    for m in models:
        m['is_current'] = 'true' if m['name'] == select_name else ''
    save_models(models)

          # current_model.txt 업데이트
    pkl_name = f"{select_name}.pkl"
    with open('../current_model.txt', 'w') as f:
        f.write(pkl_name)

def delete_model(selected_name):
    models = load_models()
    models = [m for m in models if m['name'] != selected_name]
    save_models(models)

def get_current_model(models):
    for m in models:
        if m.get('is_current') == 'true':
            return m
    return models[-1] if models else None


from datetime import datetime

def append_model_log(date, mae, loss, loss_rate, model_name):
    new_entry = {
        'date': date,
        'mae': str(round(mae, 4)),
        'loss': str(round(loss, 4)),
        'loss_rate': str(round(loss_rate, 4)),
        'name': model_name,
        'is_current': ''
    }
    models = load_models()
    models.append(new_entry)
    save_models(models)