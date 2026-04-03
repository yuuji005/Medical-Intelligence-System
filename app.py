from flask import Flask, render_template, request, redirect
import joblib
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Menggunakan path absolut agar Vercel tidak tersesat
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model_jumlah_pasien.pkl')
test_data_path = os.path.join(BASE_DIR, 'test_data.pkl')

def load_assets():
    try:
        model = joblib.load(model_path)
        test_data = joblib.load(test_data_path)
        return model, test_data
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return None, None

def generate_plot(model, X_all, y_all):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    tahun = X_all['tahun'].values
    plt.scatter(tahun, y_all, color='#2563eb', label='Data Aktual', s=120, edgecolors='white', linewidth=2)
    
    x_range = np.linspace(tahun.min() - 0.5, tahun.max() + 0.5, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='#ef4444', linewidth=3, label='Tren Prediksi Linear')
    
    plt.legend()
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    model, data = load_assets()
    if model and data:
        X_all, y_all, y_pred, mae, mse, r2 = data
        plot_url = generate_plot(model, X_all, y_all)
        return render_template('index.html', plot_url=plot_url, mae=f"{mae:.2f}", mse=f"{mse:.2f}", r2=f"{r2:.2f}")
    return "Gagal memuat model. Pastikan file .pkl ada di root dan tidak rusak."

@app.route('/predict', methods=['POST'])
def predict():
    model, data = load_assets()
    if not model: return redirect('/')
    try:
        tahun = int(request.form['tahun'])
        pred = model.predict(np.array([[tahun]]))[0]
        X_all, y_all, _, mae, mse, r2 = data
        plot_url = generate_plot(model, X_all, y_all)
        return render_template('index.html', plot_url=plot_url, mae=f"{mae:.2f}", mse=f"{mse:.2f}", r2=f"{r2:.2f}", prediksi=round(pred, 0), input_tahun=tahun)
    except:
        return redirect('/')

# Export untuk Vercel
app = app
