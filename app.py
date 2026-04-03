from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Load model dan data test
model = joblib.load('model_jumlah_pasien.pkl')
X_all, y_all, y_pred_all = joblib.load('test_data.pkl')

# Hitung metrik
mae = mean_absolute_error(y_all, y_pred_all)
mse = mean_squared_error(y_all, y_pred_all)
r2 = r2_score(y_all, y_pred_all)

def plot_comparison():
    """Scatter plot data aktual dan garis regresi linear dengan tema medis"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6), facecolor='#ffffff')
    
    tahun = X_all['tahun'].values
    
    # Data aktual Deep Blue
    plt.scatter(tahun, y_all, color='#1e3a8a', label='Data Aktual', s=100, alpha=0.8, edgecolors='white', linewidth=1.5)
    
    # Tren Prediksi Teal
    x_range = np.linspace(tahun.min() - 0.5, tahun.max() + 0.5, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='#0d9488', linewidth=3, label='Tren Prediksi', linestyle='-')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.xlabel('Tahun Analisis', fontsize=11, fontweight='bold', color='#475569')
    plt.ylabel('Volume Pasien', fontsize=11, fontweight='bold', color='#475569')
    plt.title('Visualisasi Tren Kunjungan Pasien', fontsize=14, fontweight='bold', pad=20, color='#1e293b')
    
    plt.legend(frameon=True, facecolor='white', shadow=True)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    plot_url = plot_comparison()
    return render_template('index.html', plot_url=plot_url, mae=mae, mse=mse, r2=r2, prediksi=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tahun = int(request.form['tahun'])
        input_data = np.array([[tahun]])
        pred = model.predict(input_data)[0]
        plot_url = plot_comparison()
        return render_template('index.html', plot_url=plot_url, mae=mae, mse=mse, r2=r2, prediksi=round(pred, 0))
    except Exception as e:
        plot_url = plot_comparison()
        return render_template('index.html', plot_url=plot_url, mae=mae, mse=mse, r2=r2, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
