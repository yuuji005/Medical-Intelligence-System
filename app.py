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

# Load model dan data test (seluruh data + prediksi in-sample)
model = joblib.load('model_jumlah_pasien.pkl')
X_all, y_all, y_pred_all = joblib.load('test_data.pkl')

# Hitung metrik menggunakan seluruh data (in-sample)
mae = mean_absolute_error(y_all, y_pred_all)
mse = mean_squared_error(y_all, y_pred_all)
r2 = r2_score(y_all, y_pred_all)

def plot_comparison():
    """Scatter plot data aktual dan garis regresi linear"""
    plt.figure(figsize=(10, 6))
    
    # Data aktual
    tahun = X_all['tahun'].values
    plt.scatter(tahun, y_all, color='blue', label='Data Aktual', s=80, alpha=0.7)
    
    # Garis regresi (prediksi untuk seluruh rentang tahun)
    # Buat garis lurus dari min tahun ke max tahun + sedikit perpanjangan
    x_range = np.linspace(tahun.min() - 1, tahun.max() + 1, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='red', linewidth=2, label='Prediksi (Regresi Linear)')
    
    # Label sumbu dan judul
    plt.xlabel('Tahun', fontsize=12)
    plt.ylabel('Jumlah Pasien', fontsize=12)
    plt.title('Hasil Visualisasi', fontsize=14)
    plt.legend()
    
    # Opsional: tampilkan koordinat salah satu titik (misal titik terakhir)
    # Tampilkan anotasi pada titik data aktual terakhir
    last_x = tahun[-1]
    last_y = y_all.iloc[-1] if hasattr(y_all, 'iloc') else y_all[-1]
    plt.annotate(f'x={last_x}, y={last_y:.0f}',
                 xy=(last_x, last_y),
                 xytext=(last_x+0.2, last_y+5000),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Simpan ke base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
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
app = app

if __name__ == '__main__':
    app.run(debug=True)