from flask import Flask, render_template, request, redirect
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = Flask(__name__)

# Path absolut untuk Vercel
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / 'model_jumlah_pasien.pkl'
test_data_path = BASE_DIR / 'test_data.pkl'

# Global variables
model = None
X_all = None
y_all = None
mae = mse = r2 = None

def load_assets():
    global model, X_all, y_all, mae, mse, r2
    if model is not None:
        return True
    try:
        model = joblib.load(model_path)
        X_all, y_all, _, mae, mse, r2 = joblib.load(test_data_path)
        return True
    except Exception as e:
        print(f"Error loading: {e}")
        return False

def plot_comparison():
    if not load_assets():
        return ""
    
    tahun = X_all['tahun'].values
    y_aktual = y_all.values
    
    # Prediksi untuk range
    x_range = np.linspace(tahun.min() - 0.5, tahun.max() + 0.5, 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    
    # Ukuran gambar
    width, height = 800, 500
    margin = 60
    
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Skala
    x_min, x_max = tahun.min() - 0.5, tahun.max() + 0.5
    y_min, y_max = min(y_aktual.min(), y_range.min()) - 5000, max(y_aktual.max(), y_range.max()) + 5000
    
    def map_x(x):
        return margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
    
    def map_y(y):
        return height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)
    
    # Sumbu
    draw.line([(margin, height - margin), (width - margin, height - margin)], fill='black', width=2)
    draw.line([(margin, margin), (margin, height - margin)], fill='black', width=2)
    
    # Label sumbu (pakai font default)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((width//2 - 20, height - 30), "Tahun", fill='black', font=font)
    draw.text((20, height//2), "Jumlah Pasien", fill='black', font=font)
    
    # Gambar garis regresi
    points = [(map_x(x), map_y(y)) for x, y in zip(x_range, y_range)]
    draw.line(points, fill='#ef4444', width=3)
    
    # Gambar titik data aktual
    for x, y in zip(tahun, y_aktual):
        xp, yp = map_x(x), map_y(y)
        draw.ellipse([xp-6, yp-6, xp+6, yp+6], fill='#2563eb', outline='white')
    
    # Konversi ke base64
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    plot_url = base64.b64encode(img_buffer.getvalue()).decode()
    return plot_url

@app.route('/')
def index():
    if not load_assets():
        return "Gagal memuat model. Periksa file .pkl."
    plot_url = plot_comparison()
    return render_template('index.html', 
                           plot_url=plot_url,
                           mae=f"{mae:.2f}",
                           mse=f"{mse:.2f}",
                           r2=f"{r2:.2f}")

@app.route('/predict', methods=['POST'])
def predict():
    if not load_assets():
        return redirect('/')
    try:
        tahun = int(request.form['tahun'])
        # Gunakan DataFrame untuk menghindari warning
        input_df = pd.DataFrame([[tahun]], columns=['tahun'])
        pred = model.predict(input_df)[0]
        plot_url = plot_comparison()
        return render_template('index.html',
                               plot_url=plot_url,
                               mae=f"{mae:.2f}",
                               mse=f"{mse:.2f}",
                               r2=f"{r2:.2f}",
                               prediksi=round(pred, 0),
                               input_tahun=tahun)
    except:
        return redirect('/')

# Untuk Vercel, aplikasi harus bernama 'app'
app = app

if __name__ == '__main__':
    app.run(debug=True)
