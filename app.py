from flask import Flask, render_template, request, redirect
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / 'model_jumlah_pasien.pkl'
test_data_path = BASE_DIR / 'test_data.pkl'

model = None
X_all = y_all = None
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

@app.route('/')
def index():
    if not load_assets():
        return "Gagal memuat model. Periksa file .pkl."
    # Kirim plot_url kosong sementara
    return render_template('index.html', plot_url="", 
                           mae=f"{mae:.2f}", mse=f"{mse:.2f}", r2=f"{r2:.2f}")

@app.route('/predict', methods=['POST'])
def predict():
    if not load_assets():
        return redirect('/')
    try:
        tahun = int(request.form['tahun'])
        input_df = pd.DataFrame([[tahun]], columns=['tahun'])
        pred = model.predict(input_df)[0]
        return render_template('index.html', plot_url="",
                               mae=f"{mae:.2f}", mse=f"{mse:.2f}", r2=f"{r2:.2f}",
                               prediksi=round(pred, 0), input_tahun=tahun)
    except Exception as e:
        print(f"Prediction error: {e}")
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
