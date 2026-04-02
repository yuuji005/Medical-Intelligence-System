import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==========================================
# 1. Baca dataset jumlah-pasien.csv
# ==========================================
with open('jumlah-pasien.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Cari baris "Series Column"
series_col_index = None
for i, line in enumerate(lines):
    if 'Series Column' in line:
        series_col_index = i
        break

if series_col_index is None:
    raise ValueError("Tidak menemukan 'Series Column' dalam file")

# Baris setelah "Series Column" adalah header "Tahun"
tahun_row = lines[series_col_index + 1].strip().split(';')
jumlah_row = lines[series_col_index + 2].strip().split(';')

# Ekstrak tahun dan jumlah (kolom B, C, D, E)
tahun = []
jumlah = []
for col in range(1, 5):  # kolom indeks 1 sampai 4
    tahun_val = tahun_row[col].strip().replace('"', '')
    jumlah_val = jumlah_row[col].strip().replace('"', '').replace(',', '')
    if tahun_val and jumlah_val:
        tahun.append(int(tahun_val))
        jumlah.append(float(jumlah_val))

# Buat DataFrame
df = pd.DataFrame({'tahun': tahun, 'jumlah_pasien': jumlah})

# ==========================================
# 2. Latih model regresi linear dengan seluruh data
# ==========================================
X = df[['tahun']]
y = df['jumlah_pasien']

model = LinearRegression()
model.fit(X, y)

# Prediksi in-sample
y_pred = model.predict(X)

# ==========================================
# 3. Hitung metrik evaluasi
# ==========================================
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2: {r2:.2f}")

# ==========================================
# 4. Simpan model dan data (untuk keperluan grafik)
# ==========================================
joblib.dump(model, 'model_jumlah_pasien.pkl')
joblib.dump((X, y, y_pred), 'test_data.pkl')

print("Model dan data test disimpan.")