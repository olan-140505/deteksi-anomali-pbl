# Deteksi Anomali Kegagalan Perangkat

Proyek PBL Pembelajaran Mesin, Kelompok 14, Kelas A.

Repositori ini membandingkan Autoencoder, Isolation Forest, Hybrid AE+IF (α = 0,5), Random Forest, dan XGBoost untuk memprediksi kegagalan perangkat dari data telemetri harian.

**Aplikasi (live):** https://deteksi-anomali-pbl-bkcrhl46gqxec9q92s4ghm.streamlit.app

## Dataset

- Sumber: Kaggle, `hiimanshuagarwal/predictive-maintenance-dataset` (lisensi MIT)
- 124.494 baris, 1.169 perangkat, 106 failure (0,085%)
- Dataset **tidak** disertakan di repositori. Unduh dari Kaggle lalu simpan sebagai `predictive_maintenance.csv` di folder yang sama dengan skrip.

## Struktur

| File | Isi |
|---|---|
| `pm_features.py` | Fitur per perangkat: log, rata-rata 7 hari, dan std 7 hari |
| `train_v2_fixed.py` | Training, validasi silang 5-fold dikelompokkan per perangkat, evaluasi, dan model final |
| `app.py` | Aplikasi Streamlit untuk memprediksi dari CSV riwayat harian |
| `models/` | Model hasil training: Autoencoder, Isolation Forest, Random Forest, XGBoost, scaler, `config.json`, dan hasil evaluasi |
| `requirements.txt` | Daftar paket dengan versi yang sudah diuji |

## Cara menjalankan

### 1. Instalasi

```bash
pip install -r requirements.txt
```

### 2. Training (opsional, model sudah tersedia di `models/`)

```bash
python train_v2_fixed.py --data predictive_maintenance.csv
```

Training memakan waktu sekitar 30 sampai 70 menit di CPU. Hasilnya tersimpan di `models/`, termasuk `hasil_cv_per_fold.csv`, `hasil_cv_pooled.csv`, dan `hasil_sensitivitas_alpha.csv`.

### 3. Menjalankan aplikasi

```bash
streamlit run app.py
```

Unggah CSV dengan kolom `device`, `date`, dan `metric1` sampai `metric9`. Setiap perangkat harus punya riwayat minimal 7 hari. Enam baris pertama tiap perangkat tidak dinilai karena fitur 7 hari belum lengkap.

## Metode singkat

- Validasi silang 5-fold, dikelompokkan per perangkat (`StratifiedGroupKFold`), sehingga satu perangkat tidak muncul di data latih dan uji sekaligus.
- Threshold dipilih dari validasi internal sebagai batas 1% baris teratas (anggaran alarm).
- α hybrid dikunci ke 0,5. Sensitivitas α dari 0,0 sampai 1,0 dilaporkan terpisah.
- Metrik utama: PR-AUC dan Recall@1%. Accuracy tidak dipakai sebagai dasar perbandingan karena data sangat tidak seimbang.

## Hasil (rata-rata 3 kali training)

| Model | PR-AUC | Recall@1% |
|---|---|---|
| Autoencoder | 0,040 | 0,409 |
| Isolation Forest | 0,037 | 0,338 |
| Hybrid AE+IF (α = 0,5) | 0,066 | 0,451 |
| Random Forest | 0,073 | 0,384 |
| XGBoost | 0,057 | 0,290 |

Random Forest dan XGBoost hasilnya sama di setiap run karena deterministik. Autoencoder dan Hybrid sedikit berbeda antar run karena TensorFlow di CPU tidak sepenuhnya deterministik.

## Keterbatasan

- Failure hanya 106 kejadian, sehingga satu kejadian sudah mengubah Recall@1% sekitar 1%.
- Perbedaan antar model belum signifikan secara statistik. CI bootstrap antar model saling tumpang tindih.
- Data bersumber dari Kaggle dan belum diverifikasi asal-usulnya dari pengukuran nyata.
