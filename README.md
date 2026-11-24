# Deteksi Anomali Kegagalan Perangkat

Proyek PBL Pembelajaran Mesin, Kelompok 14, Kelas A.

Repositori ini membandingkan lima model untuk mendeteksi kegagalan perangkat dari data telemetri harian: Autoencoder, Isolation Forest, Hybrid AE+IF (α = 0,5), Random Forest, dan XGBoost. Model yang dipakai di aplikasi adalah **Hybrid AE+IF**.

**Aplikasi (live):** https://deteksi-anomali-pbl-bkcrhl46gqxec9q92s4ghm.streamlit.app

## Dataset

- Sumber: Kaggle, [hiimanshuagarwal/predictive-maintenance-dataset](https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset), lisensi MIT.
- 124.494 baris (124.493 setelah membuang 1 duplikat), 1.169 perangkat, rentang tanggal 1 Januari sampai 2 November 2015.
- 106 failure (0,085%). Setiap perangkat yang gagal hanya gagal satu kali.
- Kolom: `device`, `date`, `failure`, dan `metric1` sampai `metric9`.
- **Keterbatasan:** sumber asli dan makna `metric1–9` tidak tercantum di halaman dataset, sehingga makna setiap sensor tidak diketahui. `metric7` dan `metric8` identik, jadi `metric8` dibuang.
- Dataset tidak disertakan di repositori. Jika `predictive_maintenance.csv` tidak ada, `train_v2_fixed.py` mengunduhnya dari Kaggle lewat `kagglehub` (perlu kredensial Kaggle).

## Struktur

| File | Isi |
|---|---|
| `eda.py` | Ringkasan data dan grafik EDA (disimpan di `figures/`) |
| `pm_features.py` | Fitur per perangkat: log1p, rata-rata 7 hari, dan std 7 hari (kausal) |
| `train_v2_fixed.py` | Training, validasi silang 5-fold dikelompokkan per perangkat, evaluasi, dan model final |
| `app.py` | Aplikasi Streamlit untuk memprediksi dari CSV riwayat harian |
| `uji_false_alarm.py` | Hitung false alarm pada perangkat normal dengan aturan 7 hari |
| `cek_uji_nyata.py` | Cek status GAGAL/Normal untuk satu CSV, untuk dibandingkan dengan aplikasi |
| `tests/test_pm_features.py` | Uji untuk fitur (tanpa kebocoran masa depan, per perangkat, dan jendela 7 hari) |
| `figures/` | Grafik EDA |
| `models/` | Model final, `config.json`, dan hasil evaluasi |
| `pengujian/` | Hasil uji false alarm dan sampel uji untuk aplikasi |
| `requirements.txt` | Paket dengan versi yang sudah diuji |

## Cara menjalankan

### 1. Instalasi

```bash
pip install -r requirements.txt
```

### 2. Tes dan EDA (opsional)

```bash
python -m unittest discover -s tests -v
python eda.py --data predictive_maintenance.csv
```

### 3. Training (opsional, model sudah tersedia di `models/`)

```bash
python train_v2_fixed.py --data predictive_maintenance.csv
```

Training memakan waktu sekitar 30 sampai 70 menit di CPU. Hasilnya tersimpan di `models/`.

### 4. Menjalankan aplikasi

```bash
streamlit run app.py
```

Unggah CSV dengan kolom `device`, `date`, dan `metric1` sampai `metric9`. Setiap perangkat harus punya riwayat minimal 7 hari. Enam baris pertama tiap perangkat tidak dinilai karena fitur 7 hari belum lengkap.

### 5. Uji false alarm

```bash
python uji_false_alarm.py --data predictive_maintenance.csv --hari 30 --banyak 1200
```

## Metode

- **Fitur:** log1p dari metrik (nilai negatif dipotong ke 0), rata-rata dan std 7 hari per perangkat. Fitur hanya memakai data hari itu dan sebelumnya.
- **Autoencoder dan Isolation Forest** dilatih hanya pada data normal. Skor keduanya dinormalkan dengan min-max dari data latih.
- **Hybrid AE+IF** adalah kombinasi linear `α · AE + (1 − α) · IF` dengan α = 0,5 yang dikunci. Sensitivitas α dari 0,0 sampai 1,0 dilaporkan terpisah.
- **Random Forest dan XGBoost** dilatih pada semua data dengan penanganan ketidakseimbangan (`class_weight` dan `scale_pos_weight`).
- **Validasi:** `StratifiedGroupKFold` 5-fold dikelompokkan per perangkat, sehingga satu perangkat tidak muncul di data latih dan uji sekaligus. Di dalam setiap fold, data latih dibagi lagi (dikelompokkan per perangkat) untuk threshold. Scaler hanya di-fit pada data latih setiap fold.
- **Threshold:** anggaran alarm 1%, yaitu kuantil 99% dari skor validasi. Untuk model final, threshold diambil dari kuantil 99% skor out-of-fold.
- **Model final** dilatih pada **100% data**. Threshold-nya berasal dari skor out-of-fold, bukan dari data validasi terpisah, karena model final tidak punya data validasi sendiri.
- **Metrik utama:** PR-AUC dan Recall@1%. Accuracy tidak dipakai sebagai dasar perbandingan karena data sangat tidak seimbang.

## Hasil

### Validasi silang 5-fold (rata-rata ± std antar fold)

| Model | PR-AUC | ROC-AUC | Recall@1% |
|---|---|---|---|
| Autoencoder | 0,041 ± 0,014 | 0,867 ± 0,034 | 0,434 ± 0,039 |
| Isolation Forest | 0,037 ± 0,019 | 0,857 ± 0,021 | 0,338 ± 0,121 |
| Hybrid AE+IF (α = 0,5) | 0,060 ± 0,013 | 0,866 ± 0,018 | 0,447 ± 0,057 |
| Random Forest | 0,073 ± 0,064 | 0,865 ± 0,017 | 0,384 ± 0,138 |
| XGBoost | 0,057 ± 0,030 | 0,808 ± 0,035 | 0,290 ± 0,149 |

### Skor gabungan out-of-fold (OOF) dengan CI 95% bootstrap

| Model | PR-AUC (OOF) | CI 95% | Recall@1% (OOF) |
|---|---|---|---|
| Autoencoder | 0,036 | 0,022 – 0,056 | 0,387 |
| Isolation Forest | 0,027 | 0,016 – 0,043 | 0,321 |
| Hybrid AE+IF | 0,053 | 0,032 – 0,088 | 0,415 |
| Random Forest | 0,038 | 0,021 – 0,081 | 0,368 |
| XGBoost | 0,042 | 0,016 – 0,087 | 0,274 |

Baseline PR-AUC acak = 0,00085 (sama dengan proporsi failure).

### False alarm pada perangkat normal (perkiraan out-of-fold)

Dihitung dari jumlah FP dan TN validasi silang, lalu dikonversi ke alarm per 30 hari per perangkat:

| Model | Alarm per 30 hari per perangkat |
|---|---|
| Autoencoder | 0,31 |
| Hybrid AE+IF | 0,30 |
| Isolation Forest | 0,30 |
| Random Forest | 0,33 |
| XGBoost | 0,34 |

Perkiraan ini memakai asumsi satu baris sama dengan satu hari dan belum memisahkan 6 baris pertama. Angka dalam sampel dari `uji_false_alarm.py` tidak dipakai sebagai hasil utama, karena model final sudah melihat perangkat-perangkat itu saat training.

## Pemilihan model untuk aplikasi

**Hybrid AE+IF** dipilih untuk aplikasi karena:
- Skor out-of-fold PR-AUC tertinggi (0,053) dan Recall@1% tertinggi (0,415), walaupun selisihnya dengan model lain masih berada dalam rentang CI.
- Hanya memakai data normal, sehingga tidak bergantung pada label kegagalan yang sangat jarang (106 kejadian).
- Alarm per 30 hari sekitar 0,30 per perangkat, sesuai target anggaran 1%.

Random Forest memiliki rata-rata PR-AUC per fold tertinggi (0,073), tetapi variasinya sangat besar (± 0,064), dan skor out-of-fold-nya lebih rendah dari Hybrid. XGBoost memiliki Recall@1% terendah.

## Keterbatasan

- Failure hanya 106 kejadian, sehingga satu kejadian sudah mengubah Recall@1% sekitar 1%.
- Perbedaan antar model tidak signifikan secara statistik. CI bootstrap antar model saling tumpang tindih.
- Data bersumber dari Kaggle dan makna `metric1–9` tidak diketahui, sehingga hasil ini adalah benchmark pada data tersebut.
- Threshold model final diambil dari skor out-of-fold, sedangkan skala skor antar fold sedikit berbeda. Pendekatan ini perlu dijelaskan di laporan.
- TensorFlow di CPU tidak sepenuhnya deterministik, sehingga Autoencoder dan Hybrid bisa sedikit berbeda antar run. Random Forest dan XGBoost deterministik.
