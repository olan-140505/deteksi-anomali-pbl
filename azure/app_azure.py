# Aplikasi Streamlit: deteksi kegagalan perangkat dari telemetri per jam (dataset Azure).
# Input: CSV dengan kolom datetime, machineID, volt, rotate, pressure, vibration.
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from fitur_azure import harian, FEAT

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title='Deteksi Kegagalan Perangkat', layout='wide')
st.title('Deteksi Anomali Kegagalan Perangkat Berbasis Data Sensor Harian')
st.caption('Model: XGBoost. Threshold ditentukan dari anggaran alarm 1%.')

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, 'models', 'xgb_azure.joblib'))
    scaler = joblib.load(os.path.join(BASE, 'models', 'scaler_azure.joblib'))
    cfg = json.load(open(os.path.join(BASE, 'models', 'config_azure.json')))
    return model, scaler, cfg

model, scaler, cfg = load_artifacts()
st.write(f"Threshold: **{cfg['threshold']:.4f}** | fitur: {len(FEAT)} | "
         f"data latih: {cfg['jumlah_baris_latih']} mesin-hari")

file = st.file_uploader('Unggah CSV telemetri per jam', type='csv')
st.info('Kolom wajib: datetime, machineID, volt, rotate, pressure, vibration. '
        'Setiap mesin harus punya minimal 7 hari telemetri. Hari ke-1 sampai ke-6 tidak dinilai.')

if file is not None:
    tel = pd.read_csv(file)
    wajib = {'datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration'}
    if not wajib.issubset(tel.columns):
        st.error(f'Kolom kurang: {sorted(wajib - set(tel.columns))}')
        st.stop()
    daily = harian(tel)
    if daily.empty:
        st.error('Tidak ada hari yang bisa dinilai. Pastikan minimal 7 hari telemetri.')
        st.stop()
    skor = model.predict_proba(scaler.transform(daily[FEAT].values))[:, 1]
    hasil = daily[['machineID', 'date']].copy()
    hasil['date'] = hasil['date'].dt.date
    hasil['skor'] = np.round(skor, 4)
    hasil['status'] = np.where(skor >= cfg['threshold'], 'GAGAL', 'NORMAL')

    c1, c2, c3 = st.columns(3)
    c1.metric('Hari dinilai', len(hasil))
    c2.metric('Status GAGAL', int((hasil['status'] == 'GAGAL').sum()))
    c3.metric('Mesin', hasil['machineID'].nunique())

    st.subheader('Hasil per mesin dan hari')
    st.dataframe(hasil.style.apply(
        lambda r: ['background-color: #f8d7da' if r['status'] == 'GAGAL' else ''] * len(r), axis=1),
        use_container_width=True)
    st.download_button('Unduh hasil (CSV)', hasil.to_csv(index=False).encode('utf-8'),
                       file_name='hasil_deteksi_azure.csv', mime='text/csv')
