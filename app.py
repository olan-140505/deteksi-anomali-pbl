"""
app_v2.py - Deteksi anomali dari riwayat harian per perangkat.

Jalankan (setelah train_v2.py selesai):
    streamlit run app_v2.py

Butuh: features.py dan folder models/ di folder yang sama.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from pm_features import RAW, ROLL, prepare, build_features

MODEL_DIR = Path("models")
REQUIRED = ["device", "date"] + RAW


@st.cache_resource
def load_artifacts():
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    return {
        "cfg": cfg,
        "ae": keras.models.load_model(MODEL_DIR / "autoencoder.keras", compile=False),
        "iforest": joblib.load(MODEL_DIR / "isolation_forest.joblib"),
        "scaler": joblib.load(MODEL_DIR / "scaler.joblib"),
        "rf": joblib.load(MODEL_DIR / "random_forest.joblib"),
        "xgb": joblib.load(MODEL_DIR / "xgboost.joblib"),
    }


def minmax(s, lo, hi):
    return np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def score_rows(df, art, model_name):
    """Skor semua baris dan tiga fitur paling menyimpang (|z| terbesar)."""
    cfg = art["cfg"]
    F = build_features(df)
    X = art["scaler"].transform(F[cfg["features"]].values)

    ae_raw = np.mean((X - art["ae"].predict(X, verbose=0)) ** 2, axis=1)
    if_raw = -art["iforest"].score_samples(X)
    ae_n = minmax(ae_raw, cfg["ae_lo"], cfg["ae_hi"])
    if_n = minmax(if_raw, cfg["if_lo"], cfg["if_hi"])
    a = cfg["alpha"]

    all_s = {
        "Autoencoder": ae_n,
        "Isolation Forest": if_n,
        "Hybrid AE+IF": a * ae_n + (1 - a) * if_n,
        "Random Forest": art["rf"].predict_proba(X)[:, 1],
        "XGBoost": art["xgb"].predict_proba(X)[:, 1],
    }

    names = np.array(cfg["features"])
    top3 = [", ".join(names[np.argsort(-np.abs(row))[:3]]) for row in X]
    return all_s[model_name], top3


def validate(raw):
    msgs = []
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        msgs.append("Kolom tidak lengkap, hilang: " + ", ".join(missing))
        return None, msgs

    df = raw[REQUIRED].copy()
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        msgs.append("Kolom date tidak bisa dibaca sebagai tanggal.")
        return None, msgs

    num = df[RAW].apply(pd.to_numeric, errors="coerce")
    bad = num.isna().any(axis=1) | df["date"].isna() | df["device"].isna()
    if bad.any():
        msgs.append(f"{int(bad.sum())} baris kosong atau tidak valid dilewati.")
    clean = pd.concat([df[["device", "date"]], num], axis=1)[~bad].reset_index(drop=True)
    if clean.empty:
        msgs.append("Tidak ada data valid untuk diproses.")
        return None, msgs
    return prepare(clean), msgs


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Deteksi Anomali Perangkat", layout="wide")
st.title("Deteksi Anomali Kegagalan Perangkat")
st.caption("Autoencoder + Isolation Forest (Hybrid) dibandingkan dengan Random Forest & XGBoost")

if not (MODEL_DIR / "config.json").exists():
    st.error("Model belum ada. Jalankan dulu train_v2.py.")
    st.stop()

art = load_artifacts()
cfg = art["cfg"]

with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox("Model", list(cfg["thr"].keys()), index=2)
    default_thr = float(cfg["thr"][model_name])
    thr = st.slider("Threshold", 0.0, 1.0, default_thr, 0.0001, format="%.4f",
                    key=f"thr_{model_name}",
                    help="Skor >= threshold dianggap Gagal (Anomali).")
    st.caption(f"Threshold default dari validasi: {default_thr:.4f}")
    st.caption(f"Alpha skor hybrid: {cfg['alpha']:.1f}")

st.write(
    "Unggah CSV berisi kolom **device**, **date**, dan **metric1** sampai **metric9**. "
    "Setiap perangkat harus punya riwayat minimal **7 hari**. Baris pertama 6 hari "
    "tiap perangkat tidak dinilai, karena fitur 7 hari belum lengkap."
)

file = st.file_uploader("Pilih berkas CSV", type="csv")
if file is not None:
    try:
        raw = pd.read_csv(file, sep=None, engine="python")
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

    df, msgs = validate(raw)
    for m in msgs:
        st.warning(m)
    if df is None:
        st.stop()

    s, top3 = score_rows(df, art, model_name)
    pos = df.groupby("device").cumcount().values
    scored = pos >= ROLL - 1

    if (~scored).sum():
        st.info(f"{int((~scored).sum())} baris tidak dinilai karena riwayat perangkatnya belum 7 hari.")
    if scored.sum() == 0:
        st.warning("Tidak ada perangkat dengan riwayat minimal 7 hari.")
        st.stop()

    out = df.loc[scored, ["device", "date"]].copy()
    out["skor"] = s[scored]
    out["status"] = np.where(out["skor"] >= thr, "GAGAL (Anomali)", "Normal")
    out["fitur_menyimpang"] = np.array(top3)[scored]

    c1, c2, c3 = st.columns(3)
    c1.metric("Baris dinilai", len(out))
    c2.metric("Terdeteksi gagal", int((out["skor"] >= thr).sum()))
    c3.metric("Persentase gagal", f"{(out['skor'] >= thr).mean() * 100:.2f}%")
    st.dataframe(out, use_container_width=True)
    st.download_button("Unduh hasil prediksi (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="hasil_prediksi.csv", mime="text/csv")
