"""
uji_false_alarm.py - Hitung false alarm pada perangkat yang tidak pernah gagal.

Perangkat dipilih otomatis: tanpa failure dan punya minimal --hari baris.
Skor dihitung dengan logika yang sama seperti app.py, lalu dihitung berapa hari
yang ditandai GAGAL oleh setiap model.

Jalankan (setelah models/ tersedia):
    python uji_false_alarm.py --data predictive_maintenance.csv --hari 30
    python uji_false_alarm.py --data predictive_maintenance.csv --banyak 20
    python uji_false_alarm.py --data predictive_maintenance.csv --device S1F0XXXX

Hasil: ringkasan di konsol dan pengujian/false_alarm_<device>.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from pm_features import build_features, prepare

MODEL_DIR = Path("models")
OUT_DIR = Path("pengujian")
MODELS = ["Autoencoder", "Isolation Forest", "Hybrid AE+IF", "Random Forest", "XGBoost"]


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


def score_all(df, art):
    """Skor semua model untuk setiap baris df (df sudah lewat prepare)."""
    cfg = art["cfg"]
    F = build_features(df)
    X = art["scaler"].transform(F[cfg["features"]].values)
    ae_raw = np.mean((X - art["ae"].predict(X, verbose=0)) ** 2, axis=1)
    if_raw = -art["iforest"].score_samples(X)
    ae_n = minmax(ae_raw, cfg["ae_lo"], cfg["ae_hi"])
    if_n = minmax(if_raw, cfg["if_lo"], cfg["if_hi"])
    a = cfg["alpha"]
    return {
        "Autoencoder": ae_n,
        "Isolation Forest": if_n,
        "Hybrid AE+IF": a * ae_n + (1 - a) * if_n,
        "Random Forest": art["rf"].predict_proba(X)[:, 1],
        "XGBoost": art["xgb"].predict_proba(X)[:, 1],
    }


def pick_devices(df, hari, banyak, device=None):
    """Perangkat normal (tanpa failure) dengan minimal `hari` baris, urut dari riwayat terpanjang."""
    if device:
        return [device]
    g = df.groupby("device").agg(n=("date", "size"), fail=("failure", "sum"))
    ok = g[(g["fail"] == 0) & (g["n"] >= hari)].sort_values("n", ascending=False)
    if ok.empty:
        raise SystemExit(f"Tidak ada perangkat normal dengan minimal {hari} hari.")
    return list(ok.index[:banyak])


def main(args):
    OUT_DIR.mkdir(exist_ok=True)
    df = prepare(pd.read_csv(args.data))
    devices = pick_devices(df, args.hari, args.banyak, args.device)
    art = load_artifacts()
    thr = art["cfg"]["thr"]

    totals = {n: {"alarm": 0, "hari": 0, "perangkat_alarm": 0} for n in MODELS}
    detail = []
    for dev in devices:
        full = df[df["device"] == dev].reset_index(drop=True)
        scores = score_all(full, art)
        w = slice(0, args.hari)
        for name in MODELS:
            alarm = scores[name][w] >= thr[name]
            totals[name]["alarm"] += int(alarm.sum())
            totals[name]["hari"] += int(len(alarm))
            totals[name]["perangkat_alarm"] += int(alarm.any())
        part = full.iloc[w][["device", "date"]].copy()
        for name in MODELS:
            part[f"GAGAL_{name}"] = scores[name][w] >= thr[name]
        detail.append(part)

    pd.concat(detail).to_csv(OUT_DIR / "false_alarm_semua.csv", index=False)
    rows = []
    for name in MODELS:
        t = totals[name]
        rows.append({"model": name, "perangkat": len(devices), "hari_total": t["hari"],
                     "alarm_total": t["alarm"],
                     "alarm_per_30_hari_per_perangkat": round(t["alarm"] * 30 / max(t["hari"], 1), 3),
                     "perangkat_dengan_alarm": t["perangkat_alarm"],
                     "threshold": round(thr[name], 4)})
    summary = pd.DataFrame(rows)
    print(f"Jumlah perangkat normal diuji: {len(devices)} | hari per perangkat: {args.hari}")
    print(summary.to_string(index=False))
    print(f"\nDetail per baris tersimpan di {OUT_DIR / 'false_alarm_semua.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="predictive_maintenance.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--banyak", type=int, default=20, help="jumlah perangkat normal yang diuji")
    parser.add_argument("--hari", type=int, default=30)
    args, _ = parser.parse_known_args()
    main(args)
