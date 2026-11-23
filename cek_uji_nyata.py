"""
cek_uji_nyata.py - Cek status GAGAL/Normal untuk CSV uji (format sama dengan aplikasi).

Memakai fungsi score_all dan load_artifacts dari uji_false_alarm.py,
jadi hasilnya sama dengan logika di app.py.

Jalankan (dari folder repo yang berisi models/):
    python cek_uji_nyata.py --data uji_nyata.csv
"""
import argparse

import pandas as pd

from pm_features import RAW, ROLL, prepare
from uji_false_alarm import load_artifacts, score_all

MODEL = "Hybrid AE+IF"   # model yang dipakai aplikasi


def main(args):
    raw = pd.read_csv(args.data)
    df = prepare(raw[["device", "date"] + RAW])
    art = load_artifacts()
    thr = art["cfg"]["thr"][MODEL]

    rows = []
    for dev in df["device"].unique():
        d = df[df["device"] == dev].reset_index(drop=True)
        s = score_all(d, art)[MODEL]
        d = d.assign(skor=s, status=["GAGAL" if v >= thr else "Normal" for v in s])
        # Sama seperti aplikasi: 6 baris pertama per perangkat belum 7 hari riwayat, tidak dinilai
        rows.append(d.iloc[ROLL - 1:])
    out = pd.concat(rows)
    print(f"Baris dinilai: {len(out)} dari {len(df)} (aturan aplikasi: 6 baris pertama per perangkat tidak dinilai)")

    print(f"Model: {MODEL} | threshold: {thr:.4f}")
    print(out[["device", "date", "failure", "skor", "status"]].to_string(index=False)
          if "failure" in out.columns else out[["device", "date", "skor", "status"]].to_string(index=False))
    out.to_csv("cek_uji_nyata_hasil.csv", index=False)
    print("\nHasil tersimpan di cek_uji_nyata_hasil.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="uji_nyata.csv")
    args, _ = parser.parse_known_args()
    main(args)
