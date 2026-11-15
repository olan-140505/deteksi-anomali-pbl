"""
features.py - Dipakai bersama oleh train_v2.py dan app_v2.py
agar fitur saat training dan saat aplikasi selalu sama.
"""
import numpy as np
import pandas as pd

RAW = [f"metric{i}" for i in range(1, 10)]
BASE = [m for m in RAW if m != "metric8"]   # metric8 identik dengan metric7
ROLL = 7                                    # jendela 7 hari per perangkat
FEAT = ([f"{m}_log" for m in BASE]
        + [f"{m}_mean7" for m in BASE]
        + [f"{m}_std7" for m in BASE])


def prepare(df):
    """Urutkan per perangkat dan tanggal, buang duplikat, reset index."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates().sort_values(["device", "date"]).reset_index(drop=True)
    return df


def build_features(df):
    """
    Fitur hanya memakai data hari itu dan sebelumnya (rolling ke belakang),
    dihitung terpisah untuk tiap perangkat. Index df harus 0..n-1.
    """
    out = pd.DataFrame(index=df.index)
    logged = pd.DataFrame(np.log1p(df[BASE].clip(lower=0).values),
                          columns=BASE, index=df.index)
    logged["device"] = df["device"].values
    for m in BASE:
        out[f"{m}_log"] = logged[m]
        g = logged.groupby("device")[m]
        out[f"{m}_mean7"] = g.transform(lambda s: s.rolling(ROLL, min_periods=1).mean())
        out[f"{m}_std7"] = g.transform(lambda s: s.rolling(ROLL, min_periods=2).std()).fillna(0.0)
    return out
