# Pembuatan fitur harian untuk dataset Microsoft Azure Predictive Maintenance.
# Dipakai bersama oleh skrip pelatihan dan aplikasi, supaya logikanya sama persis.
import pandas as pd

SENS = ['volt', 'rotate', 'pressure', 'vibration']
BASE = [f'{s}_{st}' for s in SENS for st in ('mean', 'std')]
FEAT = BASE + [c + '_rmean7' for c in BASE] + [c + '_rstd7' for c in BASE]


def harian(tel):
    """Telemetri per jam -> fitur harian per mesin (24 fitur).
    Hari ke-1 sampai ke-6 tiap mesin dibuang karena histori 7 hari belum lengkap.
    Rolling hanya memakai hari ini dan sebelumnya (kausal)."""
    t = tel.copy()
    t['date'] = pd.to_datetime(t['datetime']).dt.normalize()
    agg = t.groupby(['machineID', 'date'])[SENS].agg(['mean', 'std']).reset_index()
    agg.columns = ['machineID', 'date'] + BASE
    agg[BASE] = agg[BASE].fillna(0)
    agg = agg.sort_values(['machineID', 'date']).reset_index(drop=True)
    g = agg.groupby('machineID')[BASE]
    rm = g.rolling(7).mean().reset_index(level=0, drop=True)
    rs = g.rolling(7).std().reset_index(level=0, drop=True)
    rm.columns = [c + '_rmean7' for c in BASE]
    rs.columns = [c + '_rstd7' for c in BASE]
    agg = pd.concat([agg, rm, rs], axis=1)
    agg['hari_ke'] = agg.groupby('machineID').cumcount() + 1
    return agg[agg['hari_ke'] >= 7].dropna(subset=FEAT).reset_index(drop=True)
