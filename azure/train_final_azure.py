# Latih model final XGBoost untuk aplikasi (dataset Azure harian).
# Threshold 1% dari skor out-of-fold. Juga membuat file uji untuk aplikasi.
# Letakkan fitur_azure.py di folder yang sama. Jalankan di Google Colab.
import os, json, subprocess, sys
try:
    import kagglehub
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'], check=True)
    import kagglehub
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from fitur_azure import harian, FEAT

BUDGET = 0.01
os.makedirs('models', exist_ok=True)
os.makedirs('pengujian', exist_ok=True)

path = kagglehub.dataset_download('arnabbiswas1/microsoft-azure-predictive-maintenance')
tel = pd.read_csv(os.path.join(path, 'PdM_telemetry.csv'))
fail = pd.read_csv(os.path.join(path, 'PdM_failures.csv'))
fail['date'] = pd.to_datetime(fail['datetime']).dt.normalize()

# Label: 1 jika ada kegagalan pada hari itu
daily = harian(tel)
fd = fail[['machineID', 'date']].drop_duplicates().assign(failure=1)
daily = daily.merge(fd, on=['machineID', 'date'], how='left')
daily['failure'] = daily['failure'].fillna(0).astype(int)
X = daily[FEAT].values.astype(np.float32)
y = daily['failure'].values
groups = daily['machineID'].values
print('baris', len(daily), '| failure', int(y.sum()), f'({y.mean():.2%})')

def make_xgb(ytr):
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=pos_w,
                         eval_metric='logloss', random_state=42)

# 1) Skor out-of-fold untuk menentukan threshold
oof = np.zeros(len(y))
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in skf.split(X, y, groups):
    sc = StandardScaler().fit(X[tr])
    m = make_xgb(y[tr]).fit(sc.transform(X[tr]), y[tr])
    oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
thr = float(np.quantile(oof, 1 - BUDGET))
print('PR-AUC OOF:', round(average_precision_score(y, oof), 3), '| threshold:', round(thr, 4))

# 2) Model final pada 100% data
scaler = StandardScaler().fit(X)
model = make_xgb(y).fit(scaler.transform(X), y)
joblib.dump(model, 'models/xgb_azure.joblib')
joblib.dump(scaler, 'models/scaler_azure.joblib')
cfg = {'model': 'XGBoost', 'fitur': FEAT, 'threshold': thr, 'anggaran_alarm': BUDGET,
       'jumlah_baris_latih': int(len(daily)), 'jumlah_failure': int(y.sum())}
json.dump(cfg, open('models/config_azure.json', 'w'), indent=2)
pd.DataFrame({'machineID': groups, 'date': daily['date'], 'failure': y, 'skor_oof': oof}) \
  .to_csv('models/oof_azure.csv', index=False)

# 3) File uji untuk aplikasi: 4 mesin, 14 hari telemetri per jam
#    3 mesin gagal (14 hari yang berakhir pada hari kegagalan) + 1 mesin normal (14 hari pertama)
tel['date'] = pd.to_datetime(tel['datetime']).dt.normalize()
fail_last = fail.sort_values('date').groupby('machineID').tail(1).sort_values('date', ascending=False)
pilih = []
for _, r in fail_last.head(3).iterrows():
    end = r['date']; start = end - pd.Timedelta(days=13)
    pilih.append(tel[(tel['machineID'] == r['machineID']) & (tel['date'] >= start) & (tel['date'] <= end)])
mesin_normal = sorted(set(tel['machineID']) - set(fail['machineID']))[0]
t0 = tel['date'].min()
pilih.append(tel[(tel['machineID'] == mesin_normal) & (tel['date'] < t0 + pd.Timedelta(days=14))])
uji = pd.concat(pilih).drop(columns=['date'])
uji.to_csv('pengujian/uji_sampel_azure.csv', index=False)
print('file uji:', len(uji), 'baris,', uji['machineID'].nunique(), 'mesin')
print('config:', json.dumps({k: v for k, v in cfg.items() if k != 'fitur'}, indent=2))
