"""
train_v2_fixed.py - Training dengan validasi silang per perangkat.

Perubahan dari versi sebelumnya:
  - Alpha hybrid dikunci ke ALPHA_FIXED (0.5). Sensitivitas alpha 0.0-1.0 dilaporkan terpisah.
  - Threshold dipilih berdasarkan anggaran alarm (1% baris teratas di validasi),
    bukan F1 maksimum, karena jumlah failure di validasi terlalu sedikit.

Jalankan:
    python train_v2_fixed.py --data predictive_maintenance.csv

Butuh: pm_features.py di folder yang sama.
"""
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pm_features import RAW, FEAT, prepare, build_features

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ALPHA_FIXED = 0.5                                  # alpha hybrid dikunci (tidak dipilih dari validasi)
ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)   # grid hanya untuk analisis sensitivitas
N_FOLDS = 5
TOP_PCT = 0.01                                     # anggaran alarm: 1% baris teratas
N_BOOT = 1000
RB = f"Recall@{TOP_PCT * 100:g}%"                 # nama kolom recall di anggaran alarm
MODEL_NAMES = ["Autoencoder", "Isolation Forest", "Hybrid AE+IF", "Random Forest", "XGBoost"]
METRIC_COLS = ["PR-AUC", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1", RB]
MODEL_DIR = Path("models")
KAGGLE_DATASET = "hiimanshuagarwal/predictive-maintenance-dataset"


# ---------------------------------------------------------------- Data
def resolve_data(path):
    """Pakai CSV lokal jika ada; kalau tidak, unduh dari Kaggle lewat kagglehub."""
    if Path(path).exists():
        return path
    import glob
    import shutil
    import kagglehub

    print(f"'{path}' tidak ditemukan, mengunduh dari Kaggle...")
    folder = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = glob.glob(f"{folder}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV di {folder}")
    shutil.copy(csv_files[0], path)
    return path


def load_data(path):
    raw = pd.read_csv(path)
    print("Shape awal      :", raw.shape)
    print("Missing value   :", int(raw.isna().sum().sum()))
    print("Baris duplikat  :", int(raw.duplicated().sum()))

    df = prepare(raw)
    assert (df["metric7"] == df["metric8"]).all(), "metric7 dan metric8 tidak identik"
    df = pd.concat([df, build_features(df)], axis=1)

    print(f"Baris setelah pembersihan: {len(df)} | perangkat: {df['device'].nunique()}")
    print(f"Failure: {int(df['failure'].sum())} ({df['failure'].mean() * 100:.3f}%)")
    return df


# ---------------------------------------------------------------- Model
def minmax(s, lo, hi):
    return np.clip((s - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def recon_error(ae, X):
    return np.mean((X - ae.predict(X, batch_size=2048, verbose=0)) ** 2, axis=1)


def build_autoencoder(n_in):
    inp = keras.Input(shape=(n_in,))
    x = keras.layers.Dense(16, activation="relu")(inp)
    x = keras.layers.Dense(8, activation="relu")(x)
    z = keras.layers.Dense(4, activation="relu")(x)          # bottleneck
    x = keras.layers.Dense(8, activation="relu")(z)
    x = keras.layers.Dense(16, activation="relu")(x)
    out = keras.layers.Dense(n_in, activation="linear")(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae


def budget_threshold(s):
    """Threshold = skor pada kuantil (1 - TOP_PCT): tepat 1% baris teratas di atas threshold."""
    return float(np.quantile(s, 1 - TOP_PCT))


def fit_all(Xf, yf):
    """Latih AE dan IF hanya pada data normal; RF dan XGBoost pada semua data."""
    normal = Xf[yf == 0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(normal))
    cut = int(0.85 * len(normal))
    tr_n, va_n = normal[perm[:cut]], normal[perm[cut:]]

    ae = build_autoencoder(normal.shape[1])
    ae.fit(tr_n, tr_n, validation_data=(va_n, va_n), epochs=100, batch_size=256,
           callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
           verbose=0)

    iforest = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(normal)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED).fit(Xf, yf)

    neg, pos = int((yf == 0).sum()), int((yf == 1).sum())
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=neg / pos, eval_metric="aucpr",
                        n_jobs=-1, random_state=SEED).fit(Xf, yf)

    rn = recon_error(ae, normal)
    ri = -iforest.score_samples(normal)
    norm = {"ae_lo": float(rn.min()), "ae_hi": float(np.percentile(rn, 99.9)),
            "if_lo": float(ri.min()), "if_hi": float(np.percentile(ri, 99.9))}
    return {"ae": ae, "iforest": iforest, "rf": rf, "xgb": xgb, "norm": norm}


def base_scores(m, X):
    n = m["norm"]
    ae_n = minmax(recon_error(m["ae"], X), n["ae_lo"], n["ae_hi"])
    if_n = minmax(-m["iforest"].score_samples(X), n["if_lo"], n["if_hi"])
    return ae_n, if_n


def all_scores(m, X, alpha):
    ae_n, if_n = base_scores(m, X)
    return {
        "Autoencoder": ae_n,
        "Isolation Forest": if_n,
        "Hybrid AE+IF": alpha * ae_n + (1 - alpha) * if_n,
        "Random Forest": m["rf"].predict_proba(X)[:, 1],
        "XGBoost": m["xgb"].predict_proba(X)[:, 1],
    }


def run_fold(tr, te=None):
    """
    Latih semua model pada 'tr'. Threshold dipilih dari validasi internal yang
    dipisah per perangkat dari 'tr' (tidak menyentuh 'te'). Alpha tetap ALPHA_FIXED.
    """
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    fi, vi = next(inner.split(tr, tr["failure"], tr["device"]))
    fit, val = tr.iloc[fi], tr.iloc[vi]

    scaler = StandardScaler().fit(fit[FEAT].values)     # fit HANYA pada data latih
    Xf, yf = scaler.transform(fit[FEAT].values), fit["failure"].values
    Xv = scaler.transform(val[FEAT].values)

    m = fit_all(Xf, yf)

    sv = all_scores(m, Xv, ALPHA_FIXED)
    thr = {n: budget_threshold(sv[n]) for n in MODEL_NAMES}

    out = {"m": m, "scaler": scaler, "alpha": ALPHA_FIXED, "thr": thr}
    if te is not None:
        Xt = scaler.transform(te[FEAT].values)
        out["scores"] = all_scores(m, Xt, ALPHA_FIXED)
        out["base_te"] = base_scores(m, Xt)             # (AE, IF) untuk analisis sensitivitas
    return out


# ---------------------------------------------------------------- Evaluasi
def recall_at_budget(y, s):
    k = max(1, math.ceil(TOP_PCT * len(s)))
    top = np.argsort(-s)[:k]
    return y[top].sum() / max(1, y.sum())


def evaluate(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": average_precision_score(y, s),
        "ROC-AUC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        RB: recall_at_budget(y, s),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def bootstrap_ci(y, s, rng):
    """CI 95% untuk PR-AUC dengan resampling baris (bootstrap)."""
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        vals.append(average_precision_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- Main
def main(args):
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_data(resolve_data(args.data))
    y_all = df["failure"].values
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = {n: np.zeros(len(df)) for n in MODEL_NAMES}     # skor out-of-fold
    oof_ae = np.zeros(len(df))
    oof_if = np.zeros(len(df))
    fold_rows = []
    for k, (tr_i, te_i) in enumerate(sgk.split(df, y_all, df["device"].values), 1):
        print(f"\n=== Fold {k}/{N_FOLDS} | test: {len(te_i)} baris, {int(y_all[te_i].sum())} failure ===")
        res = run_fold(df.iloc[tr_i], df.iloc[te_i])
        for n in MODEL_NAMES:
            oof[n][te_i] = res["scores"][n]
            row = evaluate(y_all[te_i], res["scores"][n], res["thr"][n])
            fold_rows.append({"fold": k, "model": n, **row})
        oof_ae[te_i], oof_if[te_i] = res["base_te"]

    # --- Ringkasan per fold (mean ± std)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(MODEL_DIR / "hasil_cv_per_fold.csv", index=False)
    summary = []
    for n in MODEL_NAMES:
        sub = fold_df[fold_df["model"] == n]
        row = {"Model": n}
        for c in METRIC_COLS:
            row[c] = f"{sub[c].mean():.3f} ± {sub[c].std():.3f}"
        summary.append(row)
    print("\n=== RINGKASAN VALIDASI SILANG (mean ± std antar fold) ===")
    print(pd.DataFrame(summary).to_string(index=False))

    # --- Skor gabungan out-of-fold + CI bootstrap
    rng = np.random.default_rng(SEED)
    pooled = []
    for n in MODEL_NAMES:
        ci = bootstrap_ci(y_all, oof[n], rng)
        pooled.append({
            "Model": n,
            "PR-AUC (OOF)": average_precision_score(y_all, oof[n]),
            "CI95 bawah": ci[0],
            "CI95 atas": ci[1],
            f"{RB} (OOF)": recall_at_budget(y_all, oof[n]),
        })
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(MODEL_DIR / "hasil_cv_pooled.csv", index=False)
    print("\n=== SKOR GABUNGAN OUT-OF-FOLD ===")
    print(pooled_df.round(4).to_string(index=False))
    print(f"Baseline PR-AUC acak = {y_all.mean():.5f}")

    # --- Sensitivitas alpha (hanya analisis; alpha yang dipakai tetap ALPHA_FIXED)
    sens = []
    for a in ALPHAS:
        s = a * oof_ae + (1 - a) * oof_if
        sens.append({"alpha": a, "PR-AUC (OOF)": average_precision_score(y_all, s),
                     f"{RB} (OOF)": recall_at_budget(y_all, s)})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(MODEL_DIR / "hasil_sensitivitas_alpha.csv", index=False)
    print("\n=== SENSITIVITAS ALPHA HYBRID (OOF) ===")
    print(sens_df.round(4).to_string(index=False))

    # --- Model final: dilatih pada 100% data.
    #     Threshold diambil dari kuantil 99% skor out-of-fold (CV), karena model final
    #     tidak punya data validasi terpisah. Ini pendekatan anggaran 1%, sama seperti CV.
    print("\n=== MELATIH MODEL FINAL (100% data) ===")
    scaler = StandardScaler().fit(df[FEAT].values)
    m = fit_all(scaler.transform(df[FEAT].values), y_all)
    thr = {n: budget_threshold(oof[n]) for n in MODEL_NAMES}
    flagged = {n: float((oof[n] >= thr[n]).mean()) for n in MODEL_NAMES}
    print("Proporsi baris ditandai pada OOF (harus ~1%):", {n: round(v, 4) for n, v in flagged.items()})

    m["ae"].save(MODEL_DIR / "autoencoder.keras")
    joblib.dump(m["iforest"], MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(m["rf"], MODEL_DIR / "random_forest.joblib")
    joblib.dump(m["xgb"], MODEL_DIR / "xgboost.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    cfg = {"features": FEAT, "alpha": ALPHA_FIXED, "thr": thr, **m["norm"]}
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Alpha: {ALPHA_FIXED} | Threshold: {thr}")
    print(f"Artefak tersimpan di folder '{MODEL_DIR}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="predictive_maintenance.csv")
    args, _ = parser.parse_known_args()   # abaikan argumen tambahan dari Colab/Jupyter
    main(args)
