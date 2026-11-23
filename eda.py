"""
eda.py - Exploratory Data Analysis untuk predictive_maintenance.csv.

Menghasilkan ringkasan di konsol dan grafik di folder figures/.

Jalankan:
    python eda.py --data predictive_maintenance.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # simpan ke file, tidak perlu layar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = [f"metric{i}" for i in range(1, 10)]
FIG_DIR = Path("figures")


def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def summarize(df):
    print("=== STRUKTUR DATA ===")
    print(f"Baris: {len(df):,} | Kolom: {df.shape[1]}")
    print(f"Rentang tanggal: {df['date'].min().date()} s.d. {df['date'].max().date()}")
    print(f"Jumlah perangkat: {df['device'].nunique():,}")
    print(f"Baris duplikat: {df.duplicated().sum():,}")
    print(f"Nilai hilang per kolom:\n{df.isna().sum().to_string()}\n")

    n_fail = int(df["failure"].sum())
    print("=== TARGET ===")
    print(f"Failure: {n_fail:,} baris ({n_fail / len(df):.4%})")
    print(f"Normal : {len(df) - n_fail:,} baris\n")

    per_dev = df.groupby("device").agg(hari=("date", "size"), fail=("failure", "sum"))
    print("=== PER PERANGKAT ===")
    print(per_dev["hari"].describe().round(1).to_string())
    print(f"Perangkat dengan >=1 failure: {(per_dev['fail'] > 0).sum()}")
    print(f"Perangkat dengan >1 failure : {(per_dev['fail'] > 1).sum()}\n")

    print("=== STATISTIK METRIK ===")
    print(df[RAW].describe().T.round(3).to_string(), "\n")
    return per_dev


def plot_target(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["failure"].value_counts().sort_index()
    ax.bar(["Normal", "Failure"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Jumlah baris (skala log)")
    ax.set_title("Distribusi label: sangat tidak seimbang")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_distribusi_label.png", dpi=150)
    plt.close(fig)


def plot_histograms(df):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, m in zip(axes.ravel(), RAW):
        x = df[m].clip(lower=0)
        ax.hist(np.log1p(x), bins=60, color="#55A868")
        ax.set_title(f"{m} (log1p)")
    fig.suptitle("Distribusi metrik (skala log1p karena rentang nilai lebar)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_histogram_metrik.png", dpi=150)
    plt.close(fig)


def plot_correlation(df):
    corr = df[RAW].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(RAW)), RAW, rotation=45)
    ax.set_yticks(range(len(RAW)), RAW)
    for i in range(len(RAW)):
        for j in range(len(RAW)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Korelasi Pearson antar metrik")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_korelasi.png", dpi=150)
    plt.close(fig)
    # metric8 identik dengan metric7 (dibuang di pm_features.py)
    same = np.allclose(df["metric7"], df["metric8"], equal_nan=True)
    print(f"metric7 == metric8 untuk semua baris: {same}")


def plot_fail_vs_normal(df):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, m in zip(axes.ravel(), RAW):
        data = [np.log1p(df.loc[df["failure"] == k, m].clip(lower=0)) for k in (0, 1)]
        ax.boxplot(data, tick_labels=["Normal", "Failure"], showfliers=False)
        ax.set_title(m)
    fig.suptitle("Metrik saat normal vs failure (log1p)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_normal_vs_failure.png", dpi=150)
    plt.close(fig)


def plot_device_example(df, per_dev):
    fail_dev = per_dev.index[per_dev["fail"] > 0]
    if len(fail_dev) == 0:
        return
    dev = fail_dev[0]
    d = df[df["device"] == dev].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    for m in ["metric1", "metric2", "metric3"]:
        ax.plot(d["date"], np.log1p(d[m].clip(lower=0)), label=m)
    for t in d.loc[d["failure"] == 1, "date"]:
        ax.axvline(t, color="red", ls="--", alpha=0.7)
    ax.set_title(f"Contoh perangkat {dev} dengan failure (garis merah)")
    ax.set_ylabel("log1p(metrik)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_contoh_perangkat.png", dpi=150)
    plt.close(fig)


def main(args):
    FIG_DIR.mkdir(exist_ok=True)
    df = load(args.data)
    per_dev = summarize(df)
    plot_target(df)
    plot_histograms(df)
    plot_correlation(df)
    plot_fail_vs_normal(df)
    plot_device_example(df, per_dev)
    print(f"Grafik tersimpan di folder '{FIG_DIR}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="predictive_maintenance.csv")
    args, _ = parser.parse_known_args()
    main(args)
