import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from config import SKAB_DIZINI, BATADAL_DIZINI

def skab_yukle():
    tum_veriler = []
    vana1_dizini = os.path.join(SKAB_DIZINI, "valve1")
    if os.path.exists(vana1_dizini):
        for f in os.listdir(vana1_dizini):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(vana1_dizini, f), sep=';')
                df["source_group"] = "valve1"
                df["source_file"] = f
                tum_veriler.append(df)
    vana2_dizini = os.path.join(SKAB_DIZINI, "valve2")
    if os.path.exists(vana2_dizini):
        for f in os.listdir(vana2_dizini):
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(vana2_dizini, f), sep=';')
                df["source_group"] = "valve2"
                df["source_file"] = f
                tum_veriler.append(df)
    birlesik_veri = pd.concat(tum_veriler, ignore_index=True)
    return birlesik_veri

def skab_on_isle(df):
    df = df.copy()
    meta_sutunlar = ["datetime", "changepoint", "source_group", "source_file", "anomaly"]
    oznitelik_sutunlari = [col for col in df.columns if col not in meta_sutunlar]
    df[oznitelik_sutunlari] = df[oznitelik_sutunlari].ffill().bfill()
    X = df[oznitelik_sutunlari]
    y = df["anomaly"].astype(int)
    gruplar = df["source_file"]
    return X, y, gruplar, df

def batadal_yukle():
    yol = os.path.join(BATADAL_DIZINI, "BATADAL_dataset04.csv")
    df = pd.read_csv(yol)
    df.columns = [c.strip() for c in df.columns]
    return df

def batadal_on_isle(df):
    df = df.copy()
    etiket_sutunu = None
    for col in df.columns:
        if "flag" in col.lower() or "label" in col.lower() or "attack" in col.lower() or col == "ATT_FLAG":
            etiket_sutunu = col
            break
    zaman_sutunlari = ["DATETIME", "datetime", "timestamp", "TIMESTAMP"]
    oznitelik_sutunlari = [col for col in df.columns if col not in zaman_sutunlari and col != etiket_sutunu]
    df[oznitelik_sutunlari] = df[oznitelik_sutunlari].ffill().bfill()
    X = df[oznitelik_sutunlari]
    y = df[etiket_sutunu].map({-999: 0, 0: 0, 1: 1}).fillna(0).astype(int)
    return X, y, df

def pca_ve_olceklendirici_uygula(X_train, X_val=None, X_test=None):
    olceklendirici = StandardScaler()
    pca = PCA(n_components=1)
    X_train_olcekli = olceklendirici.fit_transform(X_train)
    X_train_pc = pca.fit_transform(X_train_olcekli)
    sonuc = [X_train_pc[:, 0]]
    if X_val is not None:
        X_val_olcekli = olceklendirici.transform(X_val)
        X_val_pc = pca.transform(X_val_olcekli)
        sonuc.append(X_val_pc[:, 0])
    if X_test is not None:
        X_test_olcekli = olceklendirici.transform(X_test)
        X_test_pc = pca.transform(X_test_olcekli)
        sonuc.append(X_test_pc[:, 0])
    return tuple(sonuc)

def skab_bolmeleri_al(X, y, gruplar):
    gkf = GroupKFold(n_splits=5)
    bolmeler = []
    for train_idx, test_idx in gkf.split(X, y, gruplar):
        bolmeler.append((train_idx, test_idx))
    return bolmeler

def batadal_bolmeleri_al(X, y):
    n = len(X)
    train_end = int(0.60 * n)
    val_end = int(0.80 * n)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
