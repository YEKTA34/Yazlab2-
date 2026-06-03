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
