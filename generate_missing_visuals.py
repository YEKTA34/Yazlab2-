"""Sadece eksik olan Confusion Matrix ve ROC Curve gorsellerini uretir."""
import os, sys, torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import OUTPUT_DIZINI
import preprocess
from automata import OlasiliksalOtomata, paa_donusumu, sax_donusumu
from explainability import AciklanabilirlikModulu
from utils.visualize import plot_confusion_matrix, plot_roc_curve

print("Veri yukleniyor...")
skab_df = preprocess.skab_yukle()
X, y, gruplar, _ = preprocess.skab_on_isle(skab_df)
bolmeler = preprocess.skab_bolmeleri_al(X, y, gruplar)
train_idx, test_idx = list(bolmeler)[0]
X_train = X.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
train_pc, test_pc = preprocess.pca_ve_olceklendirici_uygula(X_train, X_test=X_test)

# Kucuk otomata - hizli calisir
print("Otomata (W=4, A=3) egitiliyor...")
otomata = OlasiliksalOtomata(pencere_boyutu=4, alfabe_boyutu=3)
otomata.egit(train_pc, paa_penceresi=1, alpha=0.01)
print(f"  Toplam durum sayisi: {len(otomata.durumlar)}")

# Test
print("Test kumesi uzerinde tahmin yapiliyor...")
paa_test = paa_donusumu(test_pc, paa_penceresi=1)
sax_test = sax_donusumu(paa_test, otomata.alfabe_boyutu)

y_dogru = []
y_tahmin = []
y_scores = []
aciklama_modulu = AciklanabilirlikModulu(otomata, anomali_esigi=0.01)

n_adim = len(sax_test)
for t in range(otomata.pencere_boyutu, n_adim):
    if t % 5000 == 0:
        print(f"  Adim {t}/{n_adim}...")
    aciklama = aciklama_modulu.adimi_acikla(t, sax_test[:t+1])
    y_dogru.append(y_test.iloc[t])
    y_tahmin.append(1 if aciklama["karar"] == "anomali" else 0)
    y_scores.append(aciklama.get('mesafe_skoru', 0.0))

# Confusion Matrix
print("[4/5] Confusion Matrix kaydediliyor...")
plot_confusion_matrix(y_dogru, y_tahmin,
                      title="Otomata Confusion Matrix",
                      save_path=os.path.join(OUTPUT_DIZINI, 'automata_confusion_matrix.png'))
print("  Kaydedildi!")

# ROC Curve
print("[5/5] ROC Egrisi kaydediliyor...")
max_s = max(y_scores) if max(y_scores) > 0 else 1
y_scores_norm = [s / max_s for s in y_scores]
plot_roc_curve(y_dogru, y_scores_norm,
               title="Otomata ROC Egrisi",
               save_path=os.path.join(OUTPUT_DIZINI, 'automata_roc_curve.png'))
print("  Kaydedildi!")

print("\nTamamlandi! Tum PNG dosyalari:")
for f in sorted(os.listdir(OUTPUT_DIZINI)):
    if f.endswith('.png'):
        print(f"  - outputs/{f}")
