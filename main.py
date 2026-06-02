import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import PENCERE_BOYUTU, ALFABE_BOYUTU, OUTPUT_DIZINI
import preprocess
from automata import OlasiliksalOtomata, paa_donusumu, sax_donusumu
from explainability import AciklanabilirlikModulu

def test_kumesini_degerlendir(otomata, test_pc1, y_test, esik_deger=0.01):
    paa_test = paa_donusumu(test_pc1, paa_penceresi=1)
    sax_test = sax_donusumu(paa_test, otomata.alfabe_boyutu)
    
    y_dogru_hizalanmis = []
    y_tahmin_hizalanmis = []
    aciklamalar = []
    
    aciklama_modulu = AciklanabilirlikModulu(otomata, anomali_esigi=esik_deger)
    
    n_adim = len(sax_test)
    for t in range(PENCERE_BOYUTU, n_adim):
        aciklama = aciklama_modulu.adimi_acikla(t, sax_test[:t+1])
        aciklamalar.append(aciklama)
        
        y_dogru_hizalanmis.append(y_test.iloc[t])
        y_tahmin_hizalanmis.append(1 if aciklama["karar"] == "anomali" else 0)
        
    return y_dogru_hizalanmis, y_tahmin_hizalanmis, aciklamalar

def metrikleri_yazdir(y_dogru, y_tahmin, baslik):
    acc = accuracy_score(y_dogru, y_tahmin)
    prec = precision_score(y_dogru, y_tahmin, zero_division=0)
    rec = recall_score(y_dogru, y_tahmin, zero_division=0)
    f1 = f1_score(y_dogru, y_tahmin, zero_division=0)
    
    print(f"\n--- {baslik} ---")
    print(f"Dogruluk (Accuracy)    : {acc:.4f}")
    print(f"Hassasiyet (Precision) : {prec:.4f}")
    print(f"Duyarlilik (Recall)    : {rec:.4f}")
    print(f"F1 Skoru (F1-Score)    : {f1:.4f}")
    return acc, prec, rec, f1

def main():
    print("SKAB Veri Seti Yukleniyor...")
    skab_df = preprocess.skab_yukle()
    X_skab, y_skab, gruplar_skab, skab_df = preprocess.skab_on_isle(skab_df)
    
    skab_bolmeleri = preprocess.skab_bolmeleri_al(X_skab, y_skab, gruplar_skab)
    
    skab_f1_skorlari = []
    print("\nSKAB Veri Seti GroupKFold ile Degerlendiriliyor...")
    for idx, (train_idx, test_idx) in enumerate(skab_bolmeleri):
        X_train, y_train = X_skab.iloc[train_idx], y_skab.iloc[train_idx]
        X_test, y_test = X_skab.iloc[test_idx], y_skab.iloc[test_idx]
        
        train_pc, test_pc = preprocess.pca_ve_olceklendirici_uygula(X_train, X_test=X_test)
        
        otomata = OlasiliksalOtomata(pencere_boyutu=PENCERE_BOYUTU, alfabe_boyutu=ALFABE_BOYUTU)
        otomata.egit(train_pc, paa_penceresi=1, alpha=0.01)
        
        y_dogru, y_tahmin, aciklamalar = test_kumesini_degerlendir(otomata, test_pc, y_test, esik_deger=0.01)
        
        _, _, _, f1 = metrikleri_yazdir(y_dogru, y_tahmin, f"SKAB Fold {idx+1}")
        skab_f1_skorlari.append(f1)
        
    print(f"\nSKAB GroupKFold Ortalama F1 Skoru: {np.mean(skab_f1_skorlari):.4f} +/- {np.std(skab_f1_skorlari):.4f}")
    
    print("\nBATADAL Veri Seti Yukleniyor...")
    batadal_df = preprocess.batadal_yukle()
    X_bat, y_bat, batadal_df = preprocess.batadal_on_isle(batadal_df)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess.batadal_bolmeleri_al(X_bat, y_bat)
    
    train_pc, val_pc, test_pc = preprocess.pca_ve_olceklendirici_uygula(X_train, X_val=X_val, X_test=X_test)
    
    otomata_bat = OlasiliksalOtomata(pencere_boyutu=PENCERE_BOYUTU, alfabe_boyutu=ALFABE_BOYUTU)
    otomata_bat.egit(train_pc, paa_penceresi=1, alpha=0.01)
    
    y_dogru_bat, y_tahmin_bat, aciklamalar_bat = test_kumesini_degerlendir(otomata_bat, test_pc, y_test, esik_deger=0.01)
    
    metrikleri_yazdir(y_dogru_bat, y_tahmin_bat, "BATADAL Test Degerlendirmesi (%60-%20-%20 Zaman Bolmesi)")
    
    os.makedirs(OUTPUT_DIZINI, exist_ok=True)
    ornek_aciklamalar = [exp for exp in aciklamalar_bat if exp["durum_bilgisi"] == "gorulmeyen"][:5]
    if not ornek_aciklamalar:
        ornek_aciklamalar = aciklamalar_bat[:5]
        
    cikti_yolu = os.path.join(OUTPUT_DIZINI, "sample_explanation.json")
    with open(cikti_yolu, "w", encoding="utf-8") as f:
        json.dump(ornek_aciklamalar, f, indent=2)
        
    print(f"\nOrnek JSON aciklamalari sunun altina kaydedildi: {cikti_yolu}")
    print("\n--- Ornek Cikti Ornegi ---")
    print(json.dumps(ornek_aciklamalar[0], indent=2))

if __name__ == "__main__":
    main()
