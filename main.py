import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import warnings
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import PENCERE_BOYUTU, ALFABE_BOYUTU, OUTPUT_DIZINI, TOHUMLAR
import preprocess
from automata import OlasiliksalOtomata, paa_donusumu, sax_donusumu
from explainability import AciklanabilirlikModulu
from pipeline.experiment import run_dl_experiment, add_gaussian_noise
from pipeline.logger import ExperimentLogger
from utils.seed import set_seed

def test_kumesini_degerlendir(otomata, test_pc1, y_test, esik_deger=0.01):
    paa_test = paa_donusumu(test_pc1, paa_penceresi=1)
    sax_test = sax_donusumu(paa_test, otomata.alfabe_boyutu)
    
    y_dogru_hizalanmis = []
    y_tahmin_hizalanmis = []
    aciklamalar = []
    
    aciklama_modulu = AciklanabilirlikModulu(otomata, anomali_esigi=esik_deger)
    
    n_adim = len(sax_test)
    for t in range(otomata.pencere_boyutu, n_adim):
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
    if baslik:
        print(f"\n--- {baslik} ---")
        print(f"Dogruluk (Accuracy)    : {acc:.4f}")
        print(f"Hassasiyet (Precision) : {prec:.4f}")
        print(f"Duyarlilik (Recall)    : {rec:.4f}")
        print(f"F1 Skoru (F1-Score)    : {f1:.4f}")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def run_standard_experiments(logger, skab_df, batadal_df):
    senaryolar = ["Orijinal", "Gurultulu", "Unseen"]
    
    tum_oto_f1 = []
    tum_lstm_f1 = []
    
    print("\n================== ANA DENEY DONGUSU ==================")
    for seed in TOHUMLAR:
        print(f"\n>>> SEED: {seed} basliyor <<<")
        set_seed(seed)
        
        X_skab, y_skab, gruplar_skab, _ = preprocess.skab_on_isle(skab_df)
        skab_bolmeleri = preprocess.skab_bolmeleri_al(X_skab, y_skab, gruplar_skab)
        
        oto_seed_f1 = []
        lstm_seed_f1 = []
        
        for idx, (train_idx, test_idx) in enumerate(skab_bolmeleri):
            if idx > 1: # Demoda cok zaman almamasi icin sadece ilk 2 foldda deniyoruz
                break
            
            X_train, y_train = X_skab.iloc[train_idx], y_skab.iloc[train_idx]
            X_test, y_test = X_skab.iloc[test_idx], y_skab.iloc[test_idx]
            
            train_pc, test_pc_orj = preprocess.pca_ve_olceklendirici_uygula(X_train, X_test=X_test)
            
            for senaryo in senaryolar:
                print(f"Seed {seed} | SKAB Fold {idx+1} | Senaryo: {senaryo} | Egitim basliyor...")
                test_pc = test_pc_orj.copy()
                if senaryo == "Gurultulu":
                    test_pc = add_gaussian_noise(test_pc, mean=0, std=0.5)
                elif senaryo == "Unseen":
                    test_pc = test_pc * -1 # Ters cevirerek gormedigi patternlari simule ediyoruz
                    
                otomata = OlasiliksalOtomata(pencere_boyutu=PENCERE_BOYUTU, alfabe_boyutu=ALFABE_BOYUTU)
                otomata.egit(train_pc, paa_penceresi=1, alpha=0.01)
                
                y_dogru, y_tahmin, aciklamalar = test_kumesini_degerlendir(otomata, test_pc, y_test, esik_deger=0.01)
                otomata_metrics = metrikleri_yazdir(y_dogru, y_tahmin, "")
                logger.log_run("ProbabilisticAutomata", senaryo, {"dataset": "SKAB", "fold": idx+1, "seed": seed}, otomata_metrics)
                
                lstm_metrics = run_dl_experiment(train_pc, test_pc, test_pc, y_test, model_type="LSTM", scenario=senaryo)
                logger.log_run("LSTM", senaryo, {"dataset": "SKAB", "fold": idx+1, "seed": seed}, lstm_metrics)
                
                if senaryo == "Orijinal":
                    oto_seed_f1.append(otomata_metrics["F1"])
                    lstm_seed_f1.append(lstm_metrics["F1"])
        
        tum_oto_f1.append(np.mean(oto_seed_f1))
        tum_lstm_f1.append(np.mean(lstm_seed_f1))

    print(f"\nTum Seedler Icin SKAB (Orijinal) Ortalama F1 Skoru (Otomata): {np.mean(tum_oto_f1):.4f} +/- {np.std(tum_oto_f1):.4f}")
    print(f"Tum Seedler Icin SKAB (Orijinal) Ortalama F1 Skoru (LSTM): {np.mean(tum_lstm_f1):.4f} +/- {np.std(tum_lstm_f1):.4f}")
    
    # Wilcoxon Testi
    if len(tum_oto_f1) >= 5: 
        try:
            stat, p = wilcoxon(tum_oto_f1, tum_lstm_f1)
            print(f"\nWilcoxon Istatistiksel Testi (Otomata vs LSTM): Statistic={stat:.4f}, p-value={p:.4f}")
            if p < 0.05:
                print("Sonuc: Iki model arasinda istatistiksel olarak ANLAMLI bir fark vardir (p < 0.05).")
            else:
                print("Sonuc: Iki model arasinda istatistiksel olarak ANLAMLI BIR FARK YOKTUR (p >= 0.05).")
        except Exception as e:
            print(f"Wilcoxon testi hesaplanamadi: {e}")

def run_grid_search(logger, skab_df):
    print("\n================== PARAMETRE GRID SEARCH ==================")
    windows = [3, 4, 5, 6]
    alphabets = [3, 4, 5, 6]
    
    set_seed(42)
    X_skab, y_skab, gruplar_skab, _ = preprocess.skab_on_isle(skab_df)
    skab_bolmeleri = preprocess.skab_bolmeleri_al(X_skab, y_skab, gruplar_skab)
    train_idx, test_idx = list(skab_bolmeleri)[0] 
    X_train, y_train = X_skab.iloc[train_idx], y_skab.iloc[train_idx]
    X_test, y_test = X_skab.iloc[test_idx], y_skab.iloc[test_idx]
    
    train_pc, test_pc = preprocess.pca_ve_olceklendirici_uygula(X_train, X_test=X_test)
    
    for w in windows:
        for a in alphabets:
            print(f"Grid Search: Window={w}, Alphabet={a} deneniyor...")
            otomata = OlasiliksalOtomata(pencere_boyutu=w, alfabe_boyutu=a)
            otomata.egit(train_pc, paa_penceresi=1, alpha=0.01)
            
            y_dogru, y_tahmin, _ = test_kumesini_degerlendir(otomata, test_pc, y_test, esik_deger=0.01)
            metrics = metrikleri_yazdir(y_dogru, y_tahmin, "")
            logger.log_run("ProbabilisticAutomata", "GridSearch", 
                           {"window_size": w, "alphabet_size": a}, 
                           metrics)
                           
    print("Grid Search islemi tamamlandi. Sonuclar json dosyasina eklendi.")

def main():
    logger = ExperimentLogger(log_dir=OUTPUT_DIZINI)
    
    print("SKAB Veri Seti Yukleniyor...")
    skab_df = preprocess.skab_yukle()
    
    print("\nBATADAL Veri Seti Yukleniyor...")
    batadal_df = preprocess.batadal_yukle()
    
    # 1. Sabit parametrelerle 5 seed, 3 senaryo, ve istatistiksel testler
    run_standard_experiments(logger, skab_df, batadal_df)
    
    # 2. Parametre Degisimi (Grid Search)
    run_grid_search(logger, skab_df)
    
    logger.summary()

if __name__ == "__main__":
    main()
