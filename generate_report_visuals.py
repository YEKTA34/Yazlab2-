"""
Rapor Gorsellestirme Betigi
- Grid Search Heatmap
- Transition Probability Heatmap
- Automata State Diagram
- Confusion Matrix
- ROC Egrisi
"""
import os
import sys
import json
import torch  # PyTorch DLL sorunu icin en basta import edilmeli
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI penceresi acmadan cizim yapar
import matplotlib.pyplot as plt
import seaborn as sns

# Yollari ekleyelim
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from config import OUTPUT_DIZINI, PENCERE_BOYUTU, ALFABE_BOYUTU
import preprocess
from automata import OlasiliksalOtomata, paa_donusumu, sax_donusumu
from explainability import AciklanabilirlikModulu
from utils.visualize import plot_confusion_matrix, plot_roc_curve, plot_transition_heatmap

# ============================================================
# 1) GRID SEARCH HEATMAP
# ============================================================
def generate_grid_search_heatmap(json_path):
    print("[1/5] Grid Search Heatmap olusturuluyor...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    grid_results = [item for item in data if item.get('scenario') == 'GridSearch']
    if not grid_results:
        print("  GridSearch sonuclari bulunamadi, atlaniyor.")
        return

    rows = []
    for res in grid_results:
        rows.append({
            'Pencere Boyutu': res['parameters']['window_size'],
            'Alfabe Boyutu': res['parameters']['alphabet_size'],
            'F1 Skoru': res['metrics']['F1']
        })
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='Pencere Boyutu', columns='Alfabe Boyutu', values='F1 Skoru')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.4f')
    plt.title('Parametre Duyarliligi - Otomata F1 Skoru')
    path = os.path.join(OUTPUT_DIZINI, 'grid_search_heatmap.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Kaydedildi: {path}")

# ============================================================
# 2) TRANSITION HEATMAP  (kucuk otomata ile)
# ============================================================
def generate_transition_heatmap(train_pc):
    print("[2/5] Transition Heatmap olusturuluyor...")
    oto = OlasiliksalOtomata(pencere_boyutu=3, alfabe_boyutu=3)
    oto.egit(train_pc[:1000], paa_penceresi=1, alpha=0.01)

    durumlar = sorted(oto.durumlar)
    n = len(durumlar)
    mat = np.zeros((n, n))
    for i, d in enumerate(durumlar):
        for j, s in enumerate(durumlar):
            mat[i, j] = oto.gecisler[d][s]

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, cmap='viridis', fmt='.3f',
                xticklabels=durumlar, yticklabels=durumlar)
    plt.title('Durum Gecis Olasiliklari (W=3, A=3)')
    plt.xlabel('Sonraki Durum')
    plt.ylabel('Mevcut Durum')
    path = os.path.join(OUTPUT_DIZINI, 'transition_heatmap.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Kaydedildi: {path}")
    return oto, durumlar, mat

# ============================================================
# 3) STATE DIAGRAM  (networkx ile)
# ============================================================
def generate_state_diagram(durumlar, mat):
    print("[3/5] Automata State Diagram olusturuluyor...")
    try:
        import networkx as nx
    except ImportError:
        print("  networkx yuklu degil, state diagram atlaniyor.")
        return

    G = nx.DiGraph()
    threshold = 0.05
    for i, d in enumerate(durumlar):
        for j, s in enumerate(durumlar):
            if mat[i, j] > threshold:
                G.add_edge(d, s, weight=mat[i, j])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw(G, pos, with_labels=True, node_color='#87CEEB',
            node_size=1800, font_size=9, font_weight='bold',
            arrows=True, arrowsize=15, edge_color='gray')
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title('Otomata Durum Gecis Diyagrami (p > 0.05)')
    path = os.path.join(OUTPUT_DIZINI, 'automata_state_diagram.png')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Kaydedildi: {path}")

# ============================================================
# 4) CONFUSION MATRIX
# ============================================================
def generate_confusion_matrix(otomata, test_pc, y_test):
    print("[4/5] Confusion Matrix olusturuluyor...")
    paa_test = paa_donusumu(test_pc, paa_penceresi=1)
    sax_test = sax_donusumu(paa_test, otomata.alfabe_boyutu)

    y_dogru = []
    y_tahmin = []
    aciklama_modulu = AciklanabilirlikModulu(otomata, anomali_esigi=0.01)

    n_adim = len(sax_test)
    for t in range(otomata.pencere_boyutu, n_adim):
        aciklama = aciklama_modulu.adimi_acikla(t, sax_test[:t+1])
        y_dogru.append(y_test.iloc[t])
        y_tahmin.append(1 if aciklama["karar"] == "anomali" else 0)

    plot_confusion_matrix(y_dogru, y_tahmin,
                          title="Otomata Confusion Matrix",
                          save_path=os.path.join(OUTPUT_DIZINI, 'automata_confusion_matrix.png'))
    print(f"  Kaydedildi: {os.path.join(OUTPUT_DIZINI, 'automata_confusion_matrix.png')}")
    return y_dogru, y_tahmin, [None] * len(y_dogru)  # aciklamalari dondur

# ============================================================
# 5) ROC CURVE
# ============================================================
def generate_roc(otomata, test_pc, y_test):
    print("[5/5] ROC Egrisi olusturuluyor...")
    paa_test = paa_donusumu(test_pc, paa_penceresi=1)
    sax_test = sax_donusumu(paa_test, otomata.alfabe_boyutu)

    y_dogru = []
    y_scores = []
    aciklama_modulu = AciklanabilirlikModulu(otomata, anomali_esigi=0.01)

    n_adim = len(sax_test)
    for t in range(otomata.pencere_boyutu, n_adim):
        aciklama = aciklama_modulu.adimi_acikla(t, sax_test[:t+1])
        y_dogru.append(y_test.iloc[t])
        y_scores.append(aciklama.get('mesafe_skoru', 0.0))

    max_s = max(y_scores) if max(y_scores) > 0 else 1
    y_scores_norm = [s / max_s for s in y_scores]

    plot_roc_curve(y_dogru, y_scores_norm,
                   title="Otomata ROC Egrisi",
                   save_path=os.path.join(OUTPUT_DIZINI, 'automata_roc_curve.png'))
    print(f"  Kaydedildi: {os.path.join(OUTPUT_DIZINI, 'automata_roc_curve.png')}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 50)
    print("RAPOR GORSELLESTIRME BASLIYOR")
    print("=" * 50)

    # JSON dosyasini bul
    json_files = [f for f in os.listdir(OUTPUT_DIZINI)
                  if f.startswith('experiment_') and f.endswith('.json')]
    if not json_files:
        print("Hata: outputs/ dizininde experiment_*.json bulunamadi!")
        return
    latest_json = os.path.join(OUTPUT_DIZINI, sorted(json_files)[-1])
    print(f"JSON kaynak: {latest_json}\n")

    # 1) Grid Search Heatmap
    generate_grid_search_heatmap(latest_json)

    # Veriyi yukle
    print("\nSKAB verisi yukleniyor (sadece 1 fold)...")
    skab_df = preprocess.skab_yukle()
    X, y, gruplar, _ = preprocess.skab_on_isle(skab_df)
    bolmeler = preprocess.skab_bolmeleri_al(X, y, gruplar)
    train_idx, test_idx = list(bolmeler)[0]
    X_train = X.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    train_pc, test_pc = preprocess.pca_ve_olceklendirici_uygula(X_train, X_test=X_test)

    # 2) Transition Heatmap (kucuk otomata)
    _, durumlar, mat = generate_transition_heatmap(train_pc)

    # 3) State Diagram (kucuk otomata)
    generate_state_diagram(durumlar, mat)

    # Gercek performans icin buyuk otomata (W=6, A=6)
    print("\nOtomata (W=6, A=6) egitiliyor...")
    otomata = OlasiliksalOtomata(pencere_boyutu=6, alfabe_boyutu=6)
    otomata.egit(train_pc, paa_penceresi=1, alpha=0.01)

    # 4) Confusion Matrix
    generate_confusion_matrix(otomata, test_pc, y_test)

    # 5) ROC Curve
    generate_roc(otomata, test_pc, y_test)

    print("\n" + "=" * 50)
    print("TUM GORSELLER BASARIYLA OLUSTURULDU!")
    print("=" * 50)
    for f in sorted(os.listdir(OUTPUT_DIZINI)):
        if f.endswith('.png'):
            print(f"  - outputs/{f}")

if __name__ == '__main__':
    main()
