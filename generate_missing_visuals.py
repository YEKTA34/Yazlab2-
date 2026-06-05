"""
Rapor Gorselleri - Duzeltilmis Versiyon
- Transition Heatmap: W=2, A=3 (9 durum, temiz ve okunabilir)
- Confusion Matrix: W=5, A=5 (daha iyi performans)
- ROC Curve: W=5, A=5
- State Diagram: W=2, A=3 (temiz diyagram)
"""
import os, sys, torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIZINI
import preprocess
from automata import OlasiliksalOtomata, paa_donusumu, sax_donusumu
from explainability import AciklanabilirlikModulu
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ============================================================
# Veri Yukleme
# ============================================================
print("Veri yukleniyor...")
skab_df = preprocess.skab_yukle()
X, y, gruplar, _ = preprocess.skab_on_isle(skab_df)
bolmeler = preprocess.skab_bolmeleri_al(X, y, gruplar)
train_idx, test_idx = list(bolmeler)[0]
X_train = X.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
train_pc, test_pc = preprocess.pca_ve_olceklendirici_uygula(X_train, X_test=X_test)

# ============================================================
# 1) TRANSITION HEATMAP - Temiz ve okunabilir (W=2, A=3 -> 9 durum)
# ============================================================
print("\n[1/5] Transition Heatmap (W=2, A=3)...")
oto_kucuk = OlasiliksalOtomata(pencere_boyutu=2, alfabe_boyutu=3)
oto_kucuk.egit(train_pc[:2000], paa_penceresi=1, alpha=0.01)

durumlar = sorted(oto_kucuk.durumlar)
n = len(durumlar)
print(f"  Durum sayisi: {n}")
mat = np.zeros((n, n))
for i, d in enumerate(durumlar):
    for j, s in enumerate(durumlar):
        mat[i, j] = oto_kucuk.gecisler[d][s]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(mat, annot=True, cmap='YlGnBu', fmt='.2f',
            xticklabels=durumlar, yticklabels=durumlar,
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 11}, ax=ax)
ax.set_title('Durum Gecis Olasiliklari Matrisi', fontsize=14, fontweight='bold')
ax.set_xlabel('Sonraki Durum', fontsize=12)
ax.set_ylabel('Mevcut Durum', fontsize=12)
plt.tight_layout()
path = os.path.join(OUTPUT_DIZINI, 'transition_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Kaydedildi: {path}")

# ============================================================
# 2) STATE DIAGRAM (ayni kucuk otomata ile)
# ============================================================
print("\n[2/5] State Diagram...")
try:
    import networkx as nx
    G = nx.DiGraph()
    threshold = 0.05
    for i, d in enumerate(durumlar):
        for j, s in enumerate(durumlar):
            if mat[i, j] > threshold:
                G.add_edge(d, s, weight=mat[i, j])

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw(G, pos, with_labels=True, node_color='#87CEEB',
            node_size=2200, font_size=11, font_weight='bold',
            arrows=True, arrowsize=18, edge_color='gray',
            width=1.5, ax=ax)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    ax.set_title('Otomata Durum Gecis Diyagrami (p > 0.05)', fontsize=14, fontweight='bold')
    path = os.path.join(OUTPUT_DIZINI, 'automata_state_diagram.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {path}")
except ImportError:
    print("  networkx yuklu degil, atlaniyor.")

# ============================================================
# 3) CONFUSION MATRIX (W=5, A=5 -> iyi performans, hizli)
# ============================================================
print("\n[3/5] Confusion Matrix (W=5, A=5)...")
otomata = OlasiliksalOtomata(pencere_boyutu=5, alfabe_boyutu=5)
otomata.egit(train_pc, paa_penceresi=1, alpha=0.01)
print(f"  Durum sayisi: {len(otomata.durumlar)}")

paa_test = paa_donusumu(test_pc, paa_penceresi=1)
sax_test = sax_donusumu(paa_test, otomata.alfabe_boyutu)

y_dogru = []
y_tahmin = []
y_scores = []
aciklama_modulu = AciklanabilirlikModulu(otomata, anomali_esigi=0.01)

n_adim = len(sax_test)
for t in range(otomata.pencere_boyutu, n_adim):
    if t % 2000 == 0:
        print(f"  Adim {t}/{n_adim}...")
    aciklama = aciklama_modulu.adimi_acikla(t, sax_test[:t+1])
    y_dogru.append(y_test.iloc[t])
    y_tahmin.append(1 if aciklama["karar"] == "anomali" else 0)
    y_scores.append(aciklama.get('mesafe_skoru', 0.0))

cm = confusion_matrix(y_dogru, y_tahmin)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 16}, linewidths=1, linecolor='white',
            xticklabels=['Normal (0)', 'Anomali (1)'],
            yticklabels=['Normal (0)', 'Anomali (1)'], ax=ax)
ax.set_title('Otomata Confusion Matrix (W=5, A=5)', fontsize=14, fontweight='bold')
ax.set_xlabel('Tahmin Edilen', fontsize=12)
ax.set_ylabel('Gercek Deger', fontsize=12)
plt.tight_layout()
path = os.path.join(OUTPUT_DIZINI, 'automata_confusion_matrix.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Kaydedildi: {path}")

# ============================================================
# 4) ROC CURVE - anomali skoru = (1 - yol_olasiligi) + uzaklik
# ============================================================
print("\n[4/5] ROC Egrisi...")
# Yeniden hesapla cunku y_scores icin farkli skor lazim
y_dogru_roc = []
y_anomali_scores = []

paa_test2 = paa_donusumu(test_pc, paa_penceresi=1)
sax_test2 = sax_donusumu(paa_test2, otomata.alfabe_boyutu)
aciklama_modulu2 = AciklanabilirlikModulu(otomata, anomali_esigi=0.01)

for t in range(otomata.pencere_boyutu, len(sax_test2)):
    aciklama = aciklama_modulu2.adimi_acikla(t, sax_test2[:t+1])
    y_dogru_roc.append(y_test.iloc[t])
    # Anomali skoru: dusuk yol olasiligi + yuksek uzaklik = yuksek anomali riski
    anomali_skoru = (1.0 - aciklama['yol_olasiligi']) + aciklama['uzaklik']
    y_anomali_scores.append(anomali_skoru)

# Normalizasyon
max_s = max(y_anomali_scores) if max(y_anomali_scores) > 0 else 1
y_scores_norm = [s / max_s for s in y_anomali_scores]

fpr, tpr, _ = roc_curve(y_dogru_roc, y_scores_norm)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'ROC Egrisi (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Rastgele Tahmin')
ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (Yanlis Pozitif)', fontsize=12)
ax.set_ylabel('True Positive Rate (Dogru Pozitif)', fontsize=12)
ax.set_title(f'Otomata ROC Egrisi (W=5, A=5, AUC={roc_auc:.2f})', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIZINI, 'automata_roc_curve.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Kaydedildi: {path}")

# ============================================================
# 5) GRID SEARCH HEATMAP (zaten dogru ama yeniden temiz uretilsin)
# ============================================================
print("\n[5/5] Grid Search Heatmap...")
import json
json_files = [f for f in os.listdir(OUTPUT_DIZINI) if f.startswith('experiment_') and f.endswith('.json')]
latest_json = os.path.join(OUTPUT_DIZINI, sorted(json_files)[-1])
with open(latest_json, 'r') as f:
    data = json.load(f)

grid_results = [item for item in data if item.get('scenario') == 'GridSearch']
rows = []
for res in grid_results:
    rows.append({
        'Pencere Boyutu': res['parameters']['window_size'],
        'Alfabe Boyutu': res['parameters']['alphabet_size'],
        'F1 Skoru': res['metrics']['F1']
    })
import pandas as pd
df = pd.DataFrame(rows)
pivot = df.pivot(index='Pencere Boyutu', columns='Alfabe Boyutu', values='F1 Skoru')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.4f',
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 12}, ax=ax)
ax.set_title('Parametre Duyarliligi - Otomata F1 Skoru', fontsize=14, fontweight='bold')
ax.set_xlabel('Alfabe Boyutu', fontsize=12)
ax.set_ylabel('Pencere Boyutu', fontsize=12)
plt.tight_layout()
path = os.path.join(OUTPUT_DIZINI, 'grid_search_heatmap.png')
plt.savefig(path, dpi=150)
plt.close()
print(f"  Kaydedildi: {path}")

# ============================================================
print("\n" + "=" * 50)
print("TUM GORSELLER BASARIYLA GUNCELLENDI!")
print("=" * 50)
for f in sorted(os.listdir(OUTPUT_DIZINI)):
    if f.endswith('.png'):
        print(f"  - outputs/{f}")
