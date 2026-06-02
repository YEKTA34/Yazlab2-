import os

ANA_DIZIN = r"D:\yazlab2"
VERI_DIZINI = os.path.join(ANA_DIZIN, "data")
SKAB_DIZINI = os.path.join(VERI_DIZINI, "skab")
BATADAL_DIZINI = os.path.join(VERI_DIZINI, "batadal")

SKAB_DIR = SKAB_DIZINI
BATADAL_DIR = BATADAL_DIZINI

PENCERE_BOYUTU = 4
ALFABE_BOYUTU = 3
PCA_BILESEN_SAYISI = 1
TOHUMLAR = [42, 123, 2026, 7, 999]

BATADAL_BOLME = {
    "train": 0.60,
    "val": 0.20,
    "test": 0.20
}
