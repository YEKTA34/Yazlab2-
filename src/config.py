import os
import yaml

# Bu dosyanın bulunduğu dizin (src/)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Projenin ana dizini (yazlab2/)
ANA_DIZIN = os.path.abspath(os.path.join(SRC_DIR, ".."))

CONFIG_PATH = os.path.join(ANA_DIZIN, "configs", "default_config.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()

# Yollar
VERI_DIZINI = os.path.join(ANA_DIZIN, cfg["paths"]["data_dir"])
SKAB_DIZINI = os.path.join(ANA_DIZIN, cfg["paths"]["skab_dir"])
BATADAL_DIZINI = os.path.join(ANA_DIZIN, cfg["paths"]["batadal_dir"])
OUTPUT_DIZINI = os.path.join(ANA_DIZIN, cfg["paths"]["output_dir"])

# Parametreler
PENCERE_BOYUTU = cfg["automata"]["window_size"]
ALFABE_BOYUTU = cfg["automata"]["alphabet_size"]
PCA_BILESEN_SAYISI = cfg["automata"]["pca_components"]
TOHUMLAR = cfg["experiment"]["seeds"]

BATADAL_BOLME = cfg["batadal_split"]
