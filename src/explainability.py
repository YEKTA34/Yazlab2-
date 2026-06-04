import json

class AciklanabilirlikModulu:
    def __init__(self, otomata, anomali_esigi=0.05):
        self.otomata = otomata
        self.anomali_esigi = anomali_esigi
        self.yol_olasiligi = 1.0
        
    def sifirla(self):
        self.yol_olasiligi = 1.0
