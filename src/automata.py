import numpy as np
import scipy.stats as stats

def paa_donusumu(veri, paa_penceresi=1):
    n = len(veri)
    if paa_penceresi <= 1:
        return veri
    segment_sayisi = int(np.ceil(n / paa_penceresi))
    paa_verisi = []
    for i in range(segment_sayisi):
        blok = veri[i * paa_penceresi : (i + 1) * paa_penceresi]
        paa_verisi.append(np.mean(blok))
    return np.array(paa_verisi)
