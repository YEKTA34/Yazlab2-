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

def sax_donusumu(paa_verisi, alfabe_boyutu):
    kirilma_noktalari = stats.norm.ppf(np.linspace(0, 1, alfabe_boyutu + 1)[1:-1])
    semboller = [chr(97 + i) for i in range(alfabe_boyutu)]
    sax_dizisi = []
    for deger in paa_verisi:
        indeks = 0
        for kn in kirilma_noktalari:
            if deger > kn:
                indeks += 1
            else:
                break
        sax_dizisi.append(semboller[indeks])
    return "".join(sax_dizisi)

def levenshtein_uzakligi(d1, d2):
    if len(d1) < len(d2):
        return levenshtein_uzakligi(d2, d1)
    if len(d2) == 0:
        return len(d1)
    onceki_satir = range(len(d2) + 1)
    for i, c1 in enumerate(d1):
        guncel_satir = [i + 1]
        for j, c2 in enumerate(d2):
            eklemeler = onceki_satir[j + 1] + 1
            silmeler = guncel_satir[j] + 1
            degistirmeler = onceki_satir[j] + (c1 != c2)
            guncel_satir.append(min(eklemeler, silmeler, degistirmeler))
        onceki_satir = guncel_satir
    return onceki_satir[-1]
