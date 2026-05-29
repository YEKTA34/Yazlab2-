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

class OlasiliksalOtomata:
    def __init__(self, pencere_boyutu=4, alfabe_boyutu=3):
        self.pencere_boyutu = pencere_boyutu
        self.alfabe_boyutu = alfabe_boyutu
        self.durumlar = set()
        self.gecisler = {}

    def egit(self, egitim_verisi, paa_penceresi=1, alpha=0.01):
        paa_verisi = paa_donusumu(egitim_verisi, paa_penceresi)
        sax_dizisi = sax_donusumu(paa_verisi, self.alfabe_boyutu)
        oruntuler = []
        for i in range(len(sax_dizisi) - self.pencere_boyutu + 1):
            oruntuler.append(sax_dizisi[i : i + self.pencere_boyutu])
        self.durumlar = set(oruntuler)
        tum_durumlar = list(self.durumlar)
        self.gecisler = {durum: {sonraki_durum: alpha for sonraki_durum in tum_durumlar} for durum in self.durumlar}
        for i in range(len(oruntuler) - 1):
            guncel_durum = oruntuler[i]
            sonraki_durum = oruntuler[i+1]
            self.gecisler[guncel_durum][sonraki_durum] += 1
        for guncel_durum, sonraki_durumlar in self.gecisler.items():
            toplam = sum(sonraki_durumlar.values())
            if toplam > 0:
                for sonraki_durum in sonraki_durumlar:
                    sonraki_durumlar[sonraki_durum] = sonraki_durumlar[sonraki_durum] / toplam

    def en_yakin_durumu_bul(self, gorulmemis_oruntu):
        en_iyi_durum = None
        min_uzaklik = float('inf')
        for durum in self.durumlar:
            uzaklik = levenshtein_uzakligi(gorulmemis_oruntu, durum)
            if uzaklik < min_uzaklik:
                min_uzaklik = uzaklik
                en_iyi_durum = durum
        return en_iyi_durum, min_uzaklik
