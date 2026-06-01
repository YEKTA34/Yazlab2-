import json

class AciklanabilirlikModulu:
    def __init__(self, otomata, anomali_esigi=0.05):
        self.otomata = otomata
        self.anomali_esigi = anomali_esigi
        self.yol_olasiligi = 1.0
        
    def sifirla(self):
        self.yol_olasiligi = 1.0
        
    def adimi_acikla(self, zaman_adimi, sax_gecmisi):
        pencere = self.otomata.pencere_boyutu
        durum_sayisi = min(pencere, zaman_adimi - pencere + 2)
        durumlar = []
        for i in range(durum_sayisi):
            start_idx = zaman_adimi - pencere - durum_sayisi + 2 + i
            end_idx = zaman_adimi - durum_sayisi + 2 + i
            durumlar.append(sax_gecmisi[start_idx:end_idx])
            
        guncel_oruntu = durumlar[-1]
        onceki_durum = durumlar[-2] if len(durumlar) > 1 else ""
        
        guncel_durum_bilgisi = "gorulen"
        eslesen_guncel = guncel_oruntu
        uzaklik = 0
        if guncel_oruntu not in self.otomata.durumlar:
            guncel_durum_bilgisi = "gorulmeyen"
            eslesen_guncel, uzaklik = self.otomata.en_yakin_durumu_bul(guncel_oruntu)
            
        gecisler_listesi = []
        yol_olasiligi = 1.0
        
        for j in range(len(durumlar) - 1):
            s_baslangic = durumlar[j]
            s_bitis = durumlar[j+1]
            
            eslesen_baslangic = s_baslangic
            if s_baslangic not in self.otomata.durumlar:
                eslesen_baslangic, _ = self.otomata.en_yakin_durumu_bul(s_baslangic)
                
            eslesen_bitis = s_bitis
            if s_bitis not in self.otomata.durumlar:
                eslesen_bitis, _ = self.otomata.en_yakin_durumu_bul(s_bitis)
                
            gecis_olasiligi = 1.0e-5
            if eslesen_baslangic in self.otomata.gecisler:
                gecis_olasiligi = self.otomata.gecisler[eslesen_baslangic].get(eslesen_bitis, 1.0e-5)
                
            yol_olasiligi *= gecis_olasiligi
            gecisler_listesi.append({
                "baslangic_durumu": s_baslangic,
                "hedef_durum": s_bitis,
                "eslesen_baslangic": eslesen_baslangic,
                "eslesen_hedef": eslesen_bitis,
                "olasilik": float(gecis_olasiligi)
            })
            
        last_trans_prob = gecisler_listesi[-1]["olasilik"] if gecisler_listesi else 1.0
        
        karar = "normal"
        gerekce = "Normal davranis"
        if guncel_durum_bilgisi == "gorulmeyen":
            karar = "anomali"
            gerekce = "Gorulmeyen oruntu tespit edildi"
        elif last_trans_prob < self.anomali_esigi:
            karar = "anomali"
            gerekce = "Dusuk gecis olasiligi tespit edildi"
        elif yol_olasiligi < (self.anomali_esigi ** len(gecisler_listesi)) if gecisler_listesi else 1.0:
            karar = "anomali"
            gerekce = "Dusuk yol olasiligi tespit edildi"
            
        guven_skoru = yol_olasiligi
        guven_seviyesi = "Yuksek" if karar == "normal" and guven_skoru >= 1e-4 else "Dusuk"
        
        sonuc = {
            "zaman_adimi": int(zaman_adimi),
            "durum": onceki_durum,
            "oruntu": guncel_oruntu,
            "durum_bilgisi": guncel_durum_bilgisi,
            "eslesen_durum": eslesen_guncel,
            "uzaklik": int(uzaklik),
            "gecisler": gecisler_listesi,
            "yol_olasiligi": float(yol_olasiligi),
            "karar": karar,
            "guven_skoru": float(guven_skoru),
            "guven_seviyesi": guven_seviyesi,
            "gerekce": gerekce
        }
        
        return sonuc
        
    def jsona_donustur(self, aciklama_sozlugu):
        return json.dumps(aciklama_sozlugu, indent=2)
