import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from automata import levenshtein_uzakligi, OlasiliksalOtomata

class TestGorulmemisOruntuYonetimi(unittest.TestCase):
    
    def test_levenshtein_uzakligi_eslesme(self):
        self.assertEqual(levenshtein_uzakligi("abc", "abc"), 0)
        
    def test_levenshtein_uzakligi_degisim(self):
        self.assertEqual(levenshtein_uzakligi("abc", "adc"), 1)
        
    def test_levenshtein_uzakligi_ekleme(self):
        self.assertEqual(levenshtein_uzakligi("abc", "abcd"), 1)
        
    def test_levenshtein_uzakligi_silme(self):
        self.assertEqual(levenshtein_uzakligi("abc", "ab"), 1)
        
    def test_levenshtein_uzakligi_bos(self):
        self.assertEqual(levenshtein_uzakligi("", "abc"), 3)
        self.assertEqual(levenshtein_uzakligi("abc", ""), 3)
        self.assertEqual(levenshtein_uzakligi("", ""), 0)
        
    def test_en_yakin_durum_esleme(self):
        otomata = OlasiliksalOtomata(pencere_boyutu=3, alfabe_boyutu=3)
        otomata.durumlar = {"abc", "def", "ghi"}
        
        en_yakin, uzaklik = otomata.en_yakin_durumu_bul("adc")
        self.assertEqual(en_yakin, "abc")
        self.assertEqual(uzaklik, 1)
        
        en_yakin, uzaklik = otomata.en_yakin_durumu_bul("xef")
        self.assertEqual(en_yakin, "def")
        self.assertEqual(uzaklik, 1)

if __name__ == "__main__":
    unittest.main()
