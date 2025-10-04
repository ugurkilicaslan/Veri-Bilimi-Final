# ÖDEV 4 - VERİ FİLTRELEME & SIRALAMA
import pandas as pd
import numpy as np

# 1. Örnek ülke verisi oluştur
np.random.seed(42)

ülke_verisi = {
    'ülke': ['Türkiye', 'Almanya', 'Fransa', 'İtalya', 'İspanya', 'İngiltere', 'Japonya', 'Kanada', 'Avustralya', 'Brezilya', 
             'Arjantin', 'Meksika', 'Güney Kore', 'Hollanda', 'İsviçre', 'İsveç', 'Norveç', 'Danimarka', 'Finlandiya', 'Rusya',
             'Çin', 'Hindistan', 'ABD', 'Endonezya', 'Pakistan', 'Nijerya', 'Bangladesh', 'Mısır', 'Vietnam', 'Filipinler'],
    'nüfus': [85000000, 83000000, 68000000, 60000000, 47000000, 67000000, 125000000, 38000000, 26000000, 214000000,
              45000000, 126000000, 52000000, 17000000, 8000000, 10000000, 5000000, 6000000, 5000000, 144000000,
              1400000000, 1380000000, 331000000, 273000000, 220000000, 206000000, 164000000, 102000000, 97000000, 110000000],
    'gsyh_kişibaşı': [8500, 45000, 42000, 34000, 30000, 41000, 39000, 43000, 50000, 8700,
                      10500, 9500, 32000, 52000, 80000, 51000, 75000, 61000, 48000, 11000,
                      10500, 2100, 65000, 3900, 1500, 2000, 1900, 3000, 2700, 3300],
    'okuryazarlık_oranı': [96.7, 99.0, 99.0, 99.2, 98.4, 99.0, 99.0, 99.0, 99.0, 93.2,
                          98.1, 95.4, 98.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.7,
                          96.8, 74.4, 99.0, 95.7, 60.0, 62.0, 75.0, 71.0, 95.0, 96.3],
    'alan_km2': [783562, 357022, 551695, 301340, 505990, 242495, 377975, 9984670, 7692024, 8515767,
                 2780400, 1964375, 100210, 41543, 41285, 450295, 323802, 43094, 338424, 17098242,
                 9596961, 3287263, 9833517, 1904569, 881912, 923768, 147570, 1002450, 331212, 300000]
}

df = pd.DataFrame(ülke_verisi)

# Nüfus yoğunluğu hesapla (nüfus/alan)
df['nüfus_yoğunluğu'] = df['nüfus'] / df['alan_km2']

print("ÜLKE VERİSİ ÖN İZLEME:")
print(df.head())
print(f"\nToplam ülke sayısı: {len(df)}")