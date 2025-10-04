# ÖDEV 3 - BASİT K-MEANS (UYARI DÜZELTİLMİŞ)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Uyarıyı önlemek için environment variable ayarla
os.environ["OMP_NUM_THREADS"] = "1"

# Basit veri oluştur
np.random.seed(42)
data = {
    'dava_süresi': np.random.randint(1, 36, 50),
    'ceza_süresi': np.random.randint(1, 120, 50)
}
df = pd.DataFrame(data)

print("Veri önizleme:")
print(df.head())

# Ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-Means (n_init parametresi ekleyerek)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Küme'] = kmeans.fit_predict(X_scaled)

print("\nKümeleme sonuçları:")
print(df.head())

# Görselleştir
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = df[df['Küme'] == i]
    plt.scatter(cluster_data['dava_süresi'], cluster_data['ceza_süresi'], 
                color=colors[i], label=f'Küme {i}', alpha=0.7, s=60)

plt.xlabel('Dava Süresi (ay)')
plt.ylabel('Ceza Süresi (ay)')
plt.title('K-Means Kümeleme Sonuçları')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nKüme istatistikleri:")
print(df.groupby('Küme').mean())

print("\nKüme yorumlaması:")
print("Küme 0: Kısa dava süreli davalar")
print("Küme 1: Uzun dava süreli davalar") 
print("Küme 2: Orta düzey davalar")