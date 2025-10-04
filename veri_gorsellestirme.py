import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükleme
df = pd.read_csv('50_Startups.csv')

# 1. R&D harcaması ile kâr ilişkisi
plt.figure(figsize=(10, 6))
plt.scatter(df['R&D Spend'], df['Profit'])
plt.title('R&D Harcaması vs Kâr')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.grid(True)
plt.show()

# 2. Yönetim harcaması ile kâr ilişkisi
plt.figure(figsize=(10, 6))
plt.scatter(df['Administration'], df['Profit'])
plt.title('Yönetim Harcaması vs Kâr')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.grid(True)
plt.show()

# 3. Eyaletlere göre ortalama kârlar
plt.figure(figsize=(10, 6))
df.groupby('State')['Profit'].mean().plot(kind='bar')
plt.title('Eyaletlere Göre Ortalama Kâr')
plt.xlabel('State')
plt.ylabel('Ortalama Profit')
plt.xticks(rotation=45)
plt.show()

# 4. Harcama dağılımlarının boxplot ile karşılaştırılması
plt.figure(figsize=(12, 6))
df[['R&D Spend', 'Administration', 'Marketing Spend']].boxplot()
plt.title('Harcama Dağılımlarının Karşılaştırılması')
plt.ylabel('Harcama Miktarı')
plt.show()