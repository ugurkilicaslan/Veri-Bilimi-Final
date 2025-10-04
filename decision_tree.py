# ÖDEV 2 - DECISION TREE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Örnek veri oluştur
np.random.seed(42)
dava_data = {
    'yas': np.random.randint(18, 70, 100),
    'gelir': np.random.randint(20000, 100000, 100),
    'suç_tipi': np.random.choice(['A', 'B', 'C'], 100),
    'ceza': np.random.randint(1, 10, 100),
    'sonuç': np.random.choice([0, 1], 100)  # 0: beraat, 1: mahkumiyet
}

df = pd.DataFrame(dava_data)

print("VERİ ÖN İZLEME:")
print(df.head())
print(f"\nVeri boyutu: {df.shape}")
print(f"\nSütunlar ve veri tipleri:")
print(df.dtypes)

# 2. Eksik değer kontrolü (sadece sayısal sütunlar için)
print("\nEksik değerler:")
print(df.isnull().sum())

# 3. Kategorik değişkenleri sayısallaştırma
label_encoders = {}
for column in ['suç_tipi']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    print(f"\n{column} kodlaması: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nKodlanmış veri:")
print(df.head())

# 4. Veriyi eğitim ve test olarak ayırma
X = df.drop('sonuç', axis=1)  # Özellikler
y = df['sonuç']  # Hedef değişken

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nEğitim verisi: {X_train.shape}")
print(f"Test verisi: {X_test.shape}")

# 5. Decision Tree modeli
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

# 6. Tahmin ve metrikler
y_pred = model.predict(X_test)

print("\n=== MODEL SONUÇLARI ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Karar ağacını görselleştirme
plt.figure(figsize=(15, 10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=['Beraat', 'Mahkumiyet'], 
          filled=True, 
          rounded=True)
plt.title('Decision Tree - Dava Sonuçları Tahmini')
plt.show()

# 8. Özellik önemlilikleri
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== ÖZELLİK ÖNEMLİLİKLERİ ===")
print(feature_importance)

# Önemlilik grafiği
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Önemlilik Derecesi')
plt.title('Özellik Önemlilikleri')
plt.gca().invert_yaxis()
plt.show()