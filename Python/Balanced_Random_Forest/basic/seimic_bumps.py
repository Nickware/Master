# Cargando las librerias
import pandas as pd 
from scipy.io import arff
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier  # Cambiado a RandomForest estándar
from sklearn.utils import class_weight  # Importamos class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Cargando los datos
def load_data(nombres):
    data = '../data/seismic-bumps.arff'
    input_data, input_meta = arff.loadarff(data)
    df = pd.DataFrame(input_data)
    df.columns = nombres
    return df

# Rescribiendo las etiquetas
nombres = ['seismic',  
'seismoacoustic', 
'shift', 
'genergy', 
'gpuls', 
'gdenergy', 
'gdpuls', 
'ghazard',
'nbumps',
'nbumps2',
'nbumps3',
'nbumps4',
'nbumps5',
'nbumps6',
'nbumps7',
'nbumps89',
'energy',
'maxenergy',
'class']

# Cargando los datos
df = load_data(nombres)
print("Dimensiones del dataset:", df.shape)
print("\nDistribución de clases original:")
print(df['class'].value_counts())

# Transformando los datos categóricos de las etiquetas a valores entre 0 y 1
def preprocess_features(df, cols):
    """transform categorical features"""
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    return df

# Preprocesar columnas categóricas
cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
df = preprocess_features(df, cat_cols)

# Separar features y target
X = df.iloc[:, 0:18].values  # Todas las features (18 columnas)
y = df.iloc[:, 18].values    # Target 'class'

print(f"\nShape de X: {X.shape}, Shape de y: {y.shape}")
print("Proporción de clases: Clase 0: {:.2f}%, Clase 1: {:.2f}%".format(
    np.sum(y == 0)/len(y)*100, np.sum(y == 1)/len(y)*100))

# Dividiendo los datos con estratificación para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\nDistribución después de train_test_split (stratify):")
print("Entrenamiento - Clase 0: {}, Clase 1: {}".format(
    np.sum(y_train == 0), np.sum(y_train == 1)))
print("Prueba - Clase 0: {}, Clase 1: {}".format(
    np.sum(y_test == 0), np.sum(y_test == 1)))

# Calcular pesos de clases para balancear
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\nPesos calculados para balanceo: {class_weight_dict}")

# Escalar los datos (pero NO aplicar PCA)
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# Ajustar Random Forest con pesos balanceados
print("\nEntrenando RandomForestClassifier con class_weight...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weight_dict,  # Usamos los pesos calculados
    n_jobs=-1  # Usa todos los cores disponibles
)
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para clase 1

# Métricas de evaluación
print("\n" + "="*50)
print("EVALUACIÓN DEL MODELO")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Sismo', 'Sismo']))

print("Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Sismo', 'Sismo'])
disp.plot(cmap='Blues', ax=plt.gca())
plt.title('Matriz de Confusión')

# Análisis de features importantes
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1]

plt.subplot(1, 2, 2)
plt.barh(range(15), importancias[indices][:15], align='center')
plt.yticks(range(15), [nombres[i] for i in indices[:15]])
plt.xlabel('Importancia')
plt.title('Top 15 Features más Importantes')
plt.tight_layout()
plt.show()

# Mostrar ranking completo de features
print("\nRanking de Features por Importancia:")
print("-" * 40)
for i, idx in enumerate(indices):
    print(f"{i+1:2d}. {nombres[idx]:15s}: {importancias[idx]:.4f}")

# Análisis de las probabilidades predichas para la clase minoritaria
print("\nAnálisis de probabilidades predichas para Sismos (Clase 1):")
sismo_probs = y_pred_proba[y_test == 1]
no_sismo_probs = y_pred_proba[y_test == 0]

print(f"Probabilidad promedio para casos REALES de sismo: {sismo_probs.mean():.3f}")
print(f"Probabilidad promedio para casos de no-sismo: {no_sismo_probs.mean():.3f}")
print(f"Máxima probabilidad asignada a un sismo real: {sismo_probs.max():.3f}")

# Umbral alternativo para mejorar detección de sismos
umbral_optimizado = 0.3  # Más bajo que 0.5 para detectar más sismos
y_pred_optimizado = (y_pred_proba >= umbral_optimizado).astype(int)

print(f"\nResultados con umbral optimizado ({umbral_optimizado}):")
print(classification_report(y_test, y_pred_optimizado, target_names=['No Sismo', 'Sismo']))