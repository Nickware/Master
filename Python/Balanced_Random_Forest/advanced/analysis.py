# Cargando las librerias
from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_FILE = Path(__file__).resolve().parents[2] / 'data' / 'seismic-bumps.arff'
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f'No se encontró el dataset en: {DATA_FILE}. '
        'Verifica que la ruta del repositorio sea correcta.'
    )


# Cargando los datos
def load_data(nombres):
    input_data, _ = arff.loadarff(str(DATA_FILE))
    df = pd.DataFrame(input_data)
    df.columns = nombres
    return df


# Rescribiendo las etiquetas
nombres = [
    'seismic',
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
    'class'
]

# Cargando los datos
df = load_data(nombres)
print('Dimensiones del dataset:', df.shape)
print('\nDistribución de clases original:')
print(df['class'].value_counts())

# Transformando los datos categóricos de las etiquetas a valores entre 0 y 1
def preprocess_features(df, cols):
    """Transforma columnas categóricas usando LabelEncoder."""
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    return df

# Preprocesar columnas categóricas
cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
df = preprocess_features(df, cat_cols)

# =============================================================================
# ANÁLISIS EXPLORATORIO AMPLIADO
# =============================================================================
print('=' * 70)
print('ANÁLISIS EXPLORATORIO COMPLETO DEL DATASET')
print('=' * 70)

# 1. Información general del dataset
print('\n1. INFORMACIÓN GENERAL:')
print(f'Número total de instancias: {len(df)}')
print(f'Número de features: {len(df.columns) - 1}')
print(f'Proporción de clases: {np.mean(df["class"] == 1) * 100:.2f}% sismos')

# 2. Estadísticas descriptivas detalladas por clase
print('\n2. ESTADÍSTICAS DESCRIPTIVAS POR CLASE:')
for clase in [0, 1]:
    clase_nombre = 'No Sismo' if clase == 0 else 'Sismo'
    clase_data = df[df['class'] == clase]

    print(f"\n{'=' * 40}")
    print(f'CLASE {clase} - {clase_nombre}')
    print(f"{'=' * 40}")
    print(f'Número de instancias: {len(clase_data)} ({len(clase_data) / len(df) * 100:.1f}%)')

    numeric_features = [
        'genergy', 'gdenergy', 'energy', 'maxenergy', 'nbumps',
        'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7'
    ]

    print('\nMedias de features importantes:')
    for feat in numeric_features:
        if feat in df.columns:
            mean_val = clase_data[feat].mean()
            std_val = clase_data[feat].std()
            print(f'  {feat:15s}: {mean_val:6.2f} ± {std_val:5.2f}')

# 3. Análisis de correlaciones
print('\n3. ANÁLISIS DE CORRELACIONES:')
correlaciones = []
for feature in nombres[:-1]:
    correlation = np.corrcoef(df[feature], df['class'])[0, 1]
    correlaciones.append((feature, correlation))

correlaciones.sort(key=lambda x: abs(x[1]), reverse=True)
print('\nTop 10 features más correlacionadas con la clase:')
print('-' * 50)
for feat, corr in correlaciones[:10]:
    signo = '(+)' if corr > 0 else '(-)'
    print(f'  {feat:15s}: {corr:7.3f} {signo}')

# 4. Análisis de valores faltantes y tipos de datos
print('\n4. INFORMACIÓN DE CALIDAD DE DATOS:')
print(f'Valores faltantes totales: {df.isnull().sum().sum()}')
print('\nTipos de datos:')
print(df.dtypes)

# 5. Distribución de features categóricas
print('\n5. DISTRIBUCIÓN DE FEATURES CATEGÓRICAS:')
categorical_features = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
for feat in categorical_features:
    if feat in df.columns:
        print(f'\nDistribución de {feat}:')
        for clase in [0, 1]:
            clase_data = df[df['class'] == clase]
            counts = clase_data[feat].value_counts().sort_index()
            print(f'  Clase {clase}: {dict(counts)}')

# =============================================================================
# VISUALIZACIONES
# =============================================================================
print('\n6. VISUALIZACIONES...')

plt.style.use('seaborn-v0_8')
sns.set_palette('Set2')

# 6.1 Distribución de features clave entre clases
plt.figure(figsize=(16, 12))
features_clave = ['energy', 'maxenergy', 'nbumps', 'genergy', 'gdenergy', 'seismic']

for i, feat in enumerate(features_clave, 1):
    plt.subplot(2, 3, i)
    for clase in [0, 1]:
        data = df[df['class'] == clase][feat]
        sns.histplot(data, kde=True, alpha=0.6, label=f'Clase {clase}')
    plt.xlabel(feat)
    plt.ylabel('Densidad')
    plt.legend()
    plt.title(f'Distribución de {feat} por clase')

plt.tight_layout()
plt.suptitle('Distribución de Features Clave por Clase', fontsize=16, y=1.02)
plt.show()

# 6.2 Boxplots de features importantes
plt.figure(figsize=(14, 10))
important_numeric = ['energy', 'maxenergy', 'nbumps', 'genergy', 'gdenergy', 'ghazard']

for i, feat in enumerate(important_numeric, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='class', y=feat, data=df)
    plt.title(f'{feat} por clase')
    plt.xlabel('Clase (0=No Sismo, 1=Sismo)')
    plt.ylabel(feat)

plt.tight_layout()
plt.suptitle('Boxplots de Features Numéricas por Clase', fontsize=16, y=1.02)
plt.show()

# 6.3 Matriz de correlación
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
plt.title('Matriz de Correlación de Features', fontsize=16)
plt.tight_layout()
plt.show()

# 6.4 Análisis de outliers
print('\n7. ANÁLISIS DE OUTLIERS:')
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'class':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        print(f'{col:15s}: {len(outliers):3d} outliers ({len(outliers)/len(df)*100:.1f}%)')

# =============================================================================
# PREPARACIÓN DE DATOS PARA MODELADO
# =============================================================================
print('\n' + '=' * 70)
print('PREPARACIÓN DE DATOS PARA MODELADO')
print('=' * 70)

# Separar features y target
X = df.iloc[:, 0:18].values
y = df.iloc[:, 18].values

print(f'Shape de X: {X.shape}, Shape de y: {y.shape}')
print('Proporción de clases: Clase 0: {:.2f}%, Clase 1: {:.2f}%'.format(
    np.sum(y == 0) / len(y) * 100,
    np.sum(y == 1) / len(y) * 100
))

# Dividiendo los datos con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('\nDistribución después de train_test_split (stratify):')
print('Entrenamiento - Clase 0: {}, Clase 1: {}'.format(
    np.sum(y_train == 0), np.sum(y_train == 1)))
print('Prueba - Clase 0: {}, Clase 1: {}'.format(
    np.sum(y_test == 0), np.sum(y_test == 1)))

# Calcular pesos de clases para balancear
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f'\nPesos calculados para balanceo: {class_weight_dict}')

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# ENTRENAMIENTO DEL MODELO
# =============================================================================
print('\n' + '=' * 70)
print('ENTRENAMIENTO DEL MODELO')
print('=' * 70)

# Ajustar Random Forest con pesos balanceados
print('\nEntrenando RandomForestClassifier con class_weight...')
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# =============================================================================
# EVALUACIÓN DEL MODELO
# =============================================================================
print('\n' + '=' * 70)
print('EVALUACIÓN DEL MODELO')
print('=' * 70)

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['No Sismo', 'Sismo']))

print("Matriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualización de la matriz de confusión
plt.figure(figsize=(15, 5))

# Matriz de confusión
plt.subplot(1, 3, 1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Sismo', 'Sismo'])
disp.plot(cmap='Blues', ax=plt.gca())
plt.title('Matriz de Confusión')

# Curva ROC
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")

# Features importantes
plt.subplot(1, 3, 3)
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1]
plt.barh(range(10), importancias[indices][:10], align='center')
plt.yticks(range(10), [nombres[i] for i in indices[:10]])
plt.xlabel('Importancia')
plt.title('Top 10 Features más Importantes')

plt.tight_layout()
plt.show()

# Mostrar ranking completo de features
print("\nRanking de Features por Importancia:")
print("-" * 50)
for i, idx in enumerate(indices):
    print(f"{i+1:2d}. {nombres[idx]:15s}: {importancias[idx]:.4f}")

# Análisis de probabilidades
print("\nAnálisis de probabilidades predichas:")
sismo_probs = y_pred_proba[y_test == 1]
no_sismo_probs = y_pred_proba[y_test == 0]

print(f"Probabilidad promedio para casos REALES de sismo: {sismo_probs.mean():.3f}")
print(f"Probabilidad promedio para casos de no-sismo: {no_sismo_probs.mean():.3f}")
print(f"Máxima probabilidad asignada a un sismo real: {sismo_probs.max():.3f}")
print(f"Mínima probabilidad asignada a un sismo real: {sismo_probs.min():.3f}")

# Umbral alternativo
umbral_optimizado = 0.3
y_pred_optimizado = (y_pred_proba >= umbral_optimizado).astype(int)

print(f"\nResultados con umbral optimizado ({umbral_optimizado}):")
print(classification_report(y_test, y_pred_optimizado, target_names=['No Sismo', 'Sismo']))

print("\n¡Análisis completado! 🎯")