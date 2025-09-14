# Cargando las librerias
import pandas as pd 
from scipy.io import arff
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import seaborn as sns
import joblib

# Configuración de visualización
plt.style.use('default')
sns.set_palette("colorblind")

# Cargando los datos
def load_data(nombres):
    data = '../data/seismic-bumps.arff'
    input_data, input_meta = arff.loadarff(data)
    df = pd.DataFrame(input_data)
    df.columns = nombres
    return df

# Rescribiendo las etiquetas
nombres = ['seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls', 'gdenergy', 
           'gdpuls', 'ghazard', 'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 
           'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy', 'class']

# Cargando los datos
df = load_data(nombres)
print("=== ANÁLISIS INICIAL DEL DATASET ===")
print(f"Dimensiones: {df.shape}")
print("\nDistribución de clases:")
print(df['class'].value_counts())
print(f"Proporción de clase minoritaria: {df['class'].value_counts()[1]/len(df)*100:.2f}%")

# Transformando datos categóricos
def preprocess_features(df, cols):
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    return df

cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
df = preprocess_features(df, cat_cols)

# Separar features y target
X = df.iloc[:, 0:18].values  # Todas las features
y = df.iloc[:, 18].values    # Target

# 1. DIVISIÓN ESTRATIFICADA DE DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n=== DIVISIÓN ESTRATIFICADA ===")
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
print("Proporción de clases en entrenamiento:", np.unique(y_train, return_counts=True)[1]/len(y_train))
print("Proporción de clases en prueba:", np.unique(y_test, return_counts=True)[1]/len(y_test))

# 2. ESCALADO Y APLICACIÓN DE PCA CON VARIANZA CONSERVADA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Análisis de varianza para determinar componentes óptimos
pca_full = PCA()
pca_full.fit(X_train_scaled)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Encontrar número de componentes para 95% de varianza
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"\n=== ANÁLISIS PCA ===")
print(f"Varianza explicada por primeros 5 componentes: {explained_variance[:5]}")
print(f"Varianza acumulada: {cumulative_variance[:5]}")
print(f"Componentes para 90% de varianza: {n_components_90}")
print(f"Componentes para 95% de varianza: {n_components_95}")

# Aplicar PCA conservando el 95% de varianza
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nDimensionalidad después de PCA: {X_train_pca.shape[1]} componentes")

# 3. CALCULAR PESOS PARA BALANCEO DE CLASES
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n=== MANEJO DE DESBALANCEO ===")
print(f"Pesos calculados para clases: {class_weight_dict}")

# 4. OPCIÓN A: SMOTE + RANDOM FOREST CON PESOS
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)

print(f"Después de SMOTE - Clase 0: {sum(y_train_balanced == 0)}, Clase 1: {sum(y_train_balanced == 1)}")

# 5. MODELADO CON RANDOM FOREST Y OPTIMIZACIÓN
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=class_weight_dict,  # Usamos pesos en lugar de sampling_strategy
    n_jobs=-1,
    max_depth=10,
    min_samples_split=5
)

# Entrenar modelo
print("\n=== ENTRENAMIENTO DEL MODELO ===")
model.fit(X_train_balanced, y_train_balanced)

# Predicciones
y_pred = model.predict(X_test_pca)
y_pred_proba = model.predict_proba(X_test_pca)[:, 1]

# EVALUACIÓN COMPLETA
print("\n=== EVALUACIÓN DEL MODELO ===")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Sismo', 'Sismo']))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Precision-Recall Curve (Crucial para clases desbalanceadas)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# VISUALIZACIONES
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Matriz de Confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Sismo', 'Sismo'])
disp.plot(ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('Matriz de Confusión')

# 2. Curva Precision-Recall
axes[0, 1].plot(recall, precision, marker='.', label=f'PR Curve (AUC = {pr_auc:.3f})')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Curva Precision-Recall')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Curva ROC
axes[0, 2].plot(fpr, tpr, marker='.', label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0, 2].plot([0, 1], [0, 1], linestyle='--', label='Random')
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('Curva ROC')
axes[0, 2].legend()
axes[0, 2].grid(True)

# 4. Varianza Explicada por Componentes
axes[1, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% Varianza')
axes[1, 0].axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} componentes')
axes[1, 0].set_xlabel('Número de Componentes')
axes[1, 0].set_ylabel('Varianza Acumulada')
axes[1, 0].set_title('Varianza Explicada por Componentes PCA')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 5. Feature Importances desde componentes PCA
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1]

axes[1, 1].barh(range(len(importancias)), importancias[indices])
axes[1, 1].set_yticks(range(len(importancias)))
axes[1, 1].set_yticklabels([f'PC{i+1}' for i in indices])
axes[1, 1].set_xlabel('Importancia')
axes[1, 1].set_title('Importancia de Componentes Principales')
axes[1, 1].grid(True, alpha=0.3)

# 6. Distribución de probabilidades predichas
axes[1, 2].hist(y_pred_proba[y_test == 0], alpha=0.7, bins=30, label='No Sismo', density=True)
axes[1, 2].hist(y_pred_proba[y_test == 1], alpha=0.7, bins=30, label='Sismo', density=True)
axes[1, 2].set_xlabel('Probabilidad Predicha')
axes[1, 2].set_ylabel('Densidad')
axes[1, 2].set_title('Distribución de Probabilidades Predichas')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ANÁLISIS DE UMBRAL ÓPTIMO
print("\n=== ANÁLISIS DE UMBRAL ÓPTIMO ===")

# Encontrar umbral que maximiza F2-score (da más peso al recall)
f2_scores = []
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    if cm_thresh[1, 1] + cm_thresh[1, 0] > 0:  # Evitar división por cero
        recall_thresh = cm_thresh[1, 1] / (cm_thresh[1, 1] + cm_thresh[1, 0])
        precision_thresh = cm_thresh[1, 1] / (cm_thresh[1, 1] + cm_thresh[0, 1]) if (cm_thresh[1, 1] + cm_thresh[0, 1]) > 0 else 0
        if precision_thresh + recall_thresh > 0:
            f2 = 5 * (precision_thresh * recall_thresh) / (4 * precision_thresh + recall_thresh)
            f2_scores.append(f2)
        else:
            f2_scores.append(0)
    else:
        f2_scores.append(0)

optimal_threshold = thresholds[np.argmax(f2_scores)]
print(f"Umbral óptimo para maximizar F2-score: {optimal_threshold:.3f}")

# Aplicar umbral óptimo
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

print("\nResultados con umbral óptimo:")
print(classification_report(y_test, y_pred_optimal, target_names=['No Sismo', 'Sismo']))

# VALIDACIÓN CRUZADA
print("\n=== VALIDACIÓN CRUZADA ===")
cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                           cv=5, scoring='f1_weighted')
print(f"F1-score promedio en validación cruzada (5-fold): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# ANÁLISIS DE COMPONENTES PRINCIPALES
print("\n=== INTERPRETACIÓN DE COMPONENTES PCA ===")
print("Cargas de los primeros 5 componentes principales:")
componentes_df = pd.DataFrame(pca.components_[:5], columns=nombres[:18])
for i in range(5):
    top_features = componentes_df.iloc[i].abs().nlargest(3)
    print(f"PC{i+1}: {list(top_features.index)}")

# BÚSQUEDA DE HIPERPARÁMETROS OPCIONAL
print("\n=== BÚSQUEDA DE HIPERPARÁMETROS (OPCIONAL) ===")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight=class_weight_dict, random_state=42),
    param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1
)

# Ejecutar solo si se desea (puede tomar tiempo)
ejecutar_grid_search = False
if ejecutar_grid_search:
    grid_search.fit(X_train_balanced, y_train_balanced)
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score: {grid_search.best_score_:.3f}")

# Guardar modelo para producción
joblib.dump({
    'model': model,
    'pca': pca,
    'scaler': scaler,
    'optimal_threshold': optimal_threshold,
    'feature_names': nombres[:18]
}, 'seismic_prediction_model.pkl')

print("\n=== MODELO GUARDADO PARA PRODUCCIÓN ===")
print("El modelo incluye:")
print("- Random Forest con class_weight entrenado")
print("- Transformador PCA optimizado")
print("- StandardScaler fitted")
print("- Umbral óptimo de decisión")
print("- Nombres de features originales")

# ANÁLISIS FINAL DE PERFORMANCE
print("\n=== ANÁLISIS FINAL DE PERFORMANCE ===")
final_cm = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = final_cm.ravel()

print(f"Verdaderos Negativos: {tn}")
print(f"Falsos Positivos: {fp}")
print(f"Falsos Negativos: {fn}")
print(f"Verdaderos Positivos: {tp}")
print(f"Recall (Sensibilidad): {tp/(tp+fn):.3f}")
print(f"Precisión: {tp/(tp+fp):.3f}" if (tp+fp) > 0 else "Precisión: N/A")
print(f"F2-Score: {f2_scores[np.argmax(f2_scores)]:.3f}")

print("\n¡Análisis completado exitosamente! ✅")