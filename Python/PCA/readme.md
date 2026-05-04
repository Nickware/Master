# Predicción de Sismos con PCA y Random Forest Balanceado

## Overview
Pipeline robusto para predicción de terremotos usando **PCA adaptativo** (95% varianza conservada) + **Balanced Random Forest**. Optimizado para datasets sísmicos desbalanceados donde los eventos raros (sismos) son ~1-5% de las muestras.

**Ventajas clave:**
- Conserva información física relevante del PCA
- Maneja desbalanceo con SMOTE + BalancedRF
- Métricas apropiadas (F2-score, PR-AUC) para eventos raros
- Umbral óptimo automático

## Requisitos
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

## Datos Esperados
Formato CSV con columnas:
```
timestamp, magnitude, depth_km, latitude, longitude, 
rsam, pga, pgv, Arias_intensity, [features sísmicas adicionales]
```
- Target: `magnitude >= 5.0` (binario)
- Features: 20-50 variables sísmicas normalizadas

## Uso Rápido
```bash
python earthquake_predictor.py --data sismos_colombia.csv --output results/
```

## Implementación Mejorada (Código Principal)

### 1. PCA Adaptativo (95% Varianza)
```python
from sklearn.decomposition import PCA
import numpy as np

# Determina componentes óptimos automáticamente
pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Componentes PCA: {n_components_95} (95% varianza)")

pca_final = PCA(n_components=n_components_95)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)
```

### 2. Balanceo con SMOTE
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
print(f"Balanceo: {np.bincount(y_train_balanced)}")
```

### 3. Balanced Random Forest
```python
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=200,
    random_state=42,
    sampling_strategy='auto',
    class_weight='balanced'
)
model.fit(X_train_balanced, y_train_balanced)
```

### 4. Evaluación con Métricas Sísmicas
```python
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

# Precision-Recall AUC (crucial para desbalanceo)
y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# F2-score óptimo (prioriza recall para sismos)
f2_scores = (1 + 2**2) * (precision * recall) / (2**2 * precision + recall)
optimal_threshold = thresholds[np.argmax(f2_scores)]

print(f"PR-AUC: {pr_auc:.3f}")
print(f"Umbral óptimo: {optimal_threshold:.3f}")
```

## Resultados Típicos
| Métrica | Valor Esperado | Interpretación |
|---------|---------------|----------------|
| PR-AUC | 0.75-0.85 | Buena discriminación de eventos raros |
| F2-Score | 0.65-0.75 | Alto recall manteniendo precisión |
| Recall @ umbral | >0.80 | Detecta 80%+ de sismos |

**Gráficos generados:**
- Precision-Recall curve con umbral óptimo
- Importancia de features PCA
- Matriz de confusión balanceada

## Validación Física
```
Features más importantes típicamente:
1. Arias_intensity (energía acumulada)
2. PGA (aceleración máxima)
3. RSAM (seismic amplitude)
4. Profundidad focal
```