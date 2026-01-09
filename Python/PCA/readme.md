# Predicción de Sismos con PCA 

## **RECOMENDACIONES IMPLEMENTADAS:**

### 1. **PCA con Varianza Conservada (No 2 componentes arbitrarios)**
```python
# Análisis de varianza para determinar componentes óptimos
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
pca = PCA(n_components=n_components_95)  # Conserva 95% de varianza
```

### 2. **Manejo Explícito del Desbalanceo**
```python
# SMOTE para oversampling de la clase minoritaria
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
```

### 3. **Modelo Apropiado para Clases Desbalanceadas**
```python
# Balanced Random Forest en lugar de Regresión Logística
model = BalancedRandomForestClassifier(n_estimators=200, sampling_strategy='auto')
```

### 4. **Evaluación Completa con Métricas Apropiadas**
```python
# Precision-Recall curve (mejor que accuracy para desbalanceo)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# Análisis de umbral óptimo (no solo 0.5)
optimal_threshold = thresholds[np.argmax(f2_scores)]  # Maximiza F2-score
```

Este script mantiene la esencia del original (uso de PCA) pero lo aplica correctamente para el fenómeno de predicción de sismos, priorizando la detección de eventos raros sobre la optimización de accuracy general.
