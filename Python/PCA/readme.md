# **PCA + Random Forest para datos sísmicos**

Este proyecto combina **reducción de dimensionalidad con PCA** y **clasificación con Random Forest** para trabajar con un dataset sísmico desbalanceado. La idea principal es reducir la dimensionalidad de las variables originales, conservar la mayor parte de la varianza y luego entrenar un modelo que sea robusto frente a la rareza de la clase positiva.

## **Cómo funciona este proyecto**

1. **Carga del dataset**
   - Se lee el archivo `seismic-bumps.arff` desde la carpeta de datos del repositorio.
2. **Preprocesamiento**
   - Las columnas categóricas se codifican con `LabelEncoder`.
3. **Estandarización**
   - Se usa `StandardScaler` para normalizar las variables antes del PCA.
4. **Reducción de dimensionalidad**
   - Se aplica `PCA` y se selecciona un número de componentes que conserve aproximadamente el 95% de la varianza.
5. **Balanceo de clases**
   - Se usa `SMOTE` para reequilibrar la muestra de entrenamiento.
6. **Entrenamiento del modelo**
   - Se ajusta un `RandomForestClassifier` con pesos de clase y parámetros configurados para manejar datos desbalanceados.
7. **Evaluación**
   - Se revisan reportes, matrices de confusión, ROC y Precision-Recall para medir la calidad predictiva.
8. **Selección de umbral**
   - Se calcula un umbral óptimo según F2-score para priorizar la detección de la clase minoritaria.

```python
pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

X_train_pca = PCA(n_components=n_components_95).fit_transform(X_train_scaled)
```

## **Bondades para predicción de sismos**

### 1. **Reducir ruido y dimensionalidad**
```python
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(X_train_scaled)
```
**Ventaja**: elimina redundancia y conserva la mayor parte de la información útil.

### 2. **Manejo del desbalance**
```python
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
```
**Ventaja**: hace que el modelo no ignore la clase rara de eventos sísmicos peligrosos.

### 3. **Modelo robusto para relaciones complejas**
```python
RandomForestClassifier(
    n_estimators=200,
    class_weight=class_weight_dict,
    max_depth=10,
    min_samples_split=5
)
```
**Ventaja**: combina buenas propiedades no lineales con interpretabilidad y estabilidad.

### 4. **Métricas más adecuadas para clases raras**
```python
precision_recall_curve(y_test, y_pred_proba)
roc_curve(y_test, y_pred_proba)
```
**Ventaja**: evita confiar solo en la precisión global cuando la clase positiva es muy minoritaria.

### 5. **Umbral adaptado al problema**
```python
optimal_threshold = thresholds[np.argmax(f2_scores)]
```
**Ventaja**: permite priorizar la detección de la clase rara sin perder demasiado equilibrio.

## **Fenomenologías donde aplicar esta técnica**

### 1. **Predicción de eventos sísmicos peligrosos**
- **Similitud**: las clases están desbalanceadas y los eventos raros son críticos.
- **Variables**: energía, pulsos, desviaciones, intensidad, contadores por rango.
- **Ejemplo**: anticipar estados de alto riesgo en minería.

### 2. **Análisis de señales con muchas variables**
- **Similitud**: hay muchas columnas correlacionadas entre sí.
- **Variables**: métricas físicas, históricas y de intensidad.
- **Ejemplo**: resumir la información sin perder capacidad predictiva.

### 3. **Detección de escenarios raros en procesos industriales**
- **Similitud**: datos multivariantes y eventos poco frecuentes.
- **Variables**: señales de alarma, vibración, consumo, energía.
- **Ejemplo**: detectar patrones críticos antes de fallos operativos.

### 4. **Clasificaciones con costo asimétrico**
- **Similitud**: un falso negativo puede ser más costoso que un falso positivo.
- **Variables**: todas las del problema base.
- **Ejemplo**: priorizar sensibilidad sobre precisión global.

## **Patrón común en todas estas aplicaciones**

### Características compartidas:
1. **Hay variables redundantes** y correlacionadas.
2. **La clase positiva es rara**.
3. **La precisión global no es suficiente** para evaluar el modelo.
4. **Hace falta combinar reducción de dimensionalidad y manejo del desbalance**.

### Ventaja clave de este pipeline:
```python
# Para estos casos, PCA + Random Forest ofrece:
1. reducción de dimensionalidad útil
2. mejor manejo del desbalanceo
3. métricas robustas para clases raras
4. umbral óptimo para la detección real
```

## **Requisitos**

- Python 3.7 o superior
- Librerías:

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

## **Cómo ejecutar el proyecto**

Desde la raíz del repositorio:

```bash
cd /ruta/al/repositorio/Master
python Python/PCA/PCA.py
```

O desde la carpeta del script:

```bash
cd /ruta/al/repositorio/Master/Python/PCA
python PCA.py
```

## **Ruta del dataset**

El script busca el archivo automáticamente dentro del proyecto. La ruta esperada es:

```text
Python/data/seismic-bumps.arff
```

## **Salidas esperadas**

Durante la ejecución se muestran:

- gráfico del método de varianza acumulada,
- curva ROC,
- curva Precision-Recall,
- matriz de confusión,
- análisis de umbral óptimo,
- evaluación del rendimiento del modelo.

## **Propósito general del script**

Este proyecto permite:
- reducir la dimensionalidad de los datos sísmicos,
- manejar mejor la escasez de la clase positiva,
- entrenar un modelo de clasificación con buen rendimiento para eventos raros,
- evaluar el resultado con métricas apropiadas para problemas desbalanceados.

En resumen, esta versión del proyecto combina PCA con un clasificador robusto para extraer estructura útil del dataset y mejorar la capacidad de detección de eventos sísmicos raros.
