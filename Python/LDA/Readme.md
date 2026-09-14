# **Clasificación supervisada con SVM para eventos sísmicos**

Este proyecto aplica un modelo de clasificación supervisada sobre el dataset de eventos sísmicos `seismic-bumps.arff` con una **SVM (Support Vector Machine)** con kernel RBF. Aunque el nombre de la carpeta es LDA, la implementación real del script no usa LDA sino un clasificador basado en SVM, optimizado para datos desbalanceados.

## **Cómo funciona este proyecto**

1. **Carga del dataset**
   - Se lee el archivo `seismic-bumps.arff` desde la carpeta de datos del repositorio.
2. **Preprocesamiento**
   - Las columnas categóricas se convierten a valores numéricos con `LabelEncoder`.
3. **Escalado**
   - Todas las variables se escalan al rango `[-1, 1]` para evitar que algunas dominen el entrenamiento.
4. **División train/test**
   - Se separa el conjunto manteniendo la proporción de clases con `train_test_split(..., stratify=Y)`.
5. **Optimización de hiperparámetros**
   - Se usa `GridSearchCV` con kernel RBF para ajustar `C` y `gamma`.
6. **Entrenamiento del modelo**
   - El clasificador es `SVC(class_weight='balanced')` para manejar mejor el desbalance entre clases.
7. **Evaluación**
   - Se comparan métricas como precisión, sensibilidad y especificidad.

```python
svc = SVC(class_weight='balanced', probability=False)
clf = GridSearchCV(estimator=svc, param_grid=tuning_parameters, scoring='recall', cv=10)
```

## **Bondades para predicción de sismos**

### 1. **Manejo de clases desbalanceadas**
```python
SVC(class_weight='balanced')
```
**Ventaja**: reduce el problema de que la clase minoritaria (sismos peligrosos) sea ignorada por el modelo.

### 2. **Ajuste automatizado de hiperparámetros**
```python
GridSearchCV(..., scoring='recall')
```
**Ventaja**: busca la mejor combinación de `C` y `gamma` según la métrica relevante para el problema.

### 3. **Métricas apropiadas para riesgo**
```python
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
```
**Ventaja**: no se depende solo de la precisión global; se revisan sensibilidad y especificidad para entender el comportamiento del clasificador.

### 4. **Robustez en datos complejos**
```python
kernel='rbf'
```
**Ventaja**: permite modelar relaciones no lineales entre variables sísmicas y el estado final.

### 5. **Evaluación con dos objetivos de optimización**
```python
scoring='recall'
scoring='roc_auc'
```
**Ventaja**: el pipeline puede buscar alta sensibilidad o un buen equilibrio general según la prioridad del análisis.

## **Fenomenologías donde aplicar esta técnica**

### 1. **Predicción de eventos sísmicos peligrosos**
- **Similitud**: clases muy desbalanceadas y riesgo alto de falsos negativos.
- **Variables**: energía, pulsos, desviaciones de energía, tipo de turno.
- **Ejemplo**: detectar estados de alto riesgo en minería.

### 2. **Detección de anomalías industriales**
- **Similitud**: eventos raros pero críticos.
- **Variables**: señales de vibración, energía, frecuencia, condiciones operativas.
- **Ejemplo**: identificar patrones de peligro antes de un fallo o accidente.

### 3. **Clasificación de riesgo en sistemas complejos**
- **Similitud**: intervienen muchas variables y relaciones no lineales.
- **Variables**: medidas de operación y contexto del sistema.
- **Ejemplo**: decidir si un estado es crítico o no.

### 4. **Modelado en problemas con costo asimétrico**
- **Similitud**: un falso negativo puede ser más grave que un falso positivo.
- **Variables**: todas las relacionadas con la clase de interés.
- **Ejemplo**: priorizar la sensibilidad sobre la precisión global.

## **Patrón común en todas estas aplicaciones**

### Características compartidas:
1. **Clases desbalanceadas** por naturaleza.
2. **Necesidad de priorizar métricas de detección** sobre el accuracy bruto.
3. **Relaciones no lineales** entre variables y la clase objetivo.
4. **Necesidad de una validación sólida** antes de decidir un modelo.

### Ventaja clave de este pipeline:
```python
# Para estos casos, la estrategia ofrece:
1. ajuste robusto de hiperparámetros
2. tratamiento del desbalance
3. evaluación orientada a riesgo
4. capacidad para capturar relaciones no lineales
```

## **Requisitos**

- Python 3.7 o superior
- Librerías:

```bash
pip install pandas scipy scikit-learn
```

## **Cómo ejecutar el proyecto**

Desde la raíz del repositorio:

```bash
cd /ruta/al/repositorio/Master
python Python/LDA/LDA.py
```

O desde la carpeta del script:

```bash
cd /ruta/al/repositorio/Master/Python/LDA
python LDA.py
```

## **Ruta del dataset**

El script busca el archivo en la ubicación correcta del repositorio, que es:

```text
Python/data/seismic-bumps.arff
```

## **Salidas esperadas**

Durante la ejecución se muestran:

- precisión del clasificador,
- sensibilidad,
- especificidad,
- resultados obtenidos con la búsqueda en rejilla optimizando `recall` y `roc_auc`.

## **Propósito general del script**

Este proyecto permite:
- preparar datos sísmicos para clasificación supervisada,
- entrenar un modelo robusto ante clases desbalanceadas,
- optimizar hiperparámetros con validación cruzada,
- evaluar el rendimiento con métricas orientadas a detección real de riesgo.

En resumen, este script es una base útil para problemas binarios con clases desbalanceadas y con prioridad en la detección correcta de eventos raros, aunque la implementación actual corresponde a un pipeline SVM y no a un LDA puro.

