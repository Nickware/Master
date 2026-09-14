# **Clustering con K-Means**

El **K-Means** es un algoritmo de aprendizaje no supervisado que busca agrupar observaciones similares en grupos (clusters) según sus características. En el contexto de datos sísmicos, permite identificar patrones de comportamiento que no son evidentes a simple vista y facilita la segmentación de estados mineros, riesgos o comportamientos repetidos.

## **Cómo funciona este proyecto**

1. **Carga del dataset**
   - Se lee el archivo `seismic-bumps.arff` desde la carpeta de datos del proyecto.
2. **Preprocesamiento**
   - Las columnas categóricas se convierten a valores numéricos con `LabelEncoder`.
3. **Selección de variables**
   - Se usa `mutual_info_classif` para identificar las 5 variables más informativas.
4. **Estandarización**
   - Las variables seleccionadas se escalan con `StandardScaler` para evitar que variables con mayor escala dominen la distancia.
5. **Determinación del número de clusters**
   - Se aplica el método del codo sobre la suma de distancias intra-cluster (WCSS).
6. **Ajuste de K-Means**
   - Se construye un modelo con 3 clusters.
7. **Evaluación del clustering**
   - Se mide la calidad del agrupamiento con `silhouette_score`.
8. **Visualización**
   - Se añade la columna `cluster` al DataFrame y se proyecta el resultado usando PCA en 3D.

```python
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
```

## **Bondades para análisis de sismos**

### 1. **Descubrimiento de patrones ocultos**
```python
# Agrupa observaciones de similar comportamiento
clusters = kmeans.fit_predict(X_scaled)
```
**Ventaja**: permite encontrar grupos de eventos sin necesidad de etiquetas previas.

### 2. **Segmentación automática del comportamiento sísmico**
```python
# Se obtiene una etiqueta por cada registro
cluster = pd.DataFrame(y_kmeans + 1, columns=['cluster'])
```
**Ventaja**: cada observación queda asociada a un tipo de comportamiento o estado.

### 3. **Reducción de la complejidad para interpretación**
```python
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
```
**Ventaja**: hace más legible la distribución del conjunto de datos antes de extraer conclusiones.

### 4. **Evaluación cuantitativa del agrupamiento**
```python
score = silhouette_score(X_scaled, y_kmeans)
print(f'Silhouette Score: {score:.3f}')
```
**Ventaja**: ayuda a saber si la segmentación es razonable o si el número de clusters debe ajustarse.

### 5. **Interpretación de clusters por variables**
```python
kmeans_mean_cluster = df.groupby('cluster').mean().round(1)
print(kmeans_mean_cluster)
```
**Ventaja**: permite estudiar qué variables diferencian a cada grupo y detectar perfiles de riesgo.

### 6. **Visualización de grupos en espacio reducido**
```python
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_kmeans, cmap='viridis')
```
**Ventaja**: facilita la inspección espacial de las agrupaciones y la comparación entre clusters.

## **Fenomenologías donde aplicar esta técnica**

### 1. **Clasificación de estados sísmicos**
- **Similitud**: existen grupos con niveles distintos de riesgo o energía.
- **Variables**: energía, pulsos, desviaciones, incidentes por turno.
- **Ejemplo**: detectar patrones típicos de riesgo alto, medio y bajo.

### 2. **Agrupación de patrones de actividad minera**
- **Similitud**: varios turnos pueden compartir comportamiento similar.
- **Variables**: `nbumps`, `energy`, `maxenergy`, `ghazard`.
- **Ejemplo**: identificar días o turnos con actividad anómala.

### 3. **Detección de anomalías operativas**
- **Similitud**: los eventos fuera del patrón habitual aparecen como clusters pequeños o aislados.
- **Variables**: medidas de energía y pulsos registradas por geófonos.
- **Ejemplo**: encontrar comportamientos inusuales en la actividad sísmica.

### 4. **Comparación de condiciones de turno**
- **Similitud**: distintas condiciones de operación generan perfiles distintos.
- **Variables**: tipo de turno, nivel de energía, cantidad de eventos.
- **Ejemplo**: comparar desempeño entre turnos o zonas.

### 5. **Segmentación de comportamiento geológico**
- **Similitud**: varios subtipos de respuesta sísmica pueden coexistir.
- **Variables**: combinación de indicadores de riesgo y energía.
- **Ejemplo**: distinguir tipos de actividad sísmica con distinta intensidad.

## **Patrón común en todas estas aplicaciones**

### Características compartidas:
1. **Hay agrupaciones naturales** en los datos.
2. **Las observaciones similares tienden a agruparse** por sus variables.
3. **La interpretación del cluster es más útil** cuando se combina con análisis descriptivo.
4. **Es necesario evaluar la calidad del clustering** con métricas como silhouette score.
5. **La reducción de dimensionalidad** ayuda a entender la estructura del problema.

### Ventaja clave de K-Means:
```python
# Para estos casos, K-Means ofrece:
1. segmentación automática
2. patrones de comportamiento
3. visualización interpretativa
4. agrupación útil para análisis posterior
```

## **Requisitos**

- Python 3.7 o superior
- Librerías:

```bash
pip install pandas scipy scikit-learn matplotlib
```

## **Cómo ejecutar el proyecto**

Desde la raíz del repositorio:

```bash
cd /ruta/al/repositorio/Master
python Python/Cluster/kmeans.py
```

O desde la carpeta del script:

```bash
cd /ruta/al/repositorio/Master/Python/Cluster
python kmeans.py
```

## **Ruta del dataset**

El script busca el archivo de manera automática dentro del proyecto. La ruta esperada es:

```text
Python/data/seismic-bumps.arff
```

Si el archivo no existe, revisa que la jerarquía del repositorio siga este patrón.

## **Salidas esperadas**

Durante la ejecución se muestran:

- la gráfica del método del codo,
- el valor del `Silhouette Score`,
- la tabla de medias por cluster,
- la visualización 3D de los grupos con PCA.

## **Propósito general del script**

Este proyecto permite:
- identificar grupos naturales dentro de los datos sísmicos,
- explorar patrones de riesgo y anomalías,
- interpretar qué variables distinguen cada cluster,
- preparar una base visual y descriptiva para análisis posteriores.

En resumen, es una herramienta útil para entender la estructura interna de un conjunto de observaciones sísmicas y detectar patrones de comportamiento relevantes para la gestión del riesgo.

