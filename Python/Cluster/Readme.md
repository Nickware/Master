## Clustering Sísmico con K-Means

Este proyecto realiza clustering sobre datos de eventos sísmicos usando el algoritmo K-Means. El objetivo es agrupar los datos en clusters significativos para facilitar el análisis y la interpretación de patrones sísmicos.

### Requisitos

- **Python 3.7 o superior**
- **Librerías necesarias:**
    - pandas
    - scipy
    - scikit-learn
    - matplotlib

Instala las dependencias con:

```bash
pip install pandas scipy scikit-learn matplotlib
```


### Paso a paso de la rutina

1. **Carga de datos**
    - Se carga el archivo `seismic-bumps.arff` ubicado en `../data/`.
    - Se renombran las columnas según la estructura definida en el código.
2. **Preprocesamiento**
    - **Codificación de variables categóricas:** Las variables categóricas se transforman a valores numéricos usando `LabelEncoder`.
    - **Separación de características y variable objetivo:** Se separan las características (`X`) de la variable objetivo (`y`).
3. **Selección de características**
    - Se calcula la importancia de cada característica usando `mutual_info_classif`.
    - Se seleccionan las 5 características más importantes para el clustering.
4. **Estandarización**
    - Las características seleccionadas se estandarizan para que tengan media cero y varianza unitaria.
5. **Determinación del número de clusters**
    - Se aplica el **método del codo** para encontrar el número óptimo de clusters.
    - Se grafica la suma de las distancias cuadradas intra-cluster (WCSS) frente al número de clusters.
6. **Clustering con K-Means**
    - Se ajusta el modelo K-Means con 3 clusters.
    - Se predice la pertenencia a cada cluster para cada muestra.
7. **Evaluación de la calidad de los clusters**
    - Se calcula el **Silhouette Score** para evaluar la calidad de los clusters formados.
8. **Incorporación de los clusters al conjunto de datos**
    - Se añade la columna `cluster` al DataFrame original.
9. **Cálculo de medias por cluster**
    - Se calculan las medias de cada variable para cada cluster.
10. **Visualización**
    - Se reduce la dimensionalidad de los datos a 3 componentes principales usando PCA.
    - Se visualizan los clusters en un gráfico 3D para facilitar la interpretación.

### Ejecución

Para ejecutar el script, simplemente corre:

```bash
python clustering_sismico.py
```

Nota: Asegúrarse que el archivo `seismic-bumps.arff` esté disponible en la ruta correcta.

Este README describe paso a paso el flujo de trabajo del proyecto, facilitando su comprensión y reproducibilidad.

