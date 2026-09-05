# **Python: proyectos de análisis y modelado sísmico**

Esta carpeta reúne varias propuestas de análisis sobre el dataset de eventos sísmicos `seismic-bumps.arff`, orientadas a resolver problemas de clasificación, clustering y reducción de dimensionalidad. La estructura general del proyecto está pensada para estudiar distintos enfoques sobre el mismo problema: detectar señales de riesgo sísmico o comportamientos críticos a partir de variables geofísicas y operativas.

## **Lo qué incluye esta carpeta**

- [Balanced_Random_Forest/readme.md](Balanced_Random_Forest/readme.md): enfoque basado en árboles de decisión para datos desbalanceados.
- [Cluster/Readme.md](Cluster/Readme.md): análisis no supervisado con K-Means para detectar agrupaciones naturales.
- [LDA/Readme.md](LDA/Readme.md): enfoque supervisado con SVM y validación basada en sensibilidad y especificidad.
- [PCA/readme.md](PCA/readme.md): reducción de dimensionalidad con PCA y clasificación robusta con Random Forest.
- [data/seismic-bumps.arff](data/seismic-bumps.arff): conjunto de datos principal usado por todos los proyectos.

## **Contexto general del problema**

El dataset de sismos que se usa en esta carpeta corresponde a un problema de clasificación binaria con clases altamente desbalanceadas:

- clase 0: casos no peligrosos,
- clase 1: casos considerados peligrosos o críticos.

Este tipo de problema presenta desafíos comunes:

- la clase rara suele estar subrepresentada,
- los falsos negativos pueden ser más costosos que los falsos positivos,
- se requiere interpretar con cuidado qué variables son más relevantes,
- muchas variables están correlacionadas o presentan ruido.

## **Proyectos incluidos**

### 1. Balanced Random Forest

Proyecto orientado a modelado con clases desbalanceadas usando árboles de decisión con balanceo de muestreo.

- Documentación: [Balanced_Random_Forest/readme.md](Balanced_Random_Forest/readme.md)
- Objetivo principal: detectar la clase minoritaria con buena capacidad predictiva y mantener interpretabilidad.
- Mejora recomendada: comparar con técnicas modernas como XGBoost, LightGBM, CatBoost o modelos con calibración de umbral.

### 2. Cluster

Proyecto orientado a segmentación automática de patrones sísmicos mediante clustering no supervisado.

- Documentación: [Cluster/Readme.md](Cluster/Readme.md)
- Objetivo principal: descubrir grupos naturales en los datos y analizar qué tipo de perfiles sísmicos existen.
- Mejora recomendada: probar k con validación adicional, usar DBSCAN o Gaussian Mixture Models y comparar resultados con análisis por turnos o por periodos.

### 3. LDA

Proyecto orientado a clasificación supervisada usando SVM con validación cruzada y métricas de riesgo.

- Documentación: [LDA/Readme.md](LDA/Readme.md)
- Objetivo principal: clasificar correctamente casos peligrosos utilizando un modelo robusto frente a desbalance.
- Mejora recomendada: revisar el nombre del proyecto y renombrar la carpeta si se desea reflejar de forma fiel que el modelo es un SVM y no LDA; además, se puede añadir curva PR, análisis de umbral y validación externa.

### 4. PCA

Proyecto orientado a reducción de dimensionalidad plus clasificación para mejorar la capacidad predictiva de modelos con muchas variables correlacionadas.

- Documentación: [PCA/readme.md](PCA/readme.md)
- Objetivo principal: reducir dimensionalidad y mantener la mayor parte de la información útil antes de modelar.
- Mejora recomendada: comparar PCA con otras técnicas como kernel PCA, ICA, SelectKBest o modelos basados en regularización; además, documentar mejor el proceso de selección de componentes.

## **Comparación rápida de enfoques**

- Balanced Random Forest: muy útil cuando se prioriza la clase rara y quiere mantenerse la interpretabilidad.
- Cluster: útil para análisis exploratorio y agrupación de patrones no etiquetados.
- LDA/SVM: útil cuando se quiere clasificación supervisada con buen manejo del desbalance real y ajuste de hiperparámetros.
- PCA: útil cuando hay muchas variables correlacionadas y se desea simplificar el problema antes de modelar.

## **Relación entre los proyectos**

Los cuatro proyectos comparten:

- el mismo dataset base,
- la misma problemática de clases desbalanceadas,
- una lógica común de preprocesamiento y preparación de datos,
- diferentes formas de abordar la detección de eventos raros o grupos de comportamiento.

En otras palabras, se pueden entender como caminos complementarios del mismo problema:

- clustering para explorar,
- PCA para reducir dimensiones,
- SVM para clasificar,
- Random Forest balanceado para predicción con enfoque en la clase rara.

## **Perspectivas de mejora general**

### Mejoras técnicas

- centralizar la carga del dataset en una función reutilizable para todos los scripts,
- crear un módulo común de preprocessing y evaluación,
- comparar varios modelos con una tabla resumen automática,
- guardar resultados y métricas en CSV o JSON,
- usar pipelines de scikit-learn para automatizar validación y transformaciones.

### Mejoras de reproducibilidad

- documentar versiones exactas de librerías en un `requirements.txt`,
- añadir un entorno virtual (`venv` o `conda`),
- incluir instrucciones para ejecutar cada proyecto desde la raíz del repositorio,
- normalizar nombres de carpetas y archivos para mayor consistencia.

### Mejoras de análisis

- evaluar múltiples umbrales de decisión y no solo el valor por defecto de 0.5,
- comparar modelos con validación cruzada estratificada,
- estudiar qué variables son más importantes por clase,
- añadir análisis de series temporales o segmentación por turno/periodo,
- incluir métricas como F1, F2, PR-AUC, recall, precision, balanced accuracy.

## **Cómo ejecutar los proyectos**

Desde la raíz del repositorio:

```bash
cd /ruta/al/repositorio/Master
python Python/Balanced_Random_Forest/basic/seimic_bumps.py
python Python/Cluster/kmeans.py
python Python/LDA/LDA.py
python Python/PCA/PCA.py
```

## **Nota final**

Esta carpeta no solo muestra distintos enfoques de aprendizaje automático, sino que también ilustra cómo el mismo problema puede abordarse desde perspectivas distintas: clasificación, segmentación, reducción de dimensionalidad y modelado con desbalance. El conjunto resulta útil tanto para aprendizaje práctico como para comparación de metodologías sobre datos sísmicos reales.
