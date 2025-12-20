# Análisis de clasificación supervisada usando Support Vector Machines (SVM) 

Este script es una implementación de un análisis de clasificación supervisada usando Support Vector Machines (SVM) sobre un conjunto de datos de “seismic bumps” almacenado en formato ARFF, típicamente usado en minería de datos o geofísica para predecir eventos sísmicos peligrosos en minas de carbón u otros contextos.

### Propósito y flujo general

El código automatiza el pre-procesamiento de datos, el ajuste de hiperparámetros, el entrenamiento y la validación de un modelo SVM, empleando validación cruzada y métricas tanto clásicas como orientadas a modelos desbalanceados (sensibilidad y especificidad).

### Detalle paso a paso

#### 1. Carga de librerías y datos
- Utiliza `scipy.io.arff` para cargar archivos .arff, `pandas` para manipulación de datos, y varios módulos de `scikit-learn` para codificación, escalado, partición, entrenamiento y evaluación.
- Define una función `load_data` que carga los datos, asigna nombres de columnas legibles y retorna un DataFrame.

#### 2. Preprocesamiento
- Convierte variables categóricas a valores numéricos con `LabelEncoder` para preparar datos para SVM.
- Todas las variables numéricas (incluidas las recién codificadas) se escalan al rango [-1, 1] usando `MinMaxScaler`, previniendo que ninguna domine el entrenamiento.

#### 3. División de datos
- Separa el conjunto en train/test manteniendo la proporción de clases (`stratify`) mediante `train_test_split`.

#### 4. Entrenamiento y validación
- La función `train_model` realiza una búsqueda en rejilla (`GridSearchCV`) sobre hiperparámetros $ C $ y $ \gamma $ de una SVM con kernel RBF, usando validación cruzada de 10 pliegues. Hay dos optimizaciones: una por “recall” (sensibilidad), otra por “roc_auc”, ambas relevantes para datos desbalanceados.
- Evalúa el modelo usando precisión (`accuracy_score`), sensibilidad y especificidad, extraídas de la matriz de confusión (`confusion_matrix`).

#### 5. Métricas y salida
- Se muestran en pantalla los resultados de precisión, sensibilidad y especificidad para ambos enfoques de optimización.

### Métricas usadas

- **Precisión**: Índice general de acierto del modelo.
- **Sensibilidad (Recall)**: Capacidad para identificar correctamente los positivos verdaderos (importante si la clase positiva es rara y costosa de omitir).
- **Especificidad**: Capacidad para identificar correctamente los negativos (útil en contextos donde los falsos positivos tienen un costo).

### Consideraciones técnicas
- El modelo ajusta automáticamente los pesos de clase con `class_weight='balanced'`, tratando el desbalance típico de estos conjuntos.
- Las búsquedas de hiperparámetros pueden ser costosas computacionalmente, por lo que el tiempo de ejecución depende de los recursos del equipo.

### Aplicaciones
Este tipo de pipeline puede adaptarse a típicos problemas de clasificación binaria en contextos industriales, médicos o financieros donde hay clases desbalanceadas y métricas de tipo “recall” y “specifity” importan más que la precisión simple.
