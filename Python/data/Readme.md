# Características del conjunto de datos “seismic bumps”

El texto describe el contexto y las características del conjunto de datos “seismic bumps”, usado para la predicción automática de peligros sísmicos (evento de alta energía) en minas subterráneas de carbón en Polonia.

### Origen y composición del dataset

- Incluye 2584 instancias, cada una representando un turno de 8 horas en dos longwalls (frentes largos de minado).
- Cada instancia almacena 18 atributos descriptivos y una columna de clase (resultado): “hazardous state” (evento peligroso en el siguiente turno) y “non-hazardous state” (ningún evento de alta energía).
- La clase positiva (evento peligroso) es muy minoritaria: solo 170 casos (6.6%), versus 2414 negativos (93.4%), lo que crea un problema típico de desbalance de clases en machine learning aplicado a riesgos industriales.

### Detalle de los atributos

Muchos atributos representan valores registrados por geófonos y sistemas de monitoreo sísmico y seismoacústico:
- Evaluaciones de peligro usando distintos métodos y sensores (seismic, seismoacoustic, ghazard).
- Resúmenes de energía y cantidad de pulsos sísmicos del turno actual, así como desviaciones respecto a promedios anteriores.
- Detección y conteo de “seismic bumps” en rangos de energía progresivamente mayores, desde $$10^2$$ hasta $$10^{10}$$ Julios.
- Información sobre los turnos: si fueron de extracción de carbón o de preparativos.
- Atributos agregados de energía total y máxima registrada.

### Relevancia práctica y científica

El conjunto se creó para investigar y mejorar la predicción de riesgos sísmicos, usando técnicas de inducción de reglas, clustering, y redes neuronales.
- La predicción de eventos sísmicos peligrosos es extremadamente compleja, dada la rareza de los casos graves y la abundancia de eventos de baja energía.
- Los métodos estadísticos tradicionales han sido insuficientes, y se exploran técnicas de aprendizaje automático para abordar la alta desproporción y la dificultad para generar sensibilidad y especificidad aceptables.
- Predecir incrementos de actividad sísmica (incluso si no se identifica el “rockburst” exacto) sirve para tomar medidas preventivas, como reforzar la seguridad o evacuar áreas de trabajo.

### Aplicaciones típicas y problemas

- El principal reto es el desbalance: los modelos tienden a predecir la clase mayoritaria (“no peligro”) y pueden fallar en detectar los casos realmente críticos.
- Las métricas de evaluación más relevantes son la sensibilidad (para detectar positivos aunque sean raros) y la especificidad (para evitar falsas alarmas excesivas).
- El dataset ha sido usado como benchmark para comparar métodos de clasificación robustos ante desbalance, tanto estadísticos como de machine learning avanzado.

***

La clave del análisis con este dataset es diseñar y validar modelos capaces de detectar los pocos casos peligrosos sin sacrificar la precisión general, empleando técnicas robustas y métricas orientadas a riesgo. Este contexto justifica los pasos y decisiones presentes en el script que analizaste previamente.
