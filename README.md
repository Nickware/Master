## Temáticas

Gran parte del contenido consiste en notebooks, destinados al desarrollo de experimentos computacionales o investigaciones en ciencias computacionales y análisis de datos.

- **Deportes**
- **Fluidos**
- **Geología**
- **Probabilidad**


## Índice de proyectos

### Fluidos y simulación científica

- [Simulación CFD con OpenFOAM](C++/Readme.md): genera un flujo parabólico y permite estudiar la calidad de la malla y la convergencia del solver.
- [Análisis y visualización con Octave](Octave/Readme.md): procesa datos de velocidad exportados desde una simulación CFD y los compara con un perfil teórico.

Estos dos proyectos forman una cadena de trabajo: OpenFOAM produce los resultados numéricos y Octave los analiza y visualiza.

**Requisitos principales:** Linux o WSL, OpenFOAM 10 o superior, compilador GNU, Octave o MATLAB y datos de velocidad exportados a CSV.

### Geología y aprendizaje automático

- [Descripción del dataset sísmico](Python/data/Readme.md): documenta el conjunto `seismic-bumps`, sus variables y el problema de desbalance de clases.
- [Clustering sísmico con K-Means](Python/Cluster/Readme.md): agrupa los eventos y visualiza los clusters mediante PCA.
- [Clasificación con SVM](Python/LDA/Readme.md): entrena y evalúa modelos supervisados con métricas adecuadas para clases desbalanceadas.
- [Balanced Random Forest](Python/Balanced_Random_Forest/readme.md): aplica bosques aleatorios balanceados para detectar eventos sísmicos minoritarios.
- [PCA y Random Forest balanceado](Python/PCA/readme.md): combina reducción de dimensionalidad, SMOTE y clasificación con Balanced Random Forest.

El dataset documentado en `Python/data` es la base común de los análisis de `Cluster`, `LDA`, `Balanced_Random_Forest` y `PCA`. K-Means explora grupos; SVM, Random Forest y PCA + Random Forest construyen modelos predictivos.

**Requisitos principales:** Python 3.7 o superior, pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, matplotlib y seaborn, según el proyecto.

### Estadística y aplicaciones interactivas en R

- [Introducción al lenguaje R](R/Readme.md): resume las capacidades de R para estadística, visualización y machine learning.
- [Aplicación web con Shiny](Shiny/Readme.md): muestra cómo convertir análisis estadísticos en una aplicación interactiva.

Shiny utiliza R como base y extiende los análisis del repositorio hacia una interfaz web interactiva.

**Requisitos principales:** R, paquetes estadísticos necesarios para cada script y, para la aplicación, Shiny y un entorno como RStudio/Posit.


## Relaciones entre proyectos

```text
OpenFOAM / C++
	|
	v
Datos de velocidad CFD ----> Octave / MATLAB

seismic-bumps
	|
	+----> K-Means
	+----> SVM
	+----> Balanced Random Forest
	+----> PCA + Balanced Random Forest

R ----> Shiny
```

Los proyectos son independientes y pueden ejecutarse por separado. Las relaciones anteriores indican reutilización de datos, resultados o conceptos, no una dependencia obligatoria de compilación.


## Acerca de este repositorio

Repositorio experimental y académico, en el que se desarrolla y almacenan proyectos relacionados con diversos lenguajes y entornos de programación, principalmente en Jupyter Notebook, C++, R y Python. Su utilidad está orientada a fines educativos o de investigación
