# Lenguaje R

R es un **lenguaje de programación y entorno de software libre** especializado en **análisis estadístico, visualización de datos y computación científica**. [mexico.unir](https://mexico.unir.net/noticias/ingenieria/lenguaje-r-big-data/)

## Origen y características principales

- **Creado en 1993** por Ross Ihaka y Robert Gentleman en la Universidad de Auckland (Nueva Zelanda) como implementación libre del lenguaje S.
- **Interpretado y orientado a objetos**: ejecuta comandos directamente sin compilación previa; todo (datos, funciones, gráficos) se maneja como objetos.
- **Licencia GNU GPL**: completamente gratuito y de código abierto, con más de **20.000 paquetes** disponibles en CRAN (Comprehensive R Archive Network).

## Capacidades principales

**Análisis estadístico avanzado:**
- Modelos lineales, logísticos, mixtos, bayesianos
- Análisis multivariante (PCA, clustering, factor analysis)
- Series temporales y pronósticos (ARIMA, ETS)
- Machine Learning (randomForest, xgboost, caret)

**Visualización de datos:**
```r
# Ejemplo ggplot2 - estándar de facto
library(ggplot2)
ggplot(mtcars, aes(x = wt, y = mpg)) + 
  geom_point() + 
  geom_smooth(method = "lm")
```

**Manejo de datos:**
- `dplyr`, `tidyr` para transformación (tidyverse)
- `data.table` para datos masivos
- `readr`, `haven` para importar CSV, Excel, SPSS, SAS

## Áreas de aplicación

| Disciplina          | Paquetes clave              |
|---------------------|-----------------------------|
| Bioestadística      | survival, Bioconductor     |
| Finanzas            | quantmod, PerformanceAnalytics |
| Economía            | plm, AER, wooldridge       |
| Machine Learning    | caret, mlr3, h2o           |
| Genómica            | Bioconductor (~2.000 paquetes) |
| Visualización       | ggplot2, plotly, leaflet   |

## Ventajas competitivas

- **Ecosistema inigualable** para estadística: funciones específicas para cada prueba/método (t-test, ANOVA, regresiones GLS, etc.).
- **Reproducibilidad**: R Markdown/Quarto para informes automatizados.
- **Visualización líder**: ggplot2 es estándar mundial en papers científicos.
- **Comunidad académica**: miles de papers estadísticos incluyen código R.

## Desventajas

- **Curva de aprendizaje pronunciada** para programadores generales.
- **Rendimiento inferior** a Python/C++ para datos muy grandes (aunque `data.table` mitiga esto).
- **Gestión de memoria** menos eficiente que lenguajes compilados.

## Comparativa R vs Python

| Aspecto           | R                      | Python                |
|-------------------|------------------------|-----------------------|
| Estadística       | Excelente             | Bueno                |
| Machine Learning  | Muy bueno             | Excelente            |
| Visualización     | Líder (ggplot2)       | Muy bueno (matplotlib)|
| Desarrollo web    | Bueno (Shiny)         | Excelente (Django/Flask) |
| Ecosistema datos  | Enorme (CRAN)         | Enorme (PyPI)        |

## Entorno de desarrollo recomendado

**RStudio/Posit** es prácticamente obligatorio:
- Editor con autocompletado inteligente
- Visualización de objetos y paquetes
- Integración Git, Shiny, R Markdown
- Depurador visual y profiling

## Ejemplo básico completo

```r
# Análisis rápido Baloto (contexto de nuestra conversación)
baloto <- data.frame(
  nums = sample(1:43, 5, replace = FALSE),
  super = sample(1:16, 1)
)

# Probabilidad premio mayor
combinaciones <- choose(43,5) * 16  # 15.401.568
cat("Probabilidad:", 1/combinaciones, "\n")
```

**R es el lenguaje de facto mundial para estadística aplicada**, machine learning reproducible y visualización científica. Su fortaleza radica en la combinación única de potencia estadística + visualización + reproducibilidad en un solo ecosistema. [openwebinars](https://openwebinars.net/blog/introduccion-lenguaje-r/)
