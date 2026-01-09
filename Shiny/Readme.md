# Shiny

Shiny es un **paquete** de R que permite crear aplicaciones web interactivas usando solo código R, sin necesidad de saber HTML, CSS o JavaScript.[5][8]

## Qué es Shiny

- Es un framework que conecta código R con una interfaz web (en el navegador) mediante una arquitectura cliente–servidor.[2][3]
- Permite convertir scripts de análisis, gráficos y modelos en apps “vivas” donde el usuario interactúa con controles y ve resultados en tiempo real.[3][7]

## Componentes básicos

- **UI (user interface)**: Define la parte visual de la app (layout, botones, sliders, menús, tablas y gráficos).[8][2]
- **Server**: Define la lógica en R que se ejecuta cuando cambian los inputs y actualiza los outputs (gráficos, tablas, textos).[6][2]
- Habitualmente se trabaja en un solo archivo `app.R` que contiene `ui` y `server` y se ejecuta con `shiny::runApp()`.[3]

## Programación reactiva

- Shiny se basa en **reactividad**: cuando cambia un input (por ejemplo, un slider o un select), se vuelven a ejecutar solo las partes necesarias del código.[6][3]
- Esta reactividad se maneja con objetos reactivos (`reactive()`, `reactiveValues()`) y funciones de renderizado como `renderPlot()`, `renderTable()`, `renderText()`, etc.[7][6]

## Qué se puede hacer con Shiny

- Dashboards de seguimiento (por ejemplo, KPIs, monitorización de modelos, paneles de control de negocio o ciencia de datos).[5][7]
- Aplicaciones analíticas complejas: mapas interactivos, análisis de redes, modelos de ML donde el usuario sube datos, ajusta parámetros y ve resultados.[8][3]
- Apps incrustadas en R Markdown / Quarto, en páginas web mediante iframes o desplegadas en servicios como shinyapps.io o en servidores propios.[3][5]

## Integración y ecosistema

- Se integra muy bien con tidyverse (`ggplot2`, `dplyr`, etc.), con paquetes de mapas (`leaflet`) y con librerías de gráficos interactivos como `plotly`.[8][3]
- Puede extenderse con temas CSS, componentes HTML personalizados y JavaScript para crear interfaces más avanzadas y profesionales.[7][5]

[1](https://diegokoz.github.io/intro_ds/clase_6/06_explicacion.nb.html)
[2](https://cdr-book.github.io/shiny.html)
[3](https://programminghistorian.org/es/lecciones/creacion-de-aplicacion-shiny)
[4](https://www.youtube.com/watch?v=DtGfvsM1NMU)
[5](https://datascience.recursos.uoc.edu/es/shiny/)
[6](https://bookdown.org/martinmontaneb/claseshiny/Clase.html)
[7](https://github.com/rstudio/shiny)
[8](https://bastianolea.rbind.io/blog/r_introduccion/tutorial_shiny_1/)
[9](https://www.youtube.com/watch?v=AGgN72_l4QE)
[10](https://www.youtube.com/watch?v=JgQGuWrWcF8)
