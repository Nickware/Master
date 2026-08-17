# Shiny

Shiny es un framework web de R (también en Python desde 2022) que permite crear aplicaciones web interactivas directamente desde código R, sin conocimientos de HTML/CSS/JavaScript.

## Concepto básico

Shiny transforma análisis de datos estáticos en aplicaciones dinámicas donde usuarios pueden:
- Explorar datos con controles (sliders, dropdowns, botones)
- Ver gráficos que se actualizan en tiempo real
- Filtrar tablas masivas interactivamente
- Simular escenarios ajustando parámetros

## Estructura de una app Shiny

Cada aplicación tiene dos componentes principales (aunque esta aplicación contiene otra estructura diferente):

```r
# app.R - Lanzador
# Shiny detecta automáticamente global.R, ui.R y server.R si corres
# runApp("baloto_app/") sobre la carpeta completa. Este archivo es opcional
# y solo sirve como punto de entrada explícito si prefieres source() manual.

source("global.R")
source("ui.R")
source("server.R")

shinyApp(ui = ui, server = server)

```

## Características clave

Reactividad automática: Cuando cambias un input → R recalcula automáticamente los outputs dependientes.

Widgets incluidos:
- `sliderInput()`, `selectInput()`, `dateRangeInput()`
- `plotOutput()`, `tableOutput()`, `verbatimTextOutput()`
- Botones, checkboxes, radio buttons

Layouts responsivos:
- `fluidPage()`, `sidebarLayout()`
- `navbarPage()`, `tabsetPanel()`
- Integración Bootstrap nativa

## Tecnologías subyacentes

| Capa          | Tecnología     |
|---------------|----------------|
| Frontend      | HTML5 + CSS3 + JavaScript |
| Backend       | R (reactivity) + WebSocket |
| Widgets       | htmlwidgets + D3.js |
| Despliegue    | shinyapps.io, Posit Connect |

## Casos de uso reales

- Dashboards ejecutivos: KPIs interactivos con filtros por fecha/segmento
- Simuladores:** Monte Carlo, pronósticos financieros, optimización
- Exploradores de datos: Filtrado de tablas con millones de filas
- Mapas interactivos: `leaflet` + datos R en tiempo real
- Reportes dinámicos: Gráficos que responden a inputs del usuario

## Despliegue

| Plataforma          | Uso típico                    |
|---------------------|-------------------------------|
| `runApp()`          | Desarrollo local             |
| shinyapps.io        | Prototipos públicos (gratis) |
| Posit Connect       | Enterprise (autenticación)   |
| Shiny Server        | Autoservido (Linux)          |
| Docker              | Cloud/DevOps                 |

## Ventajas vs alternativas

| Aspecto           | Shiny (R)          | Streamlit (Python) | Dash (Python) |
|-------------------|--------------------|--------------------|---------------|
| Curva aprendizaje | Muy baja (solo R) | Muy baja (solo Py) | Media         |
| Reactividad       | Automática         | Automática         | Manual        |
| Widgets           | +100 incluidos     | 30+ incluidos      | Personalizables |
| Despliegue        | Excelente          | Bueno              | Bueno         |

## Ejemplo Shiny de este repositorio: Analizador de Aleatoriedad del Baloto

```r
# ui.R - Interfaz de usuario
# Analizador de aleatoriedad del Baloto (verificación estadística,)

library(shiny)

ui <- fluidPage(
  titlePanel("Analizador de Aleatoriedad del Baloto"),

  sidebarLayout(
    sidebarPanel(
      helpText(
        "Esta app evalúa si el historial de sorteos se comporta como un ",
        "Generador uniforme e independiente. No predice números futuros: ",
        "Los sorteos justos son i.i.d., así que ningún patrón histórico ",
        "Tiene poder predictivo real."
      ),
...
```

Shiny es un framework ideal para data scientists que quieren compartir análisis interactivos con stakeholders sin aprender desarrollo web tradicional.
