# Shiny

Shiny es un **framework web de R** (también Python desde 2022) que permite crear **aplicaciones web interactivas** directamente desde código R, sin conocimientos de HTML/CSS/JavaScript. [programminghistorian](https://programminghistorian.org/es/lecciones/creacion-de-aplicacion-shiny)

## Concepto básico

Shiny transforma análisis de datos estáticos en **aplicaciones dinámicas** donde usuarios pueden:
- Explorar datos con controles (sliders, dropdowns, botones)
- Ver gráficos que se actualizan en tiempo real
- Filtrar tablas masivas interactivamente
- Simular escenarios ajustando parámetros

## Estructura de una app Shiny

Cada aplicación tiene **dos componentes principales**:

```r
# ui.R - Interfaz de usuario
library(shiny)
ui <- fluidPage(
  titlePanel("Mi primera app Shiny"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("obs", "N° observaciones:", 1, 1000, 500)
    ),
    mainPanel(
      plotOutput("distPlot")
    )
  )
)

# server.R - Lógica reactiva
server <- function(input, output) {
  output$distPlot <- renderPlot({
    hist(rnorm(input$obs))
  })
}

shinyApp(ui = ui, server = server)
```

## Características clave

**Reactividad automática:** Cuando cambias un input → R recalcula automáticamente los outputs dependientes.

**Widgets incluidos:** 
- `sliderInput()`, `selectInput()`, `dateRangeInput()`
- `plotOutput()`, `tableOutput()`, `verbatimTextOutput()`
- Botones, checkboxes, radio buttons

**Layouts responsivos:**
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

- **Dashboards ejecutivos:** KPIs interactivos con filtros por fecha/segmento
- **Simuladores:** Monte Carlo, pronósticos financieros, optimización
- **Exploradores de datos:** Filtrado de tablas con millones de filas
- **Mapas interactivos:** `leaflet` + datos R en tiempo real
- **Reportes dinámicos:** Gráficos que responden a inputs del usuario

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

## Ejemplo completo: Dashboard Baloto

```r
library(shiny)
library(ggplot2)

ui <- fluidPage(
  titlePanel("Analizador Baloto"),
  sidebarLayout(
    sidebarPanel(
      numericInput("simulaciones", "Simulaciones Monte Carlo:", 10000, 1000, 100000),
      actionButton("run", "¡Simular!")
    ),
    mainPanel(
      plotOutput("histograma"),
      verbatimTextOutput("esperanza")
    )
  )
)

server <- function(input, output) {
  valores <- eventReactive(input$run, {
    simulacion_baloto(input$simulaciones)
  })
  
  output$histograma <- renderPlot({
    ggplot(valores(), aes(x = ganancia)) + 
      geom_histogram() + 
      labs(title = "Distribución ganancias Baloto")
  })
}
```

**Shiny es perfecto** para data scientists que quieren compartir análisis interactivos con stakeholders sin aprender desarrollo web tradicional. Un dashboard funcional se crea en **30 minutos** desde RStudio. [cdr-book.github](https://cdr-book.github.io/shiny.html)
