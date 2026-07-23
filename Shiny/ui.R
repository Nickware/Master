# ui.R - Interfaz de usuario
# Analizador de aleatoriedad del Baloto (verificación estadística, NO predicción)

library(shiny)

ui <- fluidPage(
  titlePanel("Analizador de Aleatoriedad del Baloto"),

  sidebarLayout(
    sidebarPanel(
      helpText(
        "Esta app evalúa si el historial de sorteos se comporta como un ",
        "generador uniforme e independiente. No predice números futuros: ",
        "los sorteos justos son i.i.d., así que ningún patrón histórico ",
        "tiene poder predictivo real."
      ),

      radioButtons(
        "fuente_datos", "Fuente de datos:",
        choices = c(
          "Usar datos de ejemplo (simulados)" = "ejemplo",
          "Cargar historial real (CSV)" = "csv"
        ),
        selected = "ejemplo"
      ),

      conditionalPanel(
        condition = "input.fuente_datos == 'csv'",
        fileInput(
          "archivo_csv", "Archivo CSV de sorteos históricos:",
          accept = c(".csv")
        ),
        helpText(
          "El CSV debe tener columnas con las bolas principales ",
          "(ej. bola1, bola2, ... o num1, num2, ...) y opcionalmente ",
          "una columna 'balota' o 'superbalota'."
        )
      ),

      conditionalPanel(
        condition = "input.fuente_datos == 'ejemplo'",
        numericInput("n_sorteos_ejemplo", "N° de sorteos de ejemplo:", 500, 50, 5000),
        numericInput("rango_max_ejemplo", "Rango máximo de números:", 43, 10, 99)
      ),

      actionButton("run", "Ejecutar análisis", class = "btn-primary"),

      hr(),
      helpText(
        "Chi-cuadrado: contrasta si cada número aparece con la frecuencia ",
        "esperada bajo un sorteo uniforme."
      ),
      helpText(
        "Prueba de rachas: contrasta si hay dependencia secuencial entre ",
        "sorteos consecutivos."
      )
    ),

    mainPanel(
      tabsetPanel(
        tabPanel(
          "Frecuencia de números",
          plotOutput("plot_frecuencias"),
          verbatimTextOutput("resultado_chi")
        ),
        tabPanel(
          "Prueba de rachas",
          plotOutput("plot_rachas"),
          verbatimTextOutput("resultado_rachas")
        ),
        tabPanel(
          "Datos utilizados",
          tableOutput("tabla_datos")
        )
      )
    )
  )
)
