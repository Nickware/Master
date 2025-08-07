library(shiny)

ui <- fluidPage(
  titlePanel("Asistente de Entrenamiento y Nutrición"),

  sidebarLayout(
    sidebarPanel(
      fileInput("data", "Cargar archivo de entrenamiento (CSV)", accept = ".csv"),
      dateInput("fecha", "Selecciona fecha del entrenamiento"),
      actionButton("procesar", "Procesar datos")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Resumen", verbatimTextOutput("resumen")),
        tabPanel("Gráficos", plotOutput("grafico")),
        tabPanel("Plan Semanal", tableOutput("plan"))
      )
    )
  )
)
