server <- function(input, output) {
  datos <- reactive({
    req(input$data)
    read.csv(input$data$datapath)
  })

  output$resumen <- renderPrint({
    req(datos())
    summary(datos())
  })

  output$grafico <- renderPlot({
    req(datos())
    plot(datos()[[1]], datos()[[2]], type = "l", main = "Ejemplo de gráfico")
  })

  output$plan <- renderTable({
    data.frame(
      Día = c("Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"),
      Actividad = c("Descanso", "Tenis", "32K Proyecto", "Descanso", "Entrenamiento", "Ciclismo", "Fondo")
    )
  })
}
