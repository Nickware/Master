# server.R - Lógica reactiva
# Analizador de aleatoriedad del Baloto (verificación estadística, NO predicción)

library(shiny)
library(ggplot2)

server <- function(input, output, session) {

  # -------------------------------------------------------------------------
  # Carga de datos: ejemplo simulado o CSV real del usuario
  # -------------------------------------------------------------------------
  datos <- eventReactive(input$run, {
    if (input$fuente_datos == "ejemplo") {
      generar_datos_ejemplo(
        n_sorteos = input$n_sorteos_ejemplo,
        rango_max = input$rango_max_ejemplo
      )
    } else {
      req(input$archivo_csv)
      read.csv(input$archivo_csv$datapath, stringsAsFactors = FALSE)
    }
  })

  rango_max <- eventReactive(input$run, {
    if (input$fuente_datos == "ejemplo") {
      input$rango_max_ejemplo
    } else {
      # Si es CSV real, se asume el máximo observado + margen razonable,
      # o se puede ajustar manualmente si el usuario conoce el rango oficial.
      cols <- detectar_columnas_bolas(datos())
      max(sapply(datos()[cols], max, na.rm = TRUE))
    }
  })

  columnas_bolas <- eventReactive(input$run, {
    if (input$fuente_datos == "ejemplo") {
      grep("^bola\\d+$", names(datos()), value = TRUE)
    } else {
      cols <- detectar_columnas_bolas(datos())
      validate(need(
        length(cols) > 0,
        "No se detectaron columnas de bolas en el CSV. Verifica los nombres de columna (ej. bola1, bola2, ...)."
      ))
      cols
    }
  })

  # -------------------------------------------------------------------------
  # Prueba chi-cuadrado sobre frecuencias de números
  # -------------------------------------------------------------------------
  resultado_chi <- eventReactive(input$run, {
    df <- datos()
    cols <- columnas_bolas()
    numeros_vector <- unlist(df[cols])
    prueba_chi_cuadrado(numeros_vector, rango_max())
  })

  output$plot_frecuencias <- renderPlot({
    res <- resultado_chi()
    ggplot(res$tabla, aes(x = numero)) +
      geom_col(aes(y = observado), fill = "steelblue", alpha = 0.8) +
      geom_hline(aes(yintercept = res$tabla$esperado[1]), linetype = "dashed", color = "firebrick") +
      labs(
        title = "Frecuencia observada por número",
        subtitle = "Línea roja = frecuencia esperada bajo distribución uniforme",
        x = "Número", y = "Frecuencia observada"
      ) +
      theme_minimal()
  })

  output$resultado_chi <- renderPrint({
    res <- resultado_chi()
    cat("Prueba de bondad de ajuste (chi-cuadrado)\n")
    cat("H0: cada número aparece con probabilidad uniforme\n\n")
    cat(sprintf("Estadístico chi-cuadrado: %.4f\n", res$estadistico))
    cat(sprintf("Grados de libertad: %d\n", res$df))
    cat(sprintf("Valor p: %.4f\n\n", res$p_valor))
    if (res$p_valor < 0.05) {
      cat("Conclusión: se rechaza H0 al 5%. Hay evidencia de desviación de la uniformidad.\n")
    } else {
      cat("Conclusión: no se rechaza H0 al 5%. Los datos son consistentes con un sorteo uniforme.\n")
    }
  })

  # -------------------------------------------------------------------------
  # Prueba de rachas sobre secuencia alto/bajo sorteo a sorteo
  # -------------------------------------------------------------------------
  resultado_rachas <- eventReactive(input$run, {
    df <- datos()
    cols <- columnas_bolas()
    suma_por_sorteo <- rowSums(df[cols])
    mediana <- median(suma_por_sorteo)
    secuencia_binaria <- as.integer(suma_por_sorteo > mediana)
    prueba_rachas(secuencia_binaria)
  })

  output$plot_rachas <- renderPlot({
    df <- datos()
    cols <- columnas_bolas()
    suma_por_sorteo <- rowSums(df[cols])
    mediana <- median(suma_por_sorteo)

    df_plot <- data.frame(
      indice = seq_along(suma_por_sorteo),
      suma = suma_por_sorteo,
      grupo = ifelse(suma_por_sorteo > mediana, "Alto", "Bajo")
    )

    ggplot(df_plot, aes(x = indice, y = suma, color = grupo)) +
      geom_point() +
      geom_hline(yintercept = mediana, linetype = "dashed", color = "gray40") +
      labs(
        title = "Secuencia de sumas por sorteo (Alto/Bajo respecto a la mediana)",
        x = "Sorteo (orden cronológico)", y = "Suma de bolas principales",
        color = "Grupo"
      ) +
      theme_minimal()
  })

  output$resultado_rachas <- renderPrint({
    res <- resultado_rachas()
    cat("Prueba de rachas (Wald-Wolfowitz)\n")
    cat("H0: no hay dependencia secuencial entre sorteos consecutivos\n\n")
    cat(sprintf("N° de sorteos: %d\n", res$n))
    cat(sprintf("Rachas observadas: %d\n", res$rachas_observadas))
    cat(sprintf("Rachas esperadas bajo H0: %.2f\n", res$rachas_esperadas))
    cat(sprintf("Estadístico Z: %.4f\n", res$estadistico_z))
    cat(sprintf("Valor p: %.4f\n\n", res$p_valor))
    if (res$p_valor < 0.05) {
      cat("Conclusión: se rechaza H0 al 5%. Hay evidencia de dependencia secuencial.\n")
    } else {
      cat("Conclusión: no se rechaza H0 al 5%. No hay evidencia de dependencia secuencial.\n")
    }
  })

  # -------------------------------------------------------------------------
  # Vista de datos crudos utilizados en el análisis
  # -------------------------------------------------------------------------
  output$tabla_datos <- renderTable({
    head(datos(), 15)
  })
}
