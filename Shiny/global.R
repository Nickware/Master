# global.R - Funciones estadísticas y utilidades de datos
# Analizador de aleatoriedad del Baloto (NO predictor)
#
# Este módulo contiene:
#   - Generación de datos sintéticos de ejemplo (para probar la app sin CSV real)
#   - Prueba de bondad de ajuste chi-cuadrado sobre frecuencias de números
#   - Prueba de rachas (Wald-Wolfowitz) sobre una secuencia binaria derivada de los sorteos
#   - Utilidades para detectar columnas de bolas en el CSV cargado por el usuario

library(shiny)
library(ggplot2)

# ---------------------------------------------------------------------------
# 1. Datos sintéticos de ejemplo (solo para demostración, no son sorteos reales)
# ---------------------------------------------------------------------------
generar_datos_ejemplo <- function(n_sorteos = 500, n_bolas = 5, rango_max = 43) {
  set.seed(NULL)
  bolas <- t(replicate(n_sorteos, sort(sample(1:rango_max, n_bolas))))
  colnames(bolas) <- paste0("bola", 1:n_bolas)
  balota <- sample(1:16, n_sorteos, replace = TRUE)
  df <- data.frame(
    fecha = seq(as.Date("2015-01-01"), by = "week", length.out = n_sorteos),
    bolas,
    balota = balota
  )
  df
}

# ---------------------------------------------------------------------------
# 2. Detección flexible de columnas de bolas principales en un CSV cargado
#    Acepta nombres como bola1, bola_1, num1, n1, b1, etc.
# ---------------------------------------------------------------------------
detectar_columnas_bolas <- function(df) {
  patron <- "^(bola|num|n|b)[_\\.]?\\d+$"
  nombres <- names(df)
  cols <- nombres[grepl(patron, tolower(nombres))]
  cols
}

detectar_columna_balota <- function(df) {
  nombres <- tolower(names(df))
  candidatos <- c("balota", "superbalota", "super_balota", "bono")
  idx <- which(nombres %in% candidatos)
  if (length(idx) == 0) return(NULL)
  names(df)[idx[1]]
}

# ---------------------------------------------------------------------------
# 3. Prueba de bondad de ajuste chi-cuadrado
#    H0: cada número entre 1 y rango_max aparece con probabilidad uniforme
#    Se agrupan TODAS las bolas principales de TODOS los sorteos en un solo
#    vector de frecuencias observadas, y se compara contra la frecuencia
#    esperada bajo un muestreo uniforme sin reemplazo dentro de cada sorteo.
# ---------------------------------------------------------------------------
prueba_chi_cuadrado <- function(numeros_vector, rango_max) {
  observado <- table(factor(numeros_vector, levels = 1:rango_max))
  n_total <- length(numeros_vector)
  esperado_prop <- rep(1 / rango_max, rango_max)

  resultado <- suppressWarnings(
    chisq.test(x = as.vector(observado), p = esperado_prop)
  )

  list(
    tabla = data.frame(
      numero = 1:rango_max,
      observado = as.vector(observado),
      esperado = n_total * esperado_prop
    ),
    estadistico = unname(resultado$statistic),
    df = unname(resultado$parameter),
    p_valor = resultado$p.value
  )
}

# ---------------------------------------------------------------------------
# 4. Prueba de rachas (Wald-Wolfowitz) sobre una secuencia binaria
#    Se construye una secuencia binaria "alto/bajo" comparando la suma de las
#    bolas de cada sorteo contra la mediana histórica, y se cuenta el número
#    de rachas (secuencias consecutivas del mismo signo) para contrastar contra
#    el número de rachas esperado bajo independencia.
# ---------------------------------------------------------------------------
prueba_rachas <- function(secuencia_binaria) {
  # secuencia_binaria: vector lógico o 0/1
  x <- as.integer(secuencia_binaria)
  n <- length(x)
  n1 <- sum(x == 1)
  n0 <- sum(x == 0)

  # Conteo de rachas observadas
  cambios <- sum(diff(x) != 0)
  rachas_obs <- cambios + 1

  # Media y varianza esperadas del número de rachas bajo H0 (independencia)
  media_rachas <- (2 * n1 * n0) / n + 1
  var_rachas <- (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n^2 * (n - 1))

  z <- (rachas_obs - media_rachas) / sqrt(var_rachas)
  p_valor <- 2 * (1 - pnorm(abs(z)))

  list(
    n = n,
    n1 = n1,
    n0 = n0,
    rachas_observadas = rachas_obs,
    rachas_esperadas = media_rachas,
    estadistico_z = z,
    p_valor = p_valor
  )
}
