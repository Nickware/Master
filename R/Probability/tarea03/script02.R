## 0. Paquetes --------------------------------------------------------------

# scales solo se usa para formatear porcentajes
library(scales)   # percent(), label_percent() [web:25][web:28]


## 1. Lectura e inspección de datos ----------------------------------------

datos_paramo <- read.csv("Paramo.csv",
                         header = TRUE,
                         sep    = ";")

# Mejor que View() para un script reproducible
str(datos_paramo)   # estructura y tipos de variables [web:23]


## 2. Separar variables cuantitativas y cualitativas ------------------------

# Ajusta estos nombres de columna a los reales en tu archivo
# (aquí supongo nombres típicos, cámbialos según corresponda)
cuantitativas <- datos_paramo[, c(2, 3, 4, 5, 6, 8)]
cualitativas  <- datos_paramo[, c(1, 7)]


## 3. Estado más frecuente (col. cualitativa) ------------------------------

# Estado más frecuente de la variable categórica (por ejemplo, "Estado")
summary(cualitativas[, 2])   # tabla de frecuencias y NA


## 4. Talla más frecuente ---------------------------------------------------

# Si la talla es numérica continua, el "más frecuente" no es tan informativo.
# Si está categorizada, summary() da la moda empírica.
summary(cuantitativas[, 5])


## 5. Estado de desarrollo vs plantas afectadas -----------------------------

# Suponiendo que la severidad está en la columna 8 (como en tu script)
tabla_severidad <- table(datos_paramo[, 8])

barplot(
  tabla_severidad,
  las   = 1,
  main  = "Estado de desarrollo vs plantas afectadas",
  xlab  = "Severidad",
  ylab  = "Número de plantas",
  col   = "steelblue"
)   # [web:23][web:26]


## 6. Probabilidad empírica en un nuevo transecto ---------------------------

# Tabla cruzada: por ejemplo, severidad (col 6) vs estado (col 7)
tabla_cruzada <- table(datos_paramo[, 6], datos_paramo[, 7])

# Barplot agrupado (barras lado a lado) [web:16][web:19][web:23]
barplot(
  tabla_cruzada,
  beside = TRUE,
  las    = 1,
  col    = c("grey80", "grey40"),
  xlab   = "Categoría",
  ylab   = "Número de plantas",
  main   = "Severidad por estado"
)
legend("topright",
       legend = rownames(tabla_cruzada),
       fill   = c("grey80", "grey40"),
       bty    = "n")

# Probabilidad empírica (global) de cada combinación
prob_empirica <- prop.table(tabla_cruzada)  # frecuencias relativas [web:22]
prob_empirica

# Si quieres formatear como porcentaje:
percent(prob_empirica)   # devuelve un vector de caracteres con % [web:22][web:25]


## 7. Incidencia global de la afectación -----------------------------------

# Histograma SIN dibujar, para obtener los conteos [web:21][web:24][web:27]
hist_severidad <- hist(datos_paramo[, 8],
                       breaks = 6,
                       plot   = FALSE)

# Conteos absolutos por clase
frecuencia_absoluta <- hist_severidad$counts    # [web:21][web:24][web:27]

# Frecuencia relativa
frecuencia_relativa <- frecuencia_absoluta / sum(frecuencia_absoluta)

# Porcentajes como texto
percent(frecuencia_relativa)   # [web:25]

# Si quieres graficar la incidencia global en porcentaje:
barplot(
  frecuencia_relativa,
  names.arg = round(hist_severidad$mids, 2),  # puntos medios de clases [web:21][web:24][web:27]
  las       = 1,
  ylim      = c(0, max(frecuencia_relativa) * 1.2),
  main      = "Incidencia global de la afectación",
  xlab      = "Severidad (clases)",
  ylab      = "Frecuencia relativa"
)
text(
  x      = seq_along(frecuencia_relativa),
  y      = frecuencia_relativa,
  labels = percent(frecuencia_relativa),
  pos    = 3,
  cex    = 0.8
)


## 8. Correlación entre altura y número de frailejones ----------------------

# Aquí depende de cómo estén tus variables:
# por ejemplo, supongamos:
#   - altura_promedio_transecto  en columna 3
#   - numero_frailejones         en columna 5
altura_prom  <- cuantitativas[, 2]  # AJUSTAR AL NOMBRE/COLUMNA REAL
n_frailejones <- cuantitativas[, 4] # AJUSTAR AL NOMBRE/COLUMNA REAL

# Gráfico de dispersión
plot(
  altura_prom,
  n_frailejones,
  xlab = "Altura promedio (m)",
  ylab = "Número de frailejones",
  main = "Altura promedio vs número de frailejones",
  pch  = 16,
  col  = "darkgreen"
)

# Correlación de Pearson (si ambas son numéricas continuas)
cor(altura_prom, n_frailejones, use = "complete.obs")

.
