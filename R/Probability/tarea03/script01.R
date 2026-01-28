## 1. Cargar datos ----------------------------------------------------------

# Leer el archivo CSV (ajusta la ruta si es necesario)
datosFragata <- read.csv("DatosEjercicios.csv",
                         header = TRUE,
                         sep = ",")   # en R actual stringsAsFactors = FALSE por defecto [web:7]

# Echar un vistazo rápido a la estructura (mejor que View para un script)
str(datosFragata)    # tipos de variables, nombres de columnas [web:7]


## 2. Seleccionar variables cuantitativas -----------------------------------

# Ideal: usar nombres de columnas en vez de posiciones
# Ajusta estos nombres a los que tenga tu archivo
# Ejemplo: supongamos que la variable de respuesta es "Gusanos"
# y el predictor que quieres usar es "pH"
cuantitativas <- datosFragata[, c("pH", "Gusanos")]

# Comprobar si hay NA
colSums(is.na(cuantitativas))


## 3. Gráfico de dispersión (Y vs X) ---------------------------------------

plot(cuantitativas$pH,
     cuantitativas$Gusanos,
     xlab = "Independiente (pH)",
     ylab = "Dependiente (Gusanos)",
     pch  = 16,
     col  = "blue",
     main = "Relación entre pH y gusanos")


## 4. Ajuste del modelo de regresión lineal --------------------------------

# Usar fórmula con nombres hace el modelo más legible [web:15]
modelo_lm <- lm(Gusanos ~ pH, data = cuantitativas)

# Resumen del modelo
summary(modelo_lm)


## 5. Recta ajustada sobre el gráfico --------------------------------------

# Ordenar los datos por pH para que la línea se dibuje sin “dientes”
ord <- order(cuantitativas$pH)
lines(cuantitativas$pH[ord],
      fitted(modelo_lm)[ord],   # valores ajustados del modelo [web:1][web:14]
      col  = "red",
      lwd  = 2)


## 6. Diagnóstico: residuales ----------------------------------------------

# Gráfico residuales vs valores ajustados (más estándar que X vs residuales) [web:2][web:11][web:14]
plot(fitted(modelo_lm),
     resid(modelo_lm),
     xlab = "Valores ajustados",
     ylab = "Residuales",
     main = "Residuales vs Valores ajustados",
     pch  = 16,
     col  = "darkgreen")
abline(h = 0, lty = 2, col = "gray40")

# (Opcional) usar los gráficos diagnósticos integrados de lm [web:14]
# plot(modelo_lm, which = 1)   # residuales vs ajustados
# plot(modelo_lm)              # todos los gráficos diagnósticos
