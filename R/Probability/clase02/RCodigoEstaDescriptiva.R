# ==== Cargar librerías necesarias ====
library(reshape2)      # Mejor uso de reshape2 sobre reshape
library(ggplot2)       # Visualizaciones modernas
library(dplyr)         # Manipulación de datos moderna

# ==== Cargar datos ====
datos <- read.csv("DatosEjercicios.csv", header = TRUE, sep = ",")

# ==== Exploración básica ====
cat("\n--- Resumen columna 4 ---\n")
print(summary(datos[,4]))
cat("\n--- Tabla de frecuencias columna 4 ---\n")
print(table(datos[,4]))

# Barplot mejorado
ggplot(datos, aes(x = factor(datos[,4]))) +
  geom_bar(fill = "red") +
  labs(x = "Tipo de vegetación", y = "Frecuencia absoluta") +
  theme_minimal()

# ==== Histogramas variables cuantitativas (col 5) ====
# Crear cortes personalizados
breaks <- seq(min(datos[,5]), max(datos[,5]), length.out = 7)
ggplot(datos, aes(x = datos[,5])) +
  geom_histogram(breaks = breaks, fill = "pink", color = "black") +
  labs(x = "Clases", y = "Frecuencia absoluta") +
  theme_minimal()

# ==== Frecuencia relativa ====
conteo <- hist(datos[,5], breaks = 6, plot = FALSE)
relativa <- conteo$density
plot(conteo$mids, relativa, type = "b", ylim = c(0, max(relativa)),
     xlab = "Clases", ylab = "Frecuencia relativa")

# ==== Tabla de contingencia y barplot agrupado ====
tablacontingencia <- table(datos[,6], datos[,4])
print(addmargins(tablacontingencia))
barplot(tablacontingencia, beside = TRUE, col = c("red", "grey40"))
legend("topright", legend = paste("Damp?", levels(factor(datos[,6]))),
       bty = "n", cex = 0.8, pch = 15, col = c("red", "grey40"))

# ==== Medidas de posición y dispersión ====
cuantitativas <- datos[,c(2,3,5,7)]
medianas <- apply(cuantitativas, 2, median)
moda <- function(x) {
  tab <- table(x)
  as.numeric(names(tab)[tab == max(tab)])
}
moda.WormDensity <- moda(datos[,7])
rangos <- apply(cuantitativas, 2, range)
varianzas <- apply(cuantitativas, 2, var)
desv.est <- apply(cuantitativas, 2, sd)
medias <- apply(cuantitativas, 2, mean)
CV <- round(desv.est / medias * 100, 2)

# ==== Visualización de medias y desviaciones ====
x <- seq_along(medias)
plot(x, medias, xaxt = "n", ylab = "Valores", xlab = "", ylim = c(min(medias - desv.est), max(medias + desv.est)),
     las = 2, pch = 15)
axis(1, at = x, labels = names(medias))
arrows(x, medias - desv.est, x, medias + desv.est, code = 3, angle = 90, length = 0.2)

# ==== Medidas de distribución ====
cuartiles <- quantile(cuantitativas[,1])
deciles <- quantile(cuantitativas[,1], probs = seq(0, 1, by = 0.1))
centiles <- quantile(cuantitativas[,1], probs = seq(0, 1, by = 0.01))

# ==== Diagrama de caja con marcas ====
reshape.cuanti <- melt(cuantitativas)
boxplot(cuantitativas, las = 1)
points(reshape.cuanti[,3], reshape.cuanti[,2], cex = 0.7, pch = 16, col = "red")
points(1:4, apply(cuantitativas, 2, mean), pch = 3, col = "blue", cex = 1.5)

# ==== Ejemplo de estadística poblacional y simulaciones ====
set.seed(123)                     # Para reproducibilidad
N <- 1e6
population <- rnorm(N, 180)
plot(density(population, na.rm = TRUE))
abline(v = mean(population), lwd = 1, col = "black", lty = 2)

n_vals <- c(10000, 100, 10)
colors <- c("green", "blue", "red")
for(i in seq_along(n_vals)) {
  sample_mean <- mean(sample(population, n_vals[i]))
  abline(v = sample_mean, lty = 1, col = colors[i])
}
legend("topright", c("Media Poblacional", "Media n=10000", "Media n=100", "Media n=10"),
       cex = 0.6, lty = c(2, rep(1, 3)), col = c("black", colors))

# ==== Simulación múltiple para distribución de medias ====
sim <- 500
for (j in 1:sim) {
  X1 <- mean(sample(population, n_vals[1]))
  X2 <- mean(sample(population, n_vals[2]))
  X3 <- mean(sample(population, n_vals[3]))
  abline(v = X1, lty = 1, col = "grey40")
  abline(v = X2, lty = 1, col = "grey60")
  abline(v = X3, lty = 1, col = "grey80")
}
legend("topright", c("Media Poblacional", "Medias n=10000", "Medias n=100", "Medias n=10"),
       cex = 0.6, lty = c(2, rep(1, 3)), col = c("black", "grey40", "grey60", "grey80"))

# ==== Ejemplo regresión lineal ====
plot(cuantitativas[,4] ~ cuantitativas[,3], ylab = "Dependiente (Gusanos)", xlab = "Independiente (pH)")
RegresionLineal <- lm(cuantitativas[,4] ~ cuantitativas[,3])
summary(RegresionLineal)
lines(cuantitativas[,3], RegresionLineal$fitted.values, col = "blue")
plot(cuantitativas[,3], RegresionLineal$residuals)
abline(h = 0)

# ==== Guardar algunos resultados a archivo (opcional) ====
write.csv(cbind(Medias = medias, Medianas = medianas, CV = CV), 
          "resumen_estadistico.csv", row.names = FALSE)

# ==== Fin del script principal ====
lines(cuantitativas[,3],RegresionLineal$fitted.values)
plot(cuantitativas[,3],RegresionLineal$residuals) 
abline(h=0) 
