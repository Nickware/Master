# Función promedio mejorada que ignora los valores de NA
promedio <- function(DATOS){
  n <- sum(!is.na(DATOS))               # Contar solo elementos no NA
  suma <- sum(DATOS, na.rm=TRUE)        # Suma, ignorando los valores de NA
  promedio <- suma / n
  return(promedio)
}

# Ejemplo de uso de la función Promedio
prueba <- c(36, NA, 37, 38, 23, 23, 28)
print(promedio(prueba))       # Imprime el promedio, excluyendo NA

# Cargar el paquete GGPLOT2 (instalar si es necesario)
if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

# Establecer semilla para la reproducibilidad
set.seed(123)

# Simular datos de altura
estaturas <- rnorm(1000, mean = 1.75, sd = 0.25)

# Convertir a la trama de datos para usar con GGPLOT2
df <- data.frame(estaturas = estaturas)

# Trazar histograma con curva de densidad usando GGPLOT2
ggplot(df, aes(x = estaturas)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  labs(
    title = "Distribution of Heights",
    x = "Height (m)",
    y = "Density"
  ) +
  theme_minimal()