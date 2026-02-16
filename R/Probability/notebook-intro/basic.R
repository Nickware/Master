# Ayuda y documentación
# Use ayuda () función? Para acceder a la documentación R para funciones y paquetes
help("mean")      # Obtener ayuda sobre la función media ()
help("function")  # Obtener ayuda para definir funciones en R
?floor            # Referencia rápida para la función de piso ()
?plot             # Información sobre la traza con la trama ()

# Vectores y datos faltantes
# Crear un vector numérico, incluido un valor faltante (NA)
data_vector <- c(36, NA, 37, 38, 23, 23, 28)
is.na(data_vector)             # Verificar qué elementos son NA
clean_vector <- data_vector[!is.na(data_vector)]  # Eliminar los valores de NA

mean(data_vector, na.rm=TRUE)  # Calcular la media mientras ignora los valores de NA
mean(clean_vector)             # Significa usar vector sin NA

# Función media personalizada (ignora NA)
my_mean <- function(x) {
  sum(x, na.rm=TRUE) / sum(!is.na(x))
}
my_mean(data_vector)           # Prueba de la función personalizada

# 3. Aritmética y reciclaje vectorial
x <- c(1, 3, 5, 6, 8, 20, 12)
y <- c(5, 6, 7)
x + y       # Demuestra la regla de reciclaje de R; advertencia si las longitudes no son múltiplos

# Resúmenes y rangos
range(data_vector, na.rm=TRUE)      # Devuelve Min y Max
diff(data_vector)                   # Diferencias entre elementos consecutivos
diff(range(data_vector, na.rm=TRUE))# Rango (max - min) ignorar los datos faltantes

# Generación y repetición de secuencia
seq1 <- seq(1, 100, 10)          # Secuencia de 1 a 100 con el paso 10
seq2 <- seq(0.1, 0.9, 0.1)       # Secuencia con decimales
seq3 <- seq(0, 100, length.out=100) # 100 Números espaciados uniformemente entre 0 y 100

cities <- c("Cali", "Pasto", "Bogota", "Tunja")
rep_seq1 <- rep(cities, 5)         # Repetir el vector completo 5 veces
rep_seq2 <- rep(cities, each=5)    # Repetir cada elemento 5 veces

# Fechas
dates <- seq(as.Date("2019-01-01"), as.Date("2019-12-31"), by="days") # Secuencia diaria para 2019

# Manipulación de cadenas
labs1 <- paste(c("x", "z"), 1:10, sep="")   # Concatenar sin separador: x1, z2, etc.
labs2 <- paste("Sample", 1:5, sep="_")      # "muestra1",Etc

# Matrices y matrices
mat <- matrix(1:20, ncol=5, byrow=FALSE)    # Crear una matriz 4x5 (llena de columna)
arr <- array(1:20, dim=c(4,5))              # Crear matriz 4x5

mat[ ,1]        # Primera columna
arr[ ,2]        # Segunda columna
arr[arr[ ,2] > 6, 2]  # Elementos en la segunda columna> 6

# Listas
mylist <- list(
  numbers = 1:10,
  random_numbers = rnorm(25),
  letters = letters[1:3]
)
mylist$numbers

# Trazado básico
ages <- c(36, 36, 38, 27, 23, 23, 28)
heights <- c(1.70, 1.69, 1.87, 1.78, 1.8, 1.7, 1.7)
names <- c("Rodrigo", "Nicolas", "Andres", "Jorge", "Jose", "Erick", "Sergio")

plot(ages, heights, type="p", pch=16, xlim=c(18,40), ylim=c(1.5,2.0), col="blue",
     main="Heights by Age", xlab="Age [years]", ylab="Height [m]")
text(ages, heights, labels = names, cex=0.7, col="red") # Etiqueta cada punto

# Opcional: trazar histograma y densidad con GGPLOT2
if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

# Simular 1000 alturas, trazar histograma + densidad
set.seed(123)
sim_heights <- rnorm(1000, mean=1.75, sd=0.25)
df <- data.frame(heights = sim_heights)
ggplot(df, aes(x = heights)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  labs(
    title = "Simulated Height Distribution",
    x = "Height (m)",
    y = "Density"
  ) +

  theme_minimal()
