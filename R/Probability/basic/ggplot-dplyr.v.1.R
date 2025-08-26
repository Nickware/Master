# ggplot2 básicos con iris
library(ggplot2)

# Scatter plot simple
ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_point()

# Guardar gráfico en objeto y añadir capa después
myplot <- ggplot(iris, aes(Sepal.Length, Sepal.Width))
myplot + geom_point()

# Cambiar tamaño de puntos
ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_point(size = 3)

# Colorear y cambiar forma según Species
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species, shape = Species)) + 
  geom_point(size = 3)

# dplyr para manipulación con iris
library(dplyr)

# Selección de columnas
select(iris, Sepal.Length)               # sólo Sepal.Length
select(iris, -Sepal.Length)              # todas menos Sepal.Length
select(iris, Sepal.Length:Petal.Length) # rango columnas
select(iris, starts_with("S"))           # columnas que inician con "S"

# Filtrar filas
filter(iris, Sepal.Length >= 4.6)
filter(iris, Sepal.Length >= 4.6, Petal.Width >= 0.5)

# Pipe operator para combinación de operaciones
iris %>% select(Sepal.Length, Sepal.Width) %>% head()
iris %>% arrange(Sepal.Width) %>% head()
iris %>% mutate(proportion = Sepal.Length / Sepal.Width)
iris %>% summarize(avg_slength = mean(Sepal.Length))

# Agrupar por Sepal.Length y resumir
iris %>% group_by(Sepal.Length) %>%
  summarize(
    avg_slength = mean(Sepal.Length),
    min_slength = min(Sepal.Length),
    max_slength = max(Sepal.Length),
    total = n()
  )

