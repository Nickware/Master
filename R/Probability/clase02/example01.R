library(ggplot2)
library(markovchain)
library(randomForest)

# Cargar datos
data(iris)
set.seed(123)  # Para reproducibilidad

# 1. Visualización exploratoria
p <- ggplot(iris, aes(x = Petal.Length, y = Sepal.Length, color = Species)) +
  geom_point(size = 2) +
  labs(
    title = "Relación entre largo de pétalo y sépalo por especie",
    x = "Longitud del pétalo (cm)",
    y = "Longitud del sépalo (cm)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p)

# 2. Cadena de Márkov: ordenar por Petal.Length y ajustar
iris_ord <- iris[order(iris$Petal.Length), ]
mc_fit <- markovchainFit(data = as.character(iris_ord$Species))

cat("Estimación de la cadena de Márkov:\n")
print(mc_fit$estimate)

# 3. Random Forest: clasificación de Species
modelo_rf <- randomForest(Species ~ ., data = iris, importance = TRUE)

# Mostrar resumen del modelo
print(modelo_rf)

# Opcional: importancia de variables
cat("\nImportancia de las variables (MeanDecreaseGini):\n")
print(importance(modelo_rf))

# Opcional: gráfica de importancia
varImpPlot(modelo_rf, main = "Importancia de variables en Random Forest")

