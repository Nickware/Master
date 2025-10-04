library(ggplot2)
library(markovchain)
library(randomForest)
data(iris)

# Visualización exploratoria
ggplot(iris, aes(x = Petal.Length, y = Sepal.Length, color = Species)) +
  geom_point(size=2) +
  labs(title="Relación entre largo de pétalo y sépalo por especie") +
  theme_minimal()

# Markov chain 
iris_ord <- iris[order(iris$Petal.Length), ]
species_chain <- new("markovchain", states = unique(iris_ord$Species),
                     transitionMatrix = matrix(0, nrow=3, ncol=3)) # Inicializar

# Calcular matriz de transición observada
mc_fit <- markovchainFit(data=as.character(iris_ord$Species))
print(mc_fit$estimate)

# Random Forest
set.seed(123)
modelo_rf <- randomForest(Species ~ ., data=iris, importance=TRUE)
print(modelo_rf)
