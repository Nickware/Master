# package dplyr
install.packages("dplyr")
library(dplyr)
head(iris)
head(select(iris, Sepal.Length))
head(select(iris, -Sepal.Length))
head(select(iris, Sepal.Length:Petal.Length))
head(select(iris, starts_with("S")))

filter(iris, Sepal.Length >= 4.6)
filter(iris, Sepal.Length >= 4.6, Petal.Width>=0.5)
iris %>% select(Sepal.Length, Sepal.Width) %>% head
iris %>% arrange(Sepal.Width) %>% head
iris %>% mutate(proportion = Sepal.Length/Sepal.Width)
iris %>% summarize(avg_slength = mean(Sepal.Length))

iris %>% group_by(Sepal.Length) %>%
  summarise(avg_slength = mean(Sepal.Length),
            min_slength = min(Sepal.Length),
            max_slength = max(Sepal.Length), total = n())

#Calcular el promedio de todas las especies
iris %>% group_by(Species) %>% summarise_all(funs(mean)) %>% data.frame
iris %>% group_by(Species) %>% summarise(Tamano = n(), Media=mean(Petal.Width)) %>% data.frame
iris %>% group_by(Species) %>% summarise(Tamano = n(), Media=mean(Petal.Width)) %>% mutate(Operacion=Tamano*Media)

