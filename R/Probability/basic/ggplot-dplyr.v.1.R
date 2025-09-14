# ggplot2 básico
library(ggplot2)
ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_point()
myplot <- ggplot(iris, aes(Sepal.Length, Sepal.Width))
myplot + geom_point()
ggplot(iris, aes(Sepal.Length, Sepal.Width)) + geom_point(size = 3)
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species, shape = Species)) + geom_point(size = 3)

# dplyr manipulación
library(dplyr)
select(iris, Sepal.Length)
select(iris, -Sepal.Length)
select(iris, Sepal.Length:Petal.Length)
select(iris, starts_with("S"))
filter(iris, Sepal.Length >= 4.6)
filter(iris, Sepal.Length >= 4.6, Petal.Width >= 0.5)

iris %>% select(Sepal.Length, Sepal.Width) %>% head()
iris %>% arrange(Sepal.Width) %>% head()
iris %>% mutate(proportion = Sepal.Length / Sepal.Width)
iris %>% summarize(avg_slength = mean(Sepal.Length))

iris %>% group_by(Sepal.Length) %>%
  summarize(
    avg_slength = mean(Sepal.Length),
    min_slength = min(Sepal.Length),
    max_slength = max(Sepal.Length),
    total = n()
  )

