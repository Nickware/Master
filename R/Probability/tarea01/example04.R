# Package ggplot
require(ggplot2)
ggplot(data=iris, aes(x=Sepal.Length, y=Sepal.Width)) +  geom_point()
myplot <-ggplot(data=iris, aes(x=Sepal.Length, y=Sepal.Width)) 
myplot + geom_point()
ggplot(data=iris, aes(x=Sepal.Length, y=Sepal.Width)) + geom_point(size=3)
ggplot(data=iris, aes(x=Sepal.Length, y=Sepal.Width, color=Species)) + geom_point(aes(shape=Species), size=3)