#is.numeric()
promedio <- function(DATOS){
  n <- length(DATOS)
  suma <- sum(DATOS)
  promedio <- suma/n;
  return (promedio)
}
prueba <- c(36,NA,37,38,23,23,28)
promedio(prueba)

estaturas <- rnorm(100, 1.75, 0.25)
hist(estaturas, prob=1)
lines(density(estaturas))

estaturas <- rnorm(1000, 1.75, 0.25)
#hist(estaturas, plot=F)

hist(estaturas, breaks=20)