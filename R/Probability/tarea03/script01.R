datosFragata = read.csv("DatosEjercicios.csv", header = TRUE, sep = ",");
View(datosFragata)
cuantitativas<-datosFragata[,c(2,3,5,7)]
plot(cuantitativas[,4], cuantitativas[,3], ylab="Dependiente (Gusanos)", xlab="Independiente (pH)")
RegresionLineal<-lm(cuantitativas[,4]~cuantitativas[,3])
summary(RegresionLineal)
lines(cuantitativas[,3],RegresionLineal$fitted.values)
plot(cuantitativas[,3],RegresionLineal$residuals)
abline(h=0)