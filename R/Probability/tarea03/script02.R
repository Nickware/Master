library('scales')
datosParamo = read.csv("Paramo.csv", header = TRUE, sep = ";");
View(datosParamo)
cuantitativas<-datosParamo[,c(2,3,4,5,6,8)]
cualitativas<-datosParamo[,c(1,7)]
# Estado mas frecuente
summary(cualitativas[,2])
# Talla mas frecuente
summary(cuantitativas[,5])
# En qué estado de desarrollo se encuentra la mayor proporción de
# plantas afectadas por el insecto.
barplot(table(datosParamo[,8]), las=1, main="Estado de desarrollo vs plantas afectadas", 
        xlab="Severidad", ylab="Numero de plantas")
# Si se realizara un nuevo transepto cual seria
# la probabilidad empirica de encontrar
tablaCruzada<-table(datosParamo[,6], datosParamo[,7])
barplot(tablaCruzada, las=1, beside = TRUE,col=c("grey80","grey20"))
# Incidencia global (en todos los transeptos) de la afectación.
absoluta<-hist(datosParamo[,8], breaks=6, plot=FALSE)
relativa<-absoluta[[2]]/sum(absoluta[[2]])
# plot(absoluta[[4]],relativa, type="b", ylim=c(0,0.5))
percent(relativa / 1)
# Existe una correlación entre la altura promedio de cada transepto
# y número de frailejones
.
