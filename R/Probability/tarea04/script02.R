
# Lanzamiento de monedas
library(animation)
library(TeachingDemos)
ani.options(interval = 0.5, nmax = 100)
## a coin would stand on the table?? just kidding :)
flip.coin(faces = c("Head","Tail"), type = "n", prob = c(0.25, 0.75), col = c(2, 4))

# Lanzamiento de monedas
dice(2, 1, sides=6)
dice(10, 2, sides=6, plot.it=T)
dice(2, 1, sides=10)
dice(2, 1, sides=12)

# Movimiento aleatorio
ani.options(interval = 0.05, nmax = 150)
brownian.motion(pch = 21, cex = 5, col = "red", bg = "yellow", 
                main = "Movimiento Aleatorio")

# Gradiente descendente
ani.options(interval = 0.3, nmax = 50)
xx = grad.desc()

# Probabilidad, valor esperado, varianza Embotelladoras
maquinasDatos = read.csv("Embotelladoras.csv", header = T, sep = ",")
View(datos)
# tabla
tabla<-datos[,-1]
# medianas 
mediasTodas<-apply(tabla,2,mean)
# maquina 1
maquina1<- table(datos[,2])
# media<-apply(datos,2,median)
# Frencuencia relativa = Frecuencias absoluta / Total  
dato1 <- as.data.frame(maquina1/sum(maquina1))
dato1$Var1 <- as.numeric(as.character(dato1$Var1))
# Probabilidad dato1$Freq)
## Valor esperado
valor <- sum(dato1$Var1*dato1$Freq)
## Varianza E(x^2)-M^2
vaianza1<- sum(dato1$Var1^2*dato1$Freq)-valor^2
# Calcular 
varianza2 <- sum(((dato1$Var1-valor)^2)*dato1$Freq)#*((dato1$Var1-dato1)^2)
# 
diabetisDatos = read.csv("DiabetisLatinoamerica.csv", header = T, sep = ",")
medias<-apply(diabetisDatos,2,mean)