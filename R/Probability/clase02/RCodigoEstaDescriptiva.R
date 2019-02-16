######### Diapositiva 9 ######
datos<-read.csv("DatosEjercicios.csv", header = TRUE, sep = ",")
summary(datos[,4])
table(datos[,4])
barplot(table(datos[,4]), las=1, xlab="Tipo de vegetacion", ylab="Frecuencia absoluta", col="red")

######### Diapositiva 11 ######
conteo<-hist(datos[,5], breaks=6, plot=FALSE)
hist(datos[,5], breaks = 6, plot=TRUE, col="pink")
hist(datos[,5], seq(from=1, to=6), plot=TRUE, col="pink")

######### Diapositiva 12 ######
hist(datos[,5], breaks=6, plot=TRUE, col="grey50", main="", xlab="Clases", ylab="Frecuencia absoluta")

######### Diapositiva 14 ######
absoluta<-hist(datos[,5], breaks=6, plot=FALSE)
relativa<-absoluta[[2]]/sum(absoluta[[2]])
plot(absoluta[[4]],relativa, type="b", ylim=c(0,0.5))

######### Diapositiva 15 ######
tablacontingencia <- table(datos[,6], datos[,4])
addmargins(tablacontingencia)
barplot(tablacontingencia, beside = TRUE, col=c("red80","grey40"))
legenda<-paste("Damp?",as.factor(unique(datos[,6])), sep=":")
legend("topright", legenda, bty="n", cex=0.8, pch=c(15,15), col=c("grey80","grey40"))

######### Diapositiva 17 ######
wt <- c(20,20,40,10,10)
x <- c(3.0, 3.5, 2.8, 4.5, 4.0)
xm <- weighted.mean(x, wt)

######### Diapositiva 18 ######
ins <- c(10, 1, 1000, 1, 10, 10,100)
prod(ins)**(1/7)

######### Diapositiva 21 ######
velo<-c(10,20,40,10) ### mean(velo)
1/mean(1/velo)

######### Diapositiva 25 ######
N<-1000000
population <- rnorm(N, 180) 
plot(density(population, na.rm = T)) 
abline(v = mean(population), lwd=1,col="black",lty=2) 
n1<-10000; n2<-100; n3<-10 
X1 <- sample(population, n1) 
X2 <- sample(population, n2) 
X3 <- sample(population, n3) 
abline(v = mean(X1), lty = 1, col="green") 
abline(v = mean(X2), lty = 1, col="blue") 
abline(v = mean(X3), lty = 1, col="red")
legend("topright", c("Media Poblacional","Media n=10000","Media n=100","Media n=10"), cex=0.6, lty = c(2,rep(1,3)),col=c("black","green","blue","red"))

######### Diapositiva 27 ######

 plot(density(population, na.rm = T))
 abline(v = mean(population), lwd=1,col='black',lty=2)  
 n1<-10000; n2<-100; n3<-10; sim<-500 
 for (j in 1:sim) {  
              X1 <- sample(population, n1)  
              X2 <- sample(population, n2)  
              X3 <- sample(population, n3) 
              abline(v = mean(X1), lty = 1, col="grey40")  
              abline(v = mean(X2), lty = 1, col="grey60")
              abline(v = mean(X3), lty = 1, col="grey80") }
legend("topright", c("Media Poblacional","Medias n=10000","Medias n=100","Medias n=10"), cex=0.6, lty = c(2,rep(1,3)),col=c("black","grey40","grey60","grey80"))

# ######### Diapositiva 28 ######
# error<-NULL 
# for (j in 1:sim){  
# X1 <- sample(population, n1) 
# X2 <- sample(population, n2) 
# X3 <- sample(population, n3) 
# Er.x1<-abs(mean(population) - mean(X1))  
# Er.x2<-abs(mean(population) - mean(X2))  
# Er.x3<-abs(mean(population) - mean(X3)) 
# ErrorEst<-c(mean(population), Er.x1, Er.x2, Er.x3) 
# error<-rbind(error, ErrorEst) }
# errores.medios<-apply(error,2,mean)  

######### Diapositiva 29 ######
N<-1000000; inusuales<-100000 
population <- c(rnorm(N, 180), seq(180,220,length.out= inusuales))
plot(density(population, na.rm = T))
abline(v = mean(population), lwd=1,col="black",lty=2) 
n1<-10000; n2<-100; n3<-10 
X1 <- sample(population, n1) 
X2 <- sample(population, n2) 
X3 <- sample(population, n3) 
abline(v = mean(X1), lty = 1, col="green")
abline(v = mean(X2), lty = 1, col="blue") 
abline(v = mean(X3), lty = 1, col="red" ) 

######### Diapositiva 31 ######
N<-1000000; inusuales<-100000 
population <- c(rnorm(N, 180), seq(180,220,length.out= inusuales))
plot(density(population, na.rm = T))
abline(v = mean(population), lty = 1, col="green") 
abline(v = median(population), lty = 1, col="blue")

######### Diapositiva 32 ######
cuantitativas<-datos[,c(2,3,5,7)]
medianas<-apply(cuantitativas,2,median)

######### Diapositiva 33 ######
tapply(datos[,7],datos[,6],median)

######### Diapositiva 34 ######
moda.WormDesnity<- table(datos[,7])
moda.WormDesnity[moda.WormDesnity == max(moda.WormDesnity)]
names(moda.WormDesnity)[moda.WormDesnity == max(moda.WormDesnity)]

######### Diapositiva 35 ######
rangos<-apply(cuantitativas,2,range)

######### Diapositiva 38 ######
varianzas<-apply(cuantitativas,2,var)

######### Diapositiva 40 ######
desv.est<-apply(cuantitativas,2,sd)
medias<-apply(cuantitativas,2,mean)
x<-seq(along.with=medias)
plot(x,medias, xaxt="n", ylab="Valores", xlab="",ylim=c(-2,8), las=2, pch=15)
axis(1,at=x, labels=names(medias)) 
arrows(x, medias-desv.est,x, medias+desv.est, code=3, angle=90,length=0.2)  

######### Diapositiva 42 ######
round(apply(datos[,c(2,3,5,7)], 2,mean),2)
round(apply(datos[,c(2,3,5,7)], 2,median),2)
round(apply(datos[,c(2,3,5,7)], 2,sd),2)
CV<-round(apply(datos[,c(2,3,5,7)], 2,sd)/apply(datos[,c(2,3,5,7)], 2,mean)*100,2)

######### Diapositiva 44 ######
cuartiles<-quantile(cuantitativas[,1])
deciles<-quantile(cuantitativas[,1], probs=seq(0,1,by=0.1))
centiles<-quantile(cuantitativas[,1], probs=seq(0,1,by=0.01))

######### Diapositiva 45 ######
DT<-rnorm(1000) 
Qs<-quantile(DT)
length(which(DT>=Qs[1] & DT<Qs[2])) 
length(which(DT>=Qs[2] & DT<Qs[3])) 
length(which(DT>=Qs[3] & DT<Qs[4])) 
length(which(DT>=Qs[4] & DT<=Qs[5]))
diff(cuartiles) 

######### Diapositiva 46 ######
library(reshape) 
reshape.cuanti<-melt(cuantitativas) 
reshape.cuanti$id<-rep(seq(1,4),each=20) 
boxplot(cuantitativas, las=1) 
points(reshape.cuanti[,3],reshape.cuanti[,2], cex=0.7, pch=16, col="red") 
points(c(1:4),apply(cuantitativas,2, mean), pch=3, col="blue", cex=1.5) 

######### Diapositiva 49 ######
Dat.Inu<-c(13, 16.3, 20.5, 18.7, 18, 18, 18.8, 22.3, 19.7, 18.1, 20, 24) 
boxplot(Dat.Inu, pch=16, cex=0.7, ylim=c(12,25)) 
cuartiles<-quantile(Dat.Inu) 
RIQ<-cuartiles[4] - cuartiles[2]  
Li<-cuartiles[2]-(1.5*RIQ)  
Ls<-cuartiles[4]+(1.5*RIQ)  
abline(h=Ls, lwd=2,lty=3, col="red")  
abline(h=Li, lwd=2,lty=3, col="red")  

######### Diapositiva 51 ######
var(cuantitativas[,3],cuantitativas[,4]) 
cov(cuantitativas[,3],cuantitativas[,4]) 
Varianzas<-diag(cov(cuantitativas))
Covarianzas<-cov(cuantitativas)

######### Diapositiva 52 ######
cor(cuantitativas[,3],cuantitativas[,4]) 
cor.test(cuantitativas[,3],cuantitativas[,4]) 
plot(cuantitativas[,3],cuantitativas[,4], pch=16) 
text(3.8, 8, paste("r",round(cor(cuantitativas[,3],cuantitativas[,4]),3),sep="=")) 

######### Diapositiva 58 ######
Fragata<-read.csv("/Users/rodrigogilcastaneda/Dropbox/Estadistica maestría modelado y simulacion/Fragata.csv")
par(mfrow=c(1, 2))
hist(Fragata[,1], breaks=6 ,main="Volumen")
hist(Fragata[,2], breaks=6 ,main="Frecuencia")
cor.test(Fragata[,1],Fragata[,2], method="spearman") 

######### Diapositiva 60 ######

x <- c(5,2,8,4,6,3,1,7)
y <- c(90,87, 89, 60, 85, 84, 75, 91)
cor.test(x, y, method = "kendall")

######### Diapositiva 69 ######
plot(cuantitativas[,4]~cuantitativas[,3], ylab="Dependiente (Gusanos)", xlab="Independiente (pH)" )
RegresionLineal<-lm(cuantitativas[,4] ~ cuantitativas[,3]) 
summary(RegresionLineal) 
lines(cuantitativas[,3],RegresionLineal$fitted.values)
plot(cuantitativas[,3],RegresionLineal$residuals) 
abline(h=0) 