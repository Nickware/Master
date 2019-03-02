# Diabetis en latinoamerica
datosDiabetis = read.csv("DiabetisLatinoamerica.csv", header = T, sep=",")
View(datos)
medias<-apply(datos,2,mean)
# ¿Cuál es la probabilidad de que en una ciudad como Bogotá, 
# de 7,000,000 habitantes, existan más de 428,000 enfermos de Diabetes.
# Genero 
pbinom(428001, 7000000, 0.061)
# ¿Cuál es la probabilidad de que en una ciudad como Santiago de Chile, 
# de 5,600,000 habitantes, existan menos de 184,000 enfermos de Diabetes?
pbinom(183999, 5600000, 0.033)
# 1-pbinom(184000, 5600000, 0.033)
# ¿Cuál es la probabilidad de que en una ciudad como Buenos Aires, 
# de 3,400,000 habitantes, existan entre 132,000 y 133,000 enfermos 
# de Diabetes?
pbinom(c(132000, 132999), 3400000, 0.039)
resta<-pbinom(132999,3400000,0.039)-pbinom(132000,3400000,0.039)
1-resta
