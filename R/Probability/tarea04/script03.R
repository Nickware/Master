datosCelulas = read.csv("CelulasAnormales.csv", head=TRUE, sep=",");
View(datosCelulas)
tabla<-datosCelulas[,-1]
# ¿Cuál es la probabilidad de que un paciente remitido tenga más de 
# 5 células anormales por ml de sangre?
ppois(tabla, lambda = 6)
# hist(tabla, xlab="Valores", main="")
# ¿Cuál es la probabilidad de que un paciente remitido tenga menos 
# de 7 células anormales por ml de sangre?
menos<-ppois(tabla, lambda = 6)
1-menos
# ¿Cuál es la probabilidad de que un paciente remitido tenga entre 
# 3 y 8 células anormales por ml de sangre?
resta<-ppois(tabla, lambda = 2)-ppois(tabla, lambda = 8)
# Si se conoce que una persona con más de 8 células anormales por ml 
# de sangre padece de la enfermedad, ¿cuál es probabilidad de que 
# un paciente remitido marque positivo para la enfermedad?
ppois(tabla, lambda = 9)