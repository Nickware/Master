# File cvs
file_cvs = read.table("DatosEjercicios.csv", header = TRUE, sep = ",")
head(file_cvs)
file_cvs[file_cvs[,4]=="Grassland",]
file_cvs[file_cvs[,6]==FALSE & file_cvs[,7]>3,]
cuantitativas <- file_cvs[, c(2,3,4,5)]
plot(cuantitativas[,3],cuantitativas[,4])
hist(cuantitativas[,2], probability = TRUE, nclass = 6)