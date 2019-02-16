# File txt
file_txt = read.table("DatosEjercicios.txt", header = TRUE, sep = "")
head(file_txt)
View(file_txt)
file_txt[file_txt[,4]=="Grassland",]
file_txt[file_txt[,6]==FALSE & file_txt[,7]>3,]
cuantitativas01 <- file_txt[, c(2,3,4,5)]
plot(cuantitativas01[,3],cuantitativas01[,4])
hist(cuantitativas01[,1], probability = TRUE, nclass = 6)