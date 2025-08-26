# Función para procesar archivo (csv o txt)
procesar_archivo <- function(nombre_archivo, separador) {
  datos <- read.table(nombre_archivo, header = TRUE, sep = separador)
  print(head(datos))
  
  # Filtrar filas donde la 4ª columna es "Grassland"
  print(datos[datos[,4] == "Grassland", ])
  
  # Filtrar filas donde la 6ª columna es FALSE y la 7ª columna > 3
  print(datos[datos[,6] == FALSE & datos[,7] > 3, ])
  
  # Seleccionar columnas cuantitativas específicas
  cuantitativas <- datos[, c(2,3,4,5)]
  
  # Gráficos básicos
  plot(cuantitativas[,3], cuantitativas[,4], main = paste("Scatter plot", nombre_archivo))
  hist(cuantitativas[,2], probability = TRUE, breaks = 6, main = paste("Histogram", nombre_archivo))
  
  # Gráfico con ggplot2 usando columnas SoilpH y WormDesnity (ajusta nombres según archivo)
  if("SoilpH" %in% colnames(datos) & "WormDesnity" %in% colnames(datos)) {
    require(ggplot2)
    print(ggplot(datos, aes(x = SoilpH, y = WormDesnity)) + 
            geom_point() + 
            ggtitle(paste("ggplot2 Scatter", nombre_archivo)))
  }
  
  return(datos)
}

# Uso para CSV
datos_csv <- procesar_archivo("DatosEjercicios.csv", ",")

# Uso para TXT
datos_txt <- procesar_archivo("DatosEjercicios.txt", "")
