# app.R - Lanzador
# Shiny detecta automáticamente global.R, ui.R y server.R si corres
# runApp("baloto_app/") sobre la carpeta completa. Este archivo es opcional
# y solo sirve como punto de entrada explícito si prefieres source() manual.

source("global.R")
source("ui.R")
source("server.R")

shinyApp(ui = ui, server = server)
