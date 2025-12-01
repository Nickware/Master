# Versión mejorada: Análisis exploratorio de iris con dplyr
# Buenas prácticas: sin install.packages(), funciones modernas, salida clara

library(dplyr)
library(ggplot2)  # Para visualizaciones opcionales

data(iris)  # Cargar datos explícitamente

# ============================================================================
## 1. SELECCIÓN DE COLUMNAS (select)
# ============================================================================
cat("=== SELECCIÓN DE COLUMNAS ===\n")
head(iris)  # Vista general
head(select(iris, Sepal.Length))  # Una columna
head(select(iris, -Sepal.Length))  # Todas menos una
head(select(iris, Sepal.Length:Petal.Length))  # Rango de columnas
head(select(iris, starts_with("S")))  # Patrón con starts_with()

# ============================================================================
## 2. FILTRADO DE FILAS (filter)
# ============================================================================
cat("\n=== FILTRADO DE FILAS ===\n")
head(filter(iris, Sepal.Length >= 4.6))
head(filter(iris, Sepal.Length >= 4.6, Petal.Width >= 0.5))

# ============================================================================
## 3. PIPE OPERATOR (%)>% - FLUJO PRINCIPAL
# ============================================================================
cat("\n=== PIPE OPERATOR (magrittr) ===\n")
iris %>% 
  select(Sepal.Length, Sepal.Width) %>% 
  head()

iris %>% 
  arrange(Sepal.Width) %>% 
  head()

iris %>% 
  mutate(
    proportion = Sepal.Length / Sepal.Width,
    total_length = Sepal.Length + Petal.Length
  ) %>% 
  head()

# ============================================================================
## 4. RESUMEN SIMPLE (summarise)
# ============================================================================
cat("\n=== RESÚMENES BÁSICOS ===\n")
iris %>% 
  summarise(
    avg_slength = mean(Sepal.Length),
    sd_slength = sd(Sepal.Length),
    n = n()
  )

# ============================================================================
## 5. AGRUPACIONES POR ESPECIE (group_by + summarise)
# ============================================================================
cat("\n=== ANÁLISIS POR ESPECIE ===\n")

# Resumen completo por especie (usar across() en lugar de summarise_all obsoleto)
iris %>% 
  group_by(Species) %>% 
  summarise(
    across(where(is.numeric), list(
      media = ~mean(.x, na.rm = TRUE),
      sd = ~sd(.x, na.rm = TRUE),
      min = ~min(.x, na.rm = TRUE),
      max = ~max(.x, na.rm = TRUE)
    )),
    total = n(),
    .groups = "drop"
  )

# Resumen específico y operaciones derivadas
iris_summary <- iris %>% 
  group_by(Species) %>% 
  summarise(
    Tamano = n(),
    Media_PetalWidth = mean(Petal.Width, na.rm = TRUE),
    .groups = "drop"
  ) %>% 
  mutate(
    Total_PetalWidth = Tamano * Media_PetalWidth,
    Prop_PetalWidth = Media_PetalWidth / sum(Media_PetalWidth)
  )

print(iris_summary)

# ============================================================================
## 6. VISUALIZACIÓN RÁPIDA (BONUS)
# ============================================================================
cat("\n=== VISUALIZACIÓN RÁPIDA ===\n")
# Boxplot por especie
iris %>% 
  ggplot(aes(x = Species, y = Petal.Width, fill = Species)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Distribución de Petal.Width por especie") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Gráfico del resumen
iris_summary %>% 
  ggplot(aes(x = Species, y = Media_PetalWidth, fill = Species)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.1f", Media_PetalWidth)), vjust = -0.3) +
  labs(title = "Media de Petal.Width por especie") +
  theme_minimal()
