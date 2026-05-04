# Simulación de Perfil Parabólico en Flujo Estacionario con OpenFOAM

## Overview
Implementa una simulación numérica de flujo incompresible alrededor de un perfil parabólico suave, ideal para validar solvers de CFD en geometrías curvas simples. El perfil se parametriza por envergadura, altura y curvatura, generando perfiles analíticos que evitan mallas complejas.

**Aplicaciones típicas:** Validación de numerics, pruebas de convergencia, benchmark de solvers personalizados.

## Requisitos
- OpenFOAM v10+ (probado en v11)
- GNU Make y gcc/g++
- Entorno Linux (WSL en Windows funciona bien)

## Instalación Rápida
```bash
git clone https://github.com/Nickware/Master.git
cd C++/parabolicChannel
make all
```

## Uso Básico
```bash
make run          # Compila, genera malla y ejecuta
paraFoam -builtin # Visualiza resultados
```

Los resultados se guardan en `results/` con campos de velocidad \( \mathbf{U} \), presión \( p \), y métricas de calidad.

## Parámetros Configurables
Edita `parabolicProfile.C` para ajustar la geometría:

```cpp
scalar referenceLength = 1.0;  // Escala del dominio [m]
scalar height = 0.5;           // Altura máxima del perfil [m] 
scalar curvature = 0.5;        // Factor de curvatura (0=plano, 1=alta curvatura)
```

**Ejemplo de perfil generado:** \( y = h \cdot (1 - (x/L)^2)^c \) donde \( h=\)height, \( c=\)curvature.

***

## Instrucciones Detalladas (Paso a Paso)

### 1. Configuración del Caso
Crea y prepara el directorio base desde el tutorial `pitzDaily`:

```bash
mkdir -p $FOAM_RUN/parabolicChannel && cd $_
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/* .
rm -rf 0 constant/polyMesh system/* dynamicCode
```

### 2. Generación de Malla Parabólica
Reemplaza `system/blockMeshDict` con la geometría parabólica y genera:

```bash
rm -rf constant/polyMesh
# Edita system/blockMeshDict con nano/vi
blockMesh
checkMesh        # Verifica calidad (non-orthogonality < 70° ideal)
paraFoam -touch && paraFoam -builtin  # Preview 3D
```

### 3. Solver Personalizado (Opcional)
Si necesitas integrar el solver en tu `$FOAM_APP`:

```bash
# Verifica entorno
echo $FOAM_APP $FOAM_USER_APPBIN $FOAM_RUN

# Crea solver directory
mkdir -p $HOME/OpenFOAM/$WM_PROJECT_VERSION/solvers/parabolicSolver
export FOAM_USER_APP=$HOME/OpenFOAM/$WM_PROJECT_VERSION/solvers

cd $FOAM_USER_APP/parabolicSolver
mkdir -p Make
touch Make/{files,options} parabolicFoam.C
wmake          # Compila solver
```

### 4. Ejecución y Post-Procesado
```bash
simpleFoam      # O tu solver personalizado
postProcess -func 'wallShearStress'  # Métricas avanzadas
```

**Salidas clave en `results/`:**
- `U_mag.png`: Magnitud de velocidad en corte central
- `Cp_distribution.dat`: Coeficiente de presión vs posición
- `convergence.log`: Residuos de convergencia