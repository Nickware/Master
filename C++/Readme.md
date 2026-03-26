# Simulación de un Perfil Parabólico con OpenFOAM

## Overview
Este script crea una simulación numérica de un perfil parabólico en flujo estacionario utilizando OpenFOAM. El perfil parabólico se define a lo largo del eje x y puede ser ajustado mediante parámetros como la envergadura, la altura y la curvatura.

## Requerimientos
- OpenFOAM instalado en tu sistema (Linux o Windows).
- Herramientas adicionales: `make`, `gcc` u otros dependientes del sistema.
- Versión de OpenFOAM >= 10.

## Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/yourusername/parabolic-profile-simulation.git](https://github.com/Nickware/Master.git
   cd C++/parabolicChannel
   ```
2. Instalar las dependencias y compilar el script:
   ```bash
   make all
   ```

## Uso
1. Compilar y ejecutar la simulación:
   ```bash
   make run
   ```
2. Visualiza los resultados utilizando OpenFOAM's `paraFoam` o herramientas equivalentes.

## Parámetros
El script admite los siguientes parámetros en el archivo `parabolicProfile.C`:

```bash
// Eje de referencia (default: 1)
referenceLength = 1;

// Altura del perfil (default: 0.5)
height = 0.5;

// Curvatura del perfil (default: 0.5)
curvature = 0.5;
```

## Resultados
Los resultados se guardan en `results/`, incluyendo:
- Velocidades en puntos específicos.
- Distribución de presión en la superficie.

# Instrucciones paso a paso

## 1. Configuración del espacio en OpenFOAM
### 1.1 Crear directorio del caso
>    
> > mkdir -p $FOAM_RUN/parabolicChannel
> 
> > cd $FOAM_RUN/parabolicChannel

### 1.2 Crear estructura básica
> cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/* .
> 
### 1.3 Limpiar caso demo
> > rm -rf 0 constant/polyMesh system/* dynamicCode

## 2. Implementación de la Geometría

### 2.1 Eliminar la malla vieja (si existe)
>    
> > rm -rf constant/polyMesh

### 2.2 Guardar el nuevo blockMeshDict
>    
> > nano system/blockMeshDict  

### 2.3 Generar la malla
>    
> > blockMesh

### 2.4 Verificar la malla
>    
> > checkMesh

### 2.5 Visualización de la Malla 

> 1. paraFoam -touch
>    
> 3. paraFoam -builtin

## 3. Configuración del entorno de desarrollo 

## 3.1 Verificar variables de entorno primero
Verifica las variables relevantes

>> echo "FOAM_APP: $FOAM_APP"
> 
>> echo "FOAM_USER_APPBIN: $FOAM_USER_APPBIN"
> 
>> echo "FOAM_RUN: $FOAM_RUN"

## 3.2 Creación del directorio del solver
Crear en tu área de usuario (necesario si no tienes permisos en $FOAM_APP)

>> mkdir -p $HOME/OpenFOAM/solvers/parabolicSolver
>
>> export FOAM_USER_APP=$HOME/OpenFOAM/solvers

## 3.3  Estructura del directorio del solver
Navega al directorio creado
>> cd $FOAM_APP/parabolicSolver  # o
>
>> cd $FOAM_USER_APP/parabolicSolver

Crea la estructura necesaria
>> mkdir -p Make
>
>> touch Make/files Make/options
>
>> touch parabolicFoam.C



