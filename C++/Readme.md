# Simulación de un Perfil Parabólico empleando OpenFOAM 
> Se utiliza Octave para la visualización de los resultados 

Se parte de un campo de velocidad simple.

# Instrucciones paso a paso

## 1. Configuración del caso OpenFOAM
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

## 2. Pasos para implementar la Geometría

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
