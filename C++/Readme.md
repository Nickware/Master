## Simulación de un Perfil Parabólico empleando OpenFOAM 
> Se utiliza Octave para la visualización de los resultados 

Se parte de un campo de velocidad simple

# 1. Pasos para implementar la Geometría

> 1. Eliminar la malla vieja (si existe)
> rm -rf constant/polyMesh

> 2. Guardar el nuevo blockMeshDict
> nano system/blockMeshDict  

> 3. Generar la malla
> blockMesh

> 4. Verificar la malla
> checkMesh

# 2. Visualización de la Malla 

> 1. paraFoam -touch
> 2. paraFoam -builtin
