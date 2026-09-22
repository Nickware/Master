# Flujo parabólico con OpenFOAM

Este proyecto contiene un solver pequeño de OpenFOAM que crea un campo de
velocidad con perfil parabólico en un canal rectangular y exporta sus datos
para analizarlos con Octave o MATLAB.

El canal no tiene una geometría parabólica: `system/blockMeshDict` define un
dominio rectangular de `1 x 0.1 x 0.1` metros. La parábola se aplica al
componente `Ux` de la velocidad según la coordenada `y`:

$$
U_x(y) = 1 - \left(\frac{y - h/2}{h/2}\right)^2,
\qquad h = 0.1.
$$

## Requisitos

- OpenFOAM v2012, que es la versión indicada por los archivos del caso.
- `wmake`, `gcc/g++` y GNU Make.
- Linux o WSL.
- Octave o MATLAB, solamente para el análisis posterior.

La versión exacta de OpenFOAM importa porque el código usa la estructura de
compilación y las bibliotecas de OpenFOAM. Carga el entorno antes de ejecutar
cualquier comando del caso. Por ejemplo:

```bash
source /opt/openfoam2012/etc/bashrc
command -v wmake
echo "$WM_PROJECT_DIR"
```

La ruta exacta del `bashrc` depende de la instalación local. La comprobación
debe mostrar una ruta para `wmake` y un valor para `WM_PROJECT_DIR`.

## Reproducir la simulación

Los siguientes pasos parten de la raíz del repositorio y generan la malla, la
verificación de calidad, el campo de velocidad y el CSV de salida:

```bash
cd /ruta/al/repositorio/Master
source /opt/openfoam2012/etc/bashrc

make -C C++ all
make -C C++ run
```

El objetivo `all` ejecuta `wmake` usando `C++/Make/files` y
`C++/Make/options`. El objetivo `run` ejecuta
`C++/parabolicChannel/Allrun`, que realiza estas operaciones en orden:

1. Entra en `C++/parabolicChannel` para que OpenFOAM encuentre el caso.
2. Ejecuta `blockMesh` con el diccionario local.
3. Ejecuta `checkMesh` y guarda su registro de ejecución.
4. Ejecuta `parabolicFoam`, que crea el campo `U` y exporta el CSV.

También se puede ejecutar desde el directorio del caso:

```bash
cd /ruta/al/repositorio/Master/C++/parabolicChannel
make all
make run
```

Para ejecutar los pasos manualmente después de compilar:

```bash
cd /ruta/al/repositorio/Master/C++/parabolicChannel
blockMesh
checkMesh
parabolicFoam
```

El solver no lee un campo inicial desde `0/`: crea `U` en memoria con
`IOobject::NO_READ`, asigna el perfil parabólico y lo escribe en el tiempo
`0`, definido por `startTime` en `system/controlDict`.

Para borrar el ejecutable compilado:

```bash
make -C C++ clean
```

Para regenerar completamente la malla y las salidas del caso, elimina los
resultados generados y vuelve a ejecutar:

```bash
rm -rf C++/parabolicChannel/constant/polyMesh \
	C++/parabolicChannel/0 \
	C++/parabolicChannel/octaveData \
	C++/parabolicChannel/log.*
make -C C++ run
```

## Archivos principales

- `parabolicFoam.C`: crea el campo `U`, asigna el perfil parabólico y exporta
-  las velocidades de las celdas.
- `Make/files` y `Make/options`: configuración estándar de compilación de
	OpenFOAM.
- `parabolicChannel/system/blockMeshDict`: define el canal rectangular y sus
	condiciones de contorno.
- `parabolicChannel/system/controlDict`: selecciona `parabolicFoam` y define
	el intervalo de ejecución.
- `parabolicChannel/Allrun`: ejecuta `blockMesh` y el solver.
- `parabolicChannel/Makefile`: ofrece los objetivos `all`, `run` y `clean`
	para ejecutar el flujo desde la carpeta del caso.

## Salidas

Al ejecutar el caso se generan:

- `parabolicChannel/constant/polyMesh/`: malla creada por `blockMesh`.
- `parabolicChannel/0/U`: campo de velocidad escrito por OpenFOAM, porque el
	`controlDict` usa `startTime 0`.
- `parabolicChannel/octaveData/velocityData.csv`: datos con las columnas
	`x,y,z,Ux,Uy,Uz`, listos para la rutina de Octave.
- `parabolicChannel/log.checkMesh` y `parabolicChannel/log.parabolicFoam`:
	registros generados por `Allrun` mediante `runApplication`.

No se generan actualmente imágenes, coeficientes de presión ni archivos de
convergencia en una carpeta `results/`. Esos resultados requerirían añadir un
postprocesamiento específico.

## Relación con Octave

La rutina de [`Octave/Readme.md`](../Octave/Readme.md) espera un CSV con
coordenadas y componentes de velocidad. El solver produce exactamente ese
formato en `octaveData/velocityData.csv`, por lo que la secuencia de trabajo
es:

1. Compilar y ejecutar el caso OpenFOAM.
2. Desde `Octave/`, hacer que el CSV esté disponible en la ruta que espera el
	script:

	```bash
	mkdir -p Octave/octaveData
	cp C++/parabolicChannel/octaveData/velocityData.csv Octave/octaveData/
	```

3. Ejecutar el análisis desde la carpeta `Octave`:

	```bash
	cd Octave
	octave --no-gui plotVelocity.m
	```

4. Comparar el perfil `Ux` con la solución parabólica y visualizar los campos
	vectoriales.
