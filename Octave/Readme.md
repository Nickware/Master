# Análisis y visualización de datos de velocidad provenientes de una simulación CFD

Este rutina en Octave (o MATLAB) realiza un análisis y visualización de datos de velocidad provenientes de una simulación CFD (por ejemplo, de OpenFOAM) y genera representaciones gráficas comparativas y vectoriales en 2D y 3D. Combina lectura de datos, extracción de componentes, graficación y comparación con un perfil teórico parabólico de flujo laminar.[1][2][3][4]

***

### Limpieza y lectura de datos
La primera sección elimina cualquier variable o figura anterior y carga un archivo CSV con los datos de simulación:

```matlab
clear; close all; clc;
data = dlmread('octaveData/velocityData.csv', ',', 1, 0);
```

El comando `dlmread` lee el archivo `velocityData.csv`, saltando una fila de encabezado y separando columnas por comas. Los datos suelen contener coordenadas espaciales (x, y, z) y componentes de velocidad (Ux, Uy, Uz) exportados de OpenFOAM.[5][1]

***

### Extracción de componentes
Cada columna se separa en variables para su análisis:

```matlab
x = data(:,1); y = data(:,2); z = data(:,3);
Ux = data(:,4); Uy = data(:,5); Uz = data(:,6);
```

Esto facilita generar gráficos vectoriales y perfiles independientes por componente.[6][7]

***

### Subplot 1: Perfil de velocidad parabólica
La primera figura muestra el perfil experimental (o numérico) comparado con una solución teórica de flujo laminar entre placas planas:

```matlab
subplot(1,2,1);
scatter(y, Ux, 20, 'filled');
hold on;
```

El perfil teórico se modela con una parábola $ U_x(y) = 1 - ((y - h/2)/(h/2))^2 $, lo cual representa la distribución de velocidad en un canal simétrico donde la velocidad máxima ocurre en el centro y cero en las paredes :[2][8]

```matlab
Ux_teorico = 1.0 - ((y_teorico - h/2)/(h/2)).^2;
```

Esto permite evaluar la concordancia del flujo simulado respecto al comportamiento físico esperado.

***

### Subplot 2: Campo vectorial 2D
La segunda figura muestra el campo de velocidades $ (U_x, U_y) $ sobre el plano XY, ideal para visualizar dirección y magnitud del flujo :[3][9]

```matlab
quiver(x(indices), y(indices), Ux(indices), Uy(indices), 0.5, 'b');
```

El comando `quiver` genera flechas 2D que representan los vectores de velocidad. Se selecciona un subconjunto de puntos (cada 5º) para mantener la legibilidad del gráfico.

***

### Campo vectorial 3D
La figura 3D proporciona una vista vectorial del flujo completo en las tres dimensiones usando `quiver3` :[4][3]

```matlab
quiver3(x(indices), y(indices), z(indices), Ux(indices), Uy(indices), Uz(indices), 0.5);
```

En este gráfico, cada flecha está posicionada en `(x,y,z)` y apunta en la dirección del vector velocidad `(Ux,Uy,Uz)`. Se utiliza un muestreo reducido (`1:10`) para optimizar la visualización, y `rotate3d on` permite rotar la vista interactivamente.

***

### Propósito general del script
Este script permite:
- Validar resultados de simulaciones CFD frente a soluciones teóricas.
- Analizar la distribución de velocidades a lo largo de un canal o dominio.
- Visualizar el comportamiento del flujo en 2D y 3D, permitiendo detectar zonas de recirculación o gradientes indebidos.

Es especialmente útil para:
- Ensayos de flujo laminar entre placas o canal.
- Análisis de experiencias académicas con OpenFOAM.
- Validación inicial de mallas CFD y condiciones de contorno.

En resumen, se trata de una herramienta didáctica y visual para vincular teoría (perfil parabólico) con resultados numéricos (datos OpenFOAM), integrando conceptos de dinámica de fluidos y postprocesamiento científico en Octave.[8][2][3]

[1](https://www.geeksforgeeks.org/data-visualization/octave-basics-of-plotting-data/)
[2](https://www.youtube.com/watch?v=_jjYmW8sBNA)
[3](https://www.mathworks.com/help/matlab/ref/quiver3.html)
[4](https://octave.sourceforge.io/octave/function/quiver3.html)
[5](https://docs.octave.org/latest/Two_002dDimensional-Plots.html)
[6](https://www.wcc.vccs.edu/sites/default/files/Introduction-to-GNU-Octave.pdf)
[7](http://ais.informatik.uni-freiburg.de/teaching/ws11/robotics2/pdfs/rob2-03-octave.pdf)
[8](https://www.youtube.com/watch?v=qnfKQ0QuH1Y)
[9](https://www.mathworks.com/matlabcentral/answers/342259-better-visualization-of-quiver)
[10](https://www.youtube.com/watch?v=t_BvFOzYxts)
[11](https://docs.octave.org/octave-8.3.0.pdf)
[12](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2004JA010956)
[13](https://www.oreilly.com/library/view/gnu-octave-beginners/9781849513326/ch07.html)
[14](https://www.cfd-online.com/Forums/paraview/124765-how-can-plot-velocity-profile-airfoil-over-line.html)
[15](https://www.ncnr.nist.gov/resources/tk_octave/)
[16](https://www.mathworks.com/matlabcentral/answers/68049-velocity-profile-for-different-z-positions-and-x-positions-in-a-microchannel)
[17](https://www.youtube.com/watch?v=8h7yMIRBYd8)
[18](https://www.mathworks.com/help/wavelet/ug/relative-velocity-changes-in-seismic-waves-using-time-frequency-analysis.html)
[19](https://www.wolfdynamics.com/training/turbulence/OF2021/turbulence_20221_OF8_guided_tutorials.pdf)
[20](https://hlevkin.com/hlevkin/92usefulBooks/Octave/Scientific%20Computing%20with%20MATLAB%20and%20Octave%20Quarteroni,%20Saleri%20&%20Gervasio.pdf)
