% Limpiar workspace
clear; close all; clc;

% Localizar el archivo CSV en diferentes ubicaciones posibles
scriptDir = fileparts(mfilename('fullpath'));
candidates = {
    fullfile(scriptDir, 'octaveData', 'velocityData.csv');
    fullfile(scriptDir, '..', 'C++', 'parabolicChannel', 'octaveData', 'velocityData.csv');
    fullfile(pwd, 'octaveData', 'velocityData.csv');
    fullfile(pwd, 'C++', 'parabolicChannel', 'octaveData', 'velocityData.csv');
};

csvFile = '';
for i = 1:numel(candidates)
    if exist(candidates{i}, 'file')
        csvFile = candidates{i};
        break;
    end
end

if isempty(csvFile)
    error(['No se encontró velocityData.csv. Ejecuta primero el caso de OpenFOAM y asegúrate de que exista en una de estas rutas:\n' ...
           '  - %s\n  - %s\n  - %s\n  - %s'], ...
           candidates{1}, candidates{2}, candidates{3}, candidates{4});
end

% Leer datos
data = dlmread(csvFile, ',', 1, 0);

% Extraer componentes
x = data(:,1);
y = data(:,2);
z = data(:,3);
Ux = data(:,4);
Uy = data(:,5);
Uz = data(:,6);

% Crear figura
figure('Position', [100, 100, 1200, 500]);

% Subplot 1: Perfil de velocidad en X
subplot(1,2,1);
scatter(y, Ux, 20, 'filled');
hold on;

% Ajustar curva teórica (parábola)
y_unique = unique(y);
h = 0.1;
y_teorico = linspace(min(y), max(y), 100);
Ux_teorico = 1.0 - ((y_teorico - h/2)/(h/2)).^2;
plot(y_teorico, Ux_teorico, 'r-', 'LineWidth', 2);

title('Perfil Parabólico de Velocidad');
xlabel('Posición Y [m]');
ylabel('Velocidad Ux [m/s]');
legend('Datos OpenFOAM', 'Solución Teórica', 'Location', 'best');
grid on;

% Subplot 2: Campo vectorial 2D
subplot(1,2,2);
% Seleccionar solo cada 5to punto para mejor visualización
indices = 1:5:length(x);
quiver(x(indices), y(indices), Ux(indices), Uy(indices), 0.5, 'b');
title('Campo de Velocidad 2D');
xlabel('Posición X [m]');
ylabel('Posición Y [m]');
axis equal;
grid on;

% Visualización 3D opcional
figure;
indices = 1:10:length(x); % Reducir densidad para visualización 3D
quiver3(x(indices), y(indices), z(indices),...
        Ux(indices), Uy(indices), Uz(indices), 0.5);
title('Campo de Velocidad 3D');
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
grid on;
rotate3d on;
