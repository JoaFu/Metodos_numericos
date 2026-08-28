function [X, iter, err, hist] = newtonRaphsonSistema3( ...
    Ffun, Jfun, X0, tol, maxiter)
% NEWTONRAPHSONSISTEMA3
% Resuelve un sistema no lineal de tres ecuaciones:
%
%       F(X) = 0
%
% mediante el método de Newton-Raphson:
%
%       J(X_i)*DeltaX_i = -F(X_i)
%       X_(i+1) = X_i + DeltaX_i
%
% Entradas:
%   Ffun    : función @(x,y,z) que devuelve [F1; F2; F3]
%   Jfun    : función @(x,y,z) que devuelve el Jacobiano 3x3
%   X0      : aproximación inicial [x0; y0; z0]
%   tol     : tolerancia del error aproximado porcentual
%   maxiter : número máximo de iteraciones
%
% Salidas:
%   X       : solución aproximada [x; y; z]
%   iter    : iteraciones realizadas
%   err     : error aproximado porcentual final
%   hist    : historial [iteracion, x, y, z, error]

% Valores predeterminados
if nargin < 5
    maxiter = 100;
end

if nargin < 4
    tol = 1e-6;
end

% Validación de entradas
if ~isa(Ffun, 'function_handle')
    error('Ffun debe ser un function_handle.');
end

if ~isa(Jfun, 'function_handle')
    error('Jfun debe ser un function_handle.');
end

if numel(X0) ~= 3
    error('X0 debe contener tres elementos [x0; y0; z0].');
end

if tol <= 0
    error('La tolerancia debe ser positiva.');
end

if maxiter <= 0 || maxiter ~= floor(maxiter)
    error('maxiter debe ser un entero positivo.');
end

% Aproximación inicial como vector columna
X = reshape(X0, 3, 1);

iter = 0;
err = Inf;

% Preasignación del historial
hist = NaN(maxiter + 1, 5);
hist(1,:) = [0, X.', NaN];

while err > tol && iter < maxiter

    % Variables de la iteración actual
    x = X(1);
    y = X(2);
    z = X(3);

    % Evaluar el vector de funciones F(X_i)
    F = Ffun(x, y, z);
    F = reshape(F, 3, 1);

    % Evaluar el Jacobiano J(X_i)
    J = Jfun(x, y, z);

    if ~isequal(size(J), [3,3])
        error('Jfun debe devolver una matriz Jacobiana 3x3.');
    end

    % Comprobar que el Jacobiano puede invertirse numéricamente
    if rcond(J) < 1e-14
        error(['Jacobiano singular o mal condicionado ', ...
               'en la iteración %d.'], iter);
    end

    % Ecuación de Newton:
    %
    %       J(X_i)*DeltaX_i = -F(X_i)
    %
    DeltaX = J \ (-F);

    % Nueva aproximación:
    %
    %       X_(i+1) = X_i + DeltaX_i
    %
    Xnuevo = X + DeltaX;

    if any(~isfinite(Xnuevo))
        error('El método diverge en la iteración %d.', iter);
    end

    % Error aproximado porcentual de cada componente
    errorComponentes = zeros(3,1);

    for k = 1:3
        errorComponentes(k) = errorPorcentual( ...
            Xnuevo(k), X(k));
    end

    % Se utiliza el mayor error de las tres variables
    err = max(errorComponentes);

    % Actualizar iteración
    X = Xnuevo;
    iter = iter + 1;

    % Guardar [iteración, x, y, z, error]
    hist(iter + 1,:) = [iter, X.', err];
end

% Eliminar las filas no utilizadas
hist = hist(1:iter + 1,:);

if iter >= maxiter && err > tol
    warning(['Se alcanzó el máximo de iteraciones ', ...
             'sin cumplir la tolerancia.']);
end

end

function e = errorPorcentual(nuevo, anterior)
% ERRORPORCENTUAL Calcula el error aproximado porcentual:
%
%       e = |(nuevo - anterior)/nuevo|*100

if abs(nuevo) > eps
    e = abs((nuevo - anterior)/nuevo)*100;
else
    % Evita división entre cero
    e = abs(nuevo - anterior)*100;
end

end


%-----Ejemplo de uso :) -----------
% Vector de funciones:
%
%   F1 = x^2 + y^2 + z^2 - 1
%   F2 = x + y + z
%   F3 = x - y

Ffun = @(x,y,z) [x^2 + y^2 + z^2 - 1; x + y + z; x - y];

% Matriz Jacobiana:
Jfun = @(x,y,z) [
    2*x, 2*y, 2*z;
    1,   1,   1;
    1,  -1,   0
    ];

% Aproximación inicial
X0 = [0.5; 0.5; -1];

% Ejecutar Newton-Raphson
[X, iter, err, hist] = newtonRaphsonSistema3( ...
    Ffun, Jfun, X0, 1e-8, 50);

disp('Solución aproximada:')
disp(X)

fprintf('Iteraciones: %d\n', iter);
fprintf('Error final: %.12g %%\n', err);

% Mostrar historial como tabla
tabla = array2table(hist, ...
    'VariableNames', {'Iteracion','x','y','z','ErrorPorcentual'});

disp(tabla)