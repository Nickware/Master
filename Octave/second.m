%clc
%fprintf('Hola Mundo\n ');
%for x=1:0.5:5
 %   disp([x x^2])
%end
%exit
%*************************************************************************
%n = input('Ingrese un numero: ');
%if(n>0)
%    disp('positivo')
%else
%    disp('Negativo')
%end
%*************************************************************************
%x = input('Ingrese un numero: ');
%if(x>0)
%    fprintf('%s es positivo\n', x)
%else
%    fprintf('%s es negativo\n', x)
%end
%*************************************************************************
x = 5;
%while(x<20)
%    x=x+1;
%    %disp(2*x);
%end
%disp(x)
X(1)=input('Ingrese primer valor ');
X(2)=input('Ingrese segundo valor ');
X(3)=input('Ingrese tercer valor ');
if(X(1)>=X(2) && X(1)>=X(3))
    mayor=X(1)
else
    if(X(2)>=X(1) && X(2)>=X(3))
        mayor=X(2)
    else mayor=X(3)
    end
end

 