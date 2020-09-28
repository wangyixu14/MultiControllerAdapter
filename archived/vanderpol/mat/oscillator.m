clear
clc 
tic
% lower bound on the x-grid
x_low= -2.4;
%upper bound on the x-grid
x_upper= 2.4;
% lower bound on the y-grid
y_low=-2.4;
%upper bound on the y-grid
y_upper=2.4;


% lower bound on the d
d_low=-0.15;
%upper bound on the d
d_upper=0.15;




% number of grids on the x-axis
n_x=240;


% number of grids on the y-axis
n_y=240;


% number of grids on the y-axis
n_d=10;
% error tolorence
epsi=10^(-20);




d=linspace(d_low,d_upper,n_d);


x=linspace(x_low,x_upper,n_x);


y=linspace(y_low,y_upper,n_y);


[X,Y]=meshgrid(x,y);


V=zeros(n_x,n_y);


V_update = zeros(1,n_d);
crit = 100;


while crit>epsi
V1 = zeros(n_x,n_y);
for i=1:n_x
  for j=1:n_y
    for k=1:n_d

[x_update(1,k),y_update(1,k)] = dynamics(x(1,i), y(1,j), d(k));
V_update(1,k)=interp2(x,y,V,x_update(1,k),y_update(1,k),'cubic');

end
%[X_update,Y_update]=meshgrid(x_update,y_update);
constraint = max((abs(x(1,i)) - 2)/(1+power(abs(x(1,i))-2, 2)), (abs(y(1,j)) - 2)/(1+power(abs(y(1,j))-2, 2)));
V1(j,i)=max(max(0.01*V_update),constraint);
  end
end

V2=abs(V1-V);
crit = max(V2(:))
V = V1;

end
toc
figure(1)
h1=surf(X,Y,V);
set(h1,'edgecolor','none','facecolor','k');
shading interp
hold on;
contour(X,Y,V,[1e-8 1e-8],'color','r','linewidth',2.0);%

figure(2);
%x1=linspace(x_low,x_upper,n_x*2);
%y1=linspace(y_low,y_upper,n_y*2);
%for i=1:n_x*2
%  for j=1:n_y*2
%V_update1=interp2(x,y,V,x1(1,i),y1(1,i),'linear',0.8);
%  if V_update1<=10^(-9)
%      plot(x1(1,i),y1(1,j),'r*')
%      hold on
%    end
%  end
%end
%for i=1:n_x
%  for j=1:n_y
%  if V(j,i)<=10^(-16)
%      V(j,i)=0;
%    else
%      V(j,i)=0.1;
%    end
%    end
%end

%[a,h]=contourf(X,Y,V,1);
%set(h,'linestyle','none');%È¡�?ûµÈÖµ�?ß�?ÔÊ¾¡£
%hold on;
contour(X,Y,V,[1e-8 1e-8],'color','r','linewidth',2.0);%
hold on
%ezplot('x^2+y^2-0.8',[-1,1])
