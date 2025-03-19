function [dydt] = odefun(t,y,N,dx,pde,lb,rb)
p=y(1:N);
q=y(N+1:end);
dp=zeros(N,1);
dq=zeros(N,1);

    %Left and right conditions
    p(1)=lb(t);
 
    p(end)=rb(t)/((pde.dEdp(p(1)+0.0001,q(1))-pde.dEdp(p(1),q(1)))/0.0001);

    dEdp=pde.dEdp(p,q);   %spacial vector at current time of dE/dp
    dEdq=pde.dEdq(p,q);   %spacial vector at current time of dE/dq

    %Wave PDE
    dp(2:end-1) = 1/2/dx*(dEdq(3:end)-dEdq(1:end-2))+pde.damping*1/dx^2*(dEdp(3:end)-2*dEdp(2:end-1)+dEdp(1:end-2));
    dq(2:end-1) = 1/2/dx*(dEdp(3:end)-dEdp(1:end-2));
    dq(1) = 1/dx*(dEdp(2)-dEdp(1));
    dq(end) = 1/dx*(dEdp(end)-dEdp(end-1));

dydt=[dp;dq];
end

