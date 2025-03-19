%%
rng(2);
n_sample=3;
data=[p_in(1:downscale:end),q_in(1:downscale:end),model_dEdp(1:downscale:end)];
samples_dp=create_sample_object(gpphs.dEdp_model,data,1000,n_sample);
rng(3);
data=[p_in(1:downscale:end),q_in(1:downscale:end),model_dEdq(1:downscale:end)];
samples_dq=create_sample_object(gpphs.dEdq_model,data,1000,n_sample);
%%
N=100;
T=1; 
xmesh = linspace(0,L,N);
gpphs.damping=0.07;
%T fit intial for simulation to experiment
xt0 = [zeros(N,1),grad_imp];
% Dirichet Boundary conditions
lbf =@(t) 0;
rbf =@(t) 0;
% 
clear pos_sample
for isamples=1:n_sample
disp(isamples) 
gpphs.dEdp = @(p,q) simple_sample_fcn([p,q],samples_dp(:,isamples));
gpphs.dEdq = @(p,q) simple_sample_fcn([p,q],samples_dq(:,isamples));
%gpphs.dEdp = @(p,q) predict(gpphs.dEdp_model,[p,q]);
%gpphs.dEdq = @(p,q) predict(gpphs.dEdq_model,[p,q]);
tic
[t,x_temp]=ode45(@(t,y) odefun(t,y,N,dx,gpphs,lbf,rbf),0:dt:T,xt0(:));
toc
x_gp=[];
x_gp(:,:,1)=x_temp(:,1:N)';
x_gp(:,:,2)=x_temp(:,N+1:end)';

pos_sample{isamples}=cumsum(x_gp(:,:,2))*dx;
end

%add mean
gpphs.dEdp = @(p,q) predict(gpphs.dEdp_model,[p,q]);
gpphs.dEdq = @(p,q) predict(gpphs.dEdq_model,[p,q]);
tic
[t,x_temp]=ode45(@(t,y) odefun(t,y,N,dx,gpphs,lbf,rbf),0:dt:T,xt0(:));
toc
x_gp=[];
x_gp(:,:,1)=x_temp(:,1:N)';
x_gp(:,:,2)=x_temp(:,N+1:end)';

pos_sample{isamples+1}=cumsum(x_gp(:,:,2))*dx;
%% GRound truth
[T_test,S_test] = meshgrid(0:dt:T,xmesh);
sh=size(T_test);
T_test=T_test(:);
S_test=S_test(:);
[defl_pred] = predict(gprMdl1,[T_test,S_test]);
T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
D_pred=reshape(defl_pred,sh);
%% mean only
v = VideoWriter("mean.mp4",'MPEG-4');
v.FrameRate=30;
open(v)

f=figure(999);
clf
handle_line0 = plot(xmesh,D_pred(:,1),'LineWidth',2,'Color','blue','Visible','off');
hold on
for ij=1:length(pos_sample)-1
    handle_line{ij} = plot(xmesh,pos_sample{ij}(:,1),'LineWidth',2,'Color',[0.5,0.5,0.5],'Visible','off');
end
ij=ij+1;
handle_line{ij} = plot(xmesh,pos_sample{ij}(:,1),'LineWidth',6,'Color',"blue");
%hold on;
%handle_dot = plot(xmesh(end/2),pos(end/2,1),'o',MarkerSize=10);

for ii=1:length(D_pred)
    handle_line0.YData = D_pred(:,ii);
    for ij=1:length(pos_sample)
        handle_line{ij}.YData = pos_sample{ij}(:,ii);
    end
axis([0,max(xmesh),-0.4,0.4]);
set(f,'Position',[0.1300 0.1100 0.5 0.4],'Units','normalized');
h=gca;
h.XAxis.Visible = 'off';
h.YAxis.Visible = 'off';
drawnow

writeVideo(v,getframe)
end
close(v)
%% samples only
v = VideoWriter("samples.mp4",'MPEG-4');
v.FrameRate=30;
open(v)

f=figure(999);
clf
handle_line0 = plot(xmesh,D_pred(:,1),'LineWidth',2,'Color','blue','Visible','off');
hold on
for ij=1:length(pos_sample)-1
    handle_line{ij} = plot(xmesh,pos_sample{ij}(:,1),'LineWidth',6,'Color',[0.5,0.5,0.5]);
end
ij=ij+1;
handle_line{ij} = plot(xmesh,pos_sample{ij}(:,1),'LineWidth',6,'Color',"blue",'Visible','off');
%hold on;
%handle_dot = plot(xmesh(end/2),pos(end/2,1),'o',MarkerSize=10);

for ii=1:length(D_pred)
    handle_line0.YData = D_pred(:,ii);
    for ij=1:length(pos_sample)
        handle_line{ij}.YData = pos_sample{ij}(:,ii);
    end
axis([0,max(xmesh),-0.4,0.4]);
set(f,'Position',[0.1300 0.1100 0.5 0.4],'Units','normalized');
h=gca;
h.XAxis.Visible = 'off';
h.YAxis.Visible = 'off';
drawnow

writeVideo(v,getframe)
end
close(v)