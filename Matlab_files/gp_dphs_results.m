clear all
close all
%% Define global parameters:
%T changed parameter
L = 1;     %lenght
N = 100;    %segments for discretization
T = 1;       %max time
dt=0.001;  % time step for disc.
train_split = 0.8;
% data processing Training set
% load and process from mat
load("../video/Data/rightdown/aligned_center_points.mat");
time=double(z)';
spatial=double(x)';
defl=double(y)';
% Normalize
time=time/max(time);
spatial=0.97*spatial/max(spatial);
defl=defl/max(defl);
% T down sampling
% T center Y around zero
% T normalize time

vector_sub=10200:5:30000;
time=time(vector_sub);
time=time-min(time);
time=time/max(time);
spatial=spatial(vector_sub);
defl=defl(vector_sub);

xyz=AxelRot([time,spatial,defl]',-1, [1,0,0],[0,0,0])';

time=xyz(:,1);
spatial=xyz(:,2);
defl=-(xyz(:,3)-0.52);


% complete data
writematrix(time, 'time_rightdown.csv');
writematrix(spatial, 'spatial_rightdown.csv');
writematrix(defl, 'defl_rightdown.csv');


% split training and testing:
train_idx = floor(train_split * length(time));
time = time(1:train_idx);
spatial = spatial(1:train_idx);
defl = defl(1:train_idx);


% % these are for right down, test in left down
% time = readmatrix('../video/time.csv');
% spatial = readmatrix('../video/spatial.csv');
% defl = readmatrix('../video/defl.csv');

figure(1)
clf
plot3(time,spatial,defl,'r+');

%% Fit
gprMdl1 = fitrgp([time,spatial],defl,"KernelFunction","ardsquaredexponential",Sigma=1,SigmaLowerBound=0.02);

%% Predict
%T reduce data set
[T_test,S_test] = meshgrid(0:0.005:1,0:0.01:1);
sh=size(T_test);
T_test=T_test(:);
S_test=S_test(:);
[defl_pred] = predict(gprMdl1,[T_test,S_test]);

T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
D_pred=reshape(defl_pred,sh);

figure(1)
clf
plot3(time,spatial,defl,'r+');
hold on
surf(T_test,S_test,D_pred)
legend({'data','GP regression'})

%% Numerical differentiation using gradient
[dz_dt, dz_ds] = gradient(D_pred, T_test(1, :), S_test(:, 1));
% [defl_pred, ~, gradients] = predict(gprMdl1, [T_test, S_test], 'IncludeInteractions', true);
% Extract partial derivatives:
% Gradient(:, 1) is the derivative w.r.t. time (t)
% dz_dt = reshape(gradients(:, 1), sh);  % Partial derivative w.r.t. t (time)

% Gradient(:, 2) is the derivative w.r.t. spatial (x)
% dz_dx = reshape(gradients(:, 2), sh);  % Partial derivative w.r.t. x (spatial)

% Plot numerical derivatives
figure(2)
clf
subplot(1, 2, 1);
surf(T_test(1:1:end,1:1:end), S_test(1:1:end,1:1:end), dz_dt(1:1:end,1:1:end), 'EdgeColor', 'none');
xlabel('Time');
ylabel('Spatial');
zlabel('d(Deflection)/d(Time)');
title('Partial Derivative w.r.t. Time');
% 
subplot(1, 2, 2);
surf(T_test(1:1:end,1:1:end), S_test(1:1:end,1:1:end), dz_ds(1:1:end,1:1:end), 'EdgeColor', 'none');
xlabel('Time');
ylabel('Spatial');
zlabel('d(Deflection)/d(Spatial)');
title('Partial Derivative w.r.t. Spatial');

%%
%T Why not just taking a subset of dz_dt here?
% Define N equidistant points along the width of the string (x variable)
eq_points = linspace(0, max(spatial), N);

% Predictions from dy/dt
dy_dt_predictions = zeros(length(T_test(1, :)), length(eq_points));
% Compute dy/dt at each spatial point in eq_points
for i = 1:length(eq_points)
    dy_dt_predictions(:, i) = transpose(interp2(T_test, S_test, dz_dt, T_test(1, :), eq_points(i), 'cubic'));
end

% For dy/dz, evaluate for each timeframe along the width (spatial) axis
dy_dz_predictions = zeros(length(S_test(1,:)), length(eq_points));
for i = 1:length(T_test(1,:))
    dy_dz_predictions(i, :) = interp2(T_test, S_test, dz_ds, T_test(1, i) * ones(size(eq_points)), eq_points, 'cubic');
end

% save
writematrix(dy_dt_predictions, 'dydt.csv');
writematrix(dy_dz_predictions, 'dydz.csv');

%% MODEL
% Get data
p_data=transpose(dy_dt_predictions);
q_data=transpose(dy_dz_predictions);
t_span = T_test(1, :);
t=t_span;
dp_data = zeros(size(p_data));
dq_data = zeros(size(q_data));
for i = 1:size(p_data,1)  % Loop over rows
    dp_data(i, :) = gradient(p_data(i, :), t_span);
    dq_data(i, :) = gradient(q_data(i, :), t_span);
end

%% Integrate and add constant
%T correct dx
dx=eq_points(2)-eq_points(1);
dp_int = cumtrapz(dp_data)*dx;
dq_int = cumtrapz(dq_data)*dx;
for i=1:length(t)
    [~,closestIndex] = min(abs(q_data(:,i)));
    dp_int(:,i)=dp_int(:,i)-dp_int(closestIndex,i);
end
%% Plot actual dEdq / dEdp and data points
[Z_mesh,T_mesh]=meshgrid(1:length(eq_points),1:length(T_test(1, :)));
z_vec=Z_mesh(:);
t_vec=T_mesh(:);
model_dEdp=zeros(length(z_vec),1);
model_dEdq=zeros(length(z_vec),1);
p_in=zeros(length(z_vec),1);
q_in=zeros(length(z_vec),1);
%
for i=1:length(z_vec)
    p_in(i)=p_data(z_vec(i),t_vec(i));
    q_in(i)=q_data(z_vec(i),t_vec(i));
    %
    model_dEdp(i)=dq_int(z_vec(i),t_vec(i));
    %
    model_dEdq(i)=dp_int(z_vec(i),t_vec(i));
end
% plot
figure(3)
clf
subplot(1,2,1);
plot3(p_in,q_in,model_dEdp,'+');
title('dHdp');
xlabel('p');
ylabel('q');

figure(3)
subplot(1,2,2);
plot3(p_in,q_in,model_dEdq,'+');
title('dHdq');
xlabel('p');
ylabel('q');

%% Train GP
downscale=20;
gpphs.dEdp_model=fitrgp([p_in(1:downscale:end),q_in(1:downscale:end)],model_dEdp(1:downscale:end),'KernelFunction','squaredexponential','FitMethod', ...
    'sr','PredictMethod','sd','BasisFunction','none');
compact_dEdp = compact(gpphs.dEdp_model);
disp('done')
gpphs.dEdq_model=fitrgp([p_in(1:downscale:end), q_in(1:downscale:end)],model_dEdq(1:downscale:end),'KernelFunction','squaredexponential','FitMethod', ...
    'Exact','BasisFunction','none');
disp('done')
compact_dEdq = compact(gpphs.dEdq_model);

gpphs.dEdp = @(p,q) predict(gpphs.dEdp_model,[p,q]);
gpphs.dEdq = @(p,q) predict(gpphs.dEdq_model,[p,q]);

%% Plot model vs actual
% [P_mesh,Q_mesh]=meshgrid(-1:0.1:1,-1:0.1:1);
[P_mesh,Q_mesh]=meshgrid(-15:0.1:10,-1:0.1:1);
P_vec=P_mesh(:);
Q_vec=Q_mesh(:);
gpdEdp_pred = gpphs.dEdp(P_vec,Q_vec);
gpdEdq_pred = gpphs.dEdq(P_vec,Q_vec);
[dppred,~,dpci] = predict(compact_dEdp,[P_vec,Q_vec]);
[dqpred,~,dqci] = predict(compact_dEdq,[P_vec,Q_vec]);
%%
data=[p_in(1:downscale:end),q_in(1:downscale:end),model_dEdp(1:downscale:end)];
n_sample=1;
samples_dp=create_sample_object(gpphs.dEdp_model,data,1000,n_sample);
dp_sample=simple_sample_fcn([P_vec,Q_vec],samples_dp);

figure(1001)
clf
surf(P_mesh,Q_mesh,reshape(gpdEdp_pred,size(P_mesh)),'FaceColor','r','FaceAlpha',0.5);
hold on
surf(P_mesh,Q_mesh,reshape(dp_sample,size(P_mesh)),'FaceColor','g','FaceAlpha',0.5);
hold on
plot3(p_in,q_in,model_dEdp,'g+');
% surf(P_mesh,Q_mesh,reshape(pde.dEdp(P_vec,Q_vec),size(P_mesh)),'FaceColor','b','FaceAlpha',0.7);
legend(["GP-DPHS model","GP-DPHS model", "Observation data", "True system"]);
% colorbar; % Adds a color bar to indicate height levels
zlabel('dHdp');
title('3D Slices of dHdp Visualization');
view(3); % Sets the view to 3D perspective
view(70, 20);
hold off; % Release the plot hold
%%
data=[p_in(1:downscale:end),q_in(1:downscale:end),model_dEdq(1:downscale:end)];
n_sample=1;
samples_dq=create_sample_object(gpphs.dEdq_model,data,1000,n_sample);
dq_sample=simple_sample_fcn([P_vec,Q_vec],samples_dq);

figure(1002)
clf
% should be mean pred?
surf(P_mesh,Q_mesh,reshape(gpdEdq_pred,size(P_mesh)),'FaceColor','r','FaceAlpha',0.3);
hold on
% looks like samples of GP
surf(P_mesh,Q_mesh,reshape(dq_sample,size(P_mesh)),'FaceColor','g','FaceAlpha',0.5);
hold on
plot3(p_in,q_in,model_dEdq,'g+');
% surf(P_mesh,Q_mesh,reshape(pde.dEdq(P_vec,Q_vec),size(P_mesh)),'FaceColor','b','FaceAlpha',0.7);
legend(["GP-DPHS model","GP-DPHS model","Observation data", "True system"]);
% colorbar; % Adds a color bar to indicate height levels
xlabel('state p', 'FontSize', 12);
ylabel('state q', 'FontSize', 12);
zlabel('derivative of Hamiltonian \delta_q H', 'FontSize', 14);
% title('3D  Derivative of Hamiltonian to Q Visualization', 'FontSize', 12);
view(3); % Sets the view to 3D perspective
view(70, 20);
hold off; % Release the plot hold
%% Hamiltonian
Ham = cumsum(reshape(gpdEdp_pred,size(P_mesh)),2)+cumsum(reshape(gpdEdq_pred,size(P_mesh)),1);
Ham=Ham-min(min(Ham));
Ham=Ham./max(max(Ham));
figure(100111)
clf
surf(P_mesh,Q_mesh,Ham,'FaceAlpha',0.5);
axis([-8 8 -1 1 0 0.8])
xlabel("q","FontSize",20);
ylabel("p","FontSize",20);
zlabel("H(q,p)","FontSize",20);
set(gcf,'Color','white')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training done
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict
%T reduce data set

time = readmatrix('time_rightdown.csv');
spatial = readmatrix('spatial_rightdown.csv');
defl = readmatrix('defl_rightdown.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comment this line for prediction from training to testing data (0-1)
% Leave this line for prediction from testing data (0.8-1)
%
T=1 - train_split; 

time=time(train_idx:end) - time(train_idx);
spatial=spatial(train_idx:end);
defl=defl(train_idx:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fit
gprMdl1 = fitrgp([time,spatial],defl,"KernelFunction","ardsquaredexponential",Sigma=1,SigmaLowerBound=0.02);

[T_test,S_test] = meshgrid(0:0.005:T,0:0.01:1);
sh=size(T_test);
T_test=T_test(:);
S_test=S_test(:);
[defl_pred] = predict(gprMdl1,[T_test,S_test]);

T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
D_pred=reshape(defl_pred,sh);

figure(12)
clf
plot3(time,spatial,defl,'r+');
hold on
surf(T_test,S_test,D_pred)
legend({'data','GP regression'})

%% Run GP model vs real
xmesh = linspace(0,L,N);
gpphs.damping=0.07;
%T fit intial for simulation to experiment
grad_imp=gradient(D_pred(1:end-1,1),dx);
grad_imp(2:end/2)=grad_imp(2:end/2)+0.02;
grad_imp(end-10:end)=grad_imp(end-10:end);
grad_imp(1)=0;
figure(34253)
clf
plot(xmesh,cumsum(grad_imp)*dx);
hold on
plot(xmesh,D_pred(1:end-1,1));

%%
xt0 = [zeros(N,1),grad_imp];
%xt0 = [zeros(N,1),max(defl)*pi/L*cos(pi/L*xmesh')];
% Dirichet Boundary conditions
lbf =@(t) 0;
rbf =@(t) 0;
% 

[t,x_temp]=ode45(@(t,y) odefun(t,y,N,dx,gpphs,lbf,rbf),0:dt:T,xt0(:));
x=zeros(N,length(t),2);
x_gp=[];
x_gp(:,:,1)=x_temp(:,1:N)';
x_gp(:,:,2)=x_temp(:,N+1:end)';
disp('done gp')

pos_gp=cumsum(x_gp(:,:,2))*dx;

%% Plot
[T_sim,X_sim] = meshgrid(t,xmesh);

writematrix(pos_gp, "GP_dPHS_result.csv");
writematrix(T_sim, "GP_dPHS_timeAxis.csv");
writematrix(X_sim, "GP_dPHS_spatialAxis.csv");


figure(123)
clf
plot3(time,spatial,defl,'r+');
hold on
surf(T_test,S_test,D_pred,'FaceAlpha',0.5)
legend({'data','GP regression','PHS model'})

surf(T_sim,X_sim,pos_gp,'FaceAlpha',0.5);
xlabel('time');
ylabel('spatial');

%% Plot MSE vs time

mse_time = zeros(length(t),1);

% Loop over simulation time steps
for i = 1:length(t)
    % Create predictors for the GP model: the current time (t(i)) repeated
    % for each spatial location in xmesh. (Assuming the GP model was trained 
    % using [time, spatial] as predictors.)
    X_pred = [t(i)*ones(length(xmesh),1), xmesh(:)];
    
    % Get the ground truth (or GP predicted) deflection along the spatial grid
    gt_defl = predict(gprMdl1, X_pred);
    
    % Compute the mean squared error between the simulation output and the GP prediction.
    % Here pos_gp(:,i) is the simulated deflection at time t(i).
    mse_time(i) = mean((pos_gp(:,i) - gt_defl).^2);
end

% Optionally, plot the MSE vs. time
writematrix(mse_time, 'mse_GP_DPHS.csv');

figure;
plot(t, mse_time, 'LineWidth', 2);
xlabel('Time');
ylabel('MSE');
title('Time vs. MSE between Model and Ground Truth');
grid on;


