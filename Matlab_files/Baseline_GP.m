clear all
close all
%% Define global parameters:
% T changed parameter
L = 1;     %lenght
N = 100;    %segments for discretization
T = 1;       %max time
dt=0.001;  % time step for disc.
train_split = 0.8;
%%

time = readmatrix('time_rightdown.csv');
spatial = readmatrix('spatial_rightdown.csv');
defl = readmatrix('defl_rightdown.csv');

% split training and testing:
train_idx = floor(train_split * length(time));
time = time(1:train_idx);
spatial = spatial(1:train_idx);
defl = defl(1:train_idx);


figure(1)
clf
plot3(time,spatial,defl,'r+');

%% Fit
gprMdl_baseline = fitrgp([time,spatial],defl,"KernelFunction","ardsquaredexponential",Sigma=1,SigmaLowerBound=0.02);
%% Predict
%T reduce data set
[T_test,S_test] = meshgrid(0:0.005:1,0:0.01:1);
sh=size(T_test);
T_test=T_test(:);
S_test=S_test(:);
[defl_pred] = predict(gprMdl_baseline,[T_test,S_test]);

T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
D_pred=reshape(defl_pred,sh);

figure(1)
clf
plot3(time,spatial,defl,'r+');
hold on
surf(T_test,S_test,D_pred)
legend({'data','GP regression'})

%% testing vanilla GP:
time = readmatrix('time_rightdown.csv');
spatial = readmatrix('spatial_rightdown.csv');
defl = readmatrix('defl_rightdown.csv');

time=time(train_idx:end)-time(train_idx);
spatial=spatial(train_idx:end);
defl=defl(train_idx:end);

gprMdl_GT = fitrgp([time,spatial],defl,"KernelFunction","ardsquaredexponential",Sigma=1,SigmaLowerBound=0.02);

%T reduce data set
% T = 1 - train_split; 
[T_test,S_test] = meshgrid(0:0.005:T - train_split,0:0.01:1);
sh=size(T_test);
T_test=T_test(:);
S_test=S_test(:);
[defl_pred] = predict(gprMdl_GT,[T_test,S_test]);

T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
D_pred=reshape(defl_pred,sh);

figure(12)
clf
plot3(time - time(1),spatial,defl,'r+');
hold on
surf(T_test,S_test,D_pred)
legend({'data','GP regression'})

%% test GP model vs real:
T_start = train_split;
[T_test,S_test] = meshgrid(T_start:0.005:T,0:0.01:1);
T_test=T_test(:);
S_test=S_test(:);
[baseline_result] = predict(gprMdl_baseline,[T_test,S_test]);
writematrix(baseline_result, "GP_Baseline_result.csv");
%% Plot
[T_test,S_test] = meshgrid(0:0.005:T - train_split,0:0.01:1);

T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
D_pred=reshape(defl_pred,sh);

figure(123)
clf
hold on

% Plot raw data points as red plus markers
plot3(time-time(1), spatial, defl, 'r+', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Data');

% Plot ground truth surface in blue with high transparency and visible grid ("web cover")
surf(T_test, S_test, D_pred, ...
    'FaceColor', 'blue', ...
    'FaceAlpha', 0.3, ...         % High transparency
    'EdgeColor', 'black', ...       % Web cover edges
    'LineStyle', '-', ...
    'DisplayName', 'GP Regression on GT data');

% Ensure baseline_result is reshaped correctly
baseline_result = reshape(baseline_result, size(T_test));

% Plot GP model prediction surface in red (opaque) with web cover
surf(T_test, S_test, baseline_result, ...
    'FaceColor', 'red', ...
    'FaceAlpha', 0.7, ...           % Fully opaque
    'EdgeColor', 'black', ...       % Web cover edges
    'LineStyle', '-', ...
    'DisplayName', 'GP Baseline');

% Customize axes labels and title
xlabel('Time', 'FontSize', 12);
ylabel('Spatial', 'FontSize', 12);
zlabel('Deflection', 'FontSize', 12);
title('GP Regression vs. Baseline Comparison', 'FontSize', 14);

% Set fixed axis ranges
xlim([0 0.2])
ylim([-0.3 0.3])
zlim([-0.3 0.3])

grid on
axis tight
view(3)  % 3D view
% campos([1.24363772002886, -7.733758743018377, 1.081145588181244]);

% Enhance visualization with lighting
lighting gouraud
camlight headlight

% Add legend
legend('Location', 'best')



%% Plot MSE vs time
fontsize = 18;
max_time = 1181;

shifted_time = train_split * max_time / 100;

% read in the MSE for GP-dPHS:
mse_NN = readmatrix("LSTM_MSE.csv");
mse_GP_dPHS = readmatrix('mse_GP_DPHS.csv');
mse_GP_dPHS_time = linspace(0 + shifted_time, shifted_time + max_time * (1 - train_split) / 100, length(mse_GP_dPHS));

[nSpatial, nTime] = size(D_pred);

% Preallocate an array to store the MSE for each time step.
mse_time = zeros(1, nTime);

% Compute the MSE at each time step (column) across the spatial domain.
for idx = 1:nTime
    mse_time(idx) = mean( (baseline_result(:,idx) - D_pred(:,idx)).^2 );
end

% Extract the time vector from T_test.
% Since T_test was created using meshgrid, each row is identical.
time_vector = shifted_time + T_test(1,:) * max_time / 100;
NN_time = linspace(0 + shifted_time, shifted_time + max_time * (1 - train_split) / 100, length(mse_NN));

% Plot MSE vs Time.
figure;
hold on;
% Plot Baseline GP MSE with a red solid line
plot(time_vector, (mse_time), 'r-', 'LineWidth', 2);

% Plot GP_dPHS MSE with a blue dashed line
plot(mse_GP_dPHS_time, (mse_GP_dPHS), 'b-', 'LineWidth', 2);

plot(NN_time, (mse_NN), 'g-', 'LineWidth',2);

xlabel('Time (s)', 'FontSize', fontsize, 'FontWeight', 'bold');
ylabel('Mean Squared Error (MSE)', 'FontSize', fontsize, 'FontWeight', 'bold');
xlim([shifted_time, max(NN_time)]);

% title('MSE vs Time: Baseline GP vs Our Method', 'FontSize', fontsize);
lgd = legend({'Vanilla GP','PnP-PIML (Ours)', 'Neural Network'}, 'Location', 'Best');
set(gca, 'FontSize', 16);
lgd.FontSize = 16;
grid on;
hold off;
exportgraphics(gcf, 'MSE_vs_Time.pdf', 'ContentType', 'vector');


