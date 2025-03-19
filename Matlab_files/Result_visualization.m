%% read in results and global params:
L = 1;     %lenght
N = 100;    %segments for discretization
T = 1;       %max time
dt=0.001;  % time step for disc.
train_split = 0.8;
zmin = -0.4;
zmax = 0.3;

% map back to original coordinates:
max_defl = 1318;
max_time = 1181;
max_spatial = 1919;

time_data = readmatrix('time_rightdown.csv');
spatial_data = readmatrix('spatial_rightdown.csv');
defl_data = readmatrix('defl_rightdown.csv');

train_idx = floor(train_split * length(time_data));


time=time_data(train_idx:end) - time_data(train_idx);
spatial=spatial_data(train_idx:end);
defl=defl_data(train_idx:end);

time_train=time_data(1:train_idx);
spatial_train=spatial_data(1:train_idx);
defl_train=defl_data(1:train_idx);


gprMdl_global = fitrgp([time_data,spatial_data],defl_data,"KernelFunction","ardsquaredexponential",Sigma=1,SigmaLowerBound=0.02);
gprMdl_GT = fitrgp([time,spatial],defl,"KernelFunction","ardsquaredexponential",Sigma=1,SigmaLowerBound=0.02);



%% Plot vanilla GP Baseline:
% uncomment for use

% T = T - train_split;
% [T_test,S_test] = meshgrid(0:0.005:T,0:0.01:1);
% sh=size(T_test);
% 
% T_test=T_test(:);
% S_test=S_test(:);
% [defl_pred] = predict(gprMdl_GT,[T_test,S_test]);
% 
% 
% T_test=reshape(T_test,sh);
% S_test=reshape(S_test,sh);
% D_pred=reshape(defl_pred,sh);
% 
% 
% 
% baseline_result = readmatrix("GP_Baseline_result.csv");
% 
% figure(1)
% clf
% hold on
% 
% % Plot raw data points as red plus markers
% plot3(time, spatial, defl, 'g+', 'MarkerSize', 3, 'LineWidth', 0.7, 'DisplayName', 'Data');
% 
% % Plot ground truth surface in blue with high transparency and visible grid ("web cover")
% surf(T_test, S_test, D_pred, ...
%     'FaceColor', 'blue', ...
%     'FaceAlpha', 0.5, ...         % High transparency
%     'EdgeColor', 'black', ...       % Web cover edges
%     'LineStyle', '-', ...
%     'DisplayName', 'GP Regression on GT data');
% 
% % Ensure baseline_result is reshaped correctly
% baseline_result = reshape(baseline_result, size(T_test));
% 
% % Plot GP model prediction surface in red (opaque) with web cover
% surf(T_test, S_test, baseline_result, ...
%     'FaceColor', 'red', ...
%     'FaceAlpha', 0.8, ...           % Fully opaque
%     'EdgeColor', 'black', ...       % Web cover edges
%     'LineStyle', '-', ...
%     'DisplayName', 'GP Baseline');
% 
% % Customize axes labels and title
% xlabel('Time (t)', 'FontSize', fontsize, 'FontWeight', 'bold');
% ylabel('Spatial (z)', 'FontSize', fontsize, 'FontWeight', 'bold');
% zlabel('Deflection (s)', 'FontSize', fontsize, 'FontWeight', 'bold');
% % title('GP Regression vs. Baseline Comparison', 'FontSize', 16, 'FontWeight', 'bold');
% 
% % Set fixed axis ranges
% xlim([0 0.2])
% ylim([0 1])
% zlim([zmin zmax])
% % axis manual;
% 
% 
% grid on
% % axis tight
% view(3)  % 3D view
% 
% % Set custom camera parameters
% campos([0.289403013612007, -8.078223274184722, 0.429934340274663]);
% camva(7.190042297051135);
% view(6.299999630701684, 4.763211779162734);  % view(azimuth, elevation)
% % Enhance visualization with lighting
% lighting gouraud
% camlight headlight
% 
% % Add legend
% % legend('Location', 'best')
% lgd = legend('Observed Data','Ground Truth','Vanilla GP', 'Location', 'best');
% lgd.FontSize = 16;
% 
% exportgraphics(gcf, 'GP_results.eps', 'ContentType', 'vector');
% 


%% plot GP dPHS
fontsize = 18;

[T_test,S_test] = meshgrid(0:0.005:T,0:0.01:1);
sh=size(T_test);
T_test=T_test(:);
S_test=S_test(:);
[defl_pred] = predict(gprMdl_global,[T_test,S_test]);


T_test=reshape(T_test,sh);
S_test=reshape(S_test,sh);
GT_gp_result=reshape(defl_pred,sh);

T_sim = readmatrix("GP_dPHS_timeAxis.csv");
X_sim = readmatrix("GP_dPHS_spatialAxis.csv");
pos_gpdphs = readmatrix("GP_dPHS_result.csv");
figure(2)
clf
hold on

% Plot raw data points as red plus markers
plot3(time_train * max_time / 100, spatial_train * max_spatial / 10000, defl_train * max_defl / 100, 'g+', 'MarkerSize', 3, 'LineWidth', 0.7, 'DisplayName', 'Data');


% Draw a vertical plane at 80% of the time axis.
% Compute the x-coordinate for the plane:
x_plane = 0.8 * max_time / 100;  % 80% of your scaled time axis

% Get current y (spatial) and z (deflection) limits from the axes:
y_limits = get(gca, 'YLim');
z_limits = get(gca, 'ZLim');

% Create a grid for the plane:
[Y_plane, Z_plane] = meshgrid(linspace(y_limits(1), y_limits(2), 20), linspace(z_limits(1), z_limits(2), 20));
X_plane = x_plane * ones(size(Y_plane));

% Plot the vertical plane with some transparency
surf(X_plane, Y_plane, Z_plane, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', [0.8, 0.8, 0.8]);

% Plot ground truth surface using a custom blue and remove black edges
surf(T_test * max_time / 100, S_test * max_spatial / 10000, GT_gp_result * max_defl / 100, ...
    'FaceColor', [0.3 0.6 1], ...  % custom blue
    'FaceAlpha', 0.6, ...         % moderate transparency
    'EdgeColor', 'none', ...      % no edges
    'LineStyle', '-', ...
    'DisplayName', 'GP Regression on GT data');

% Plot GP model prediction surface using a custom red and remove black edges
surf(T_sim * max_time / 100, X_sim * max_spatial / 10000, pos_gpdphs * max_defl / 100, ...
    'FaceColor', [1 0.3 0.3], ...  % custom red
    'FaceAlpha', 0.8, ...          % less transparent (opaque)
    'EdgeColor', 'none', ...       % no edges
    'LineStyle', '-', ...
    'DisplayName', 'GP Baseline');

% Customize axes labels and title
xlabel('Time (s)', 'FontSize', fontsize, 'FontWeight', 'bold');
xtickformat('%.1f');
xtickangle(15);


ylabel('Spatial (m)', 'FontSize', fontsize, 'FontWeight', 'bold');
zlabel('Deflection (cm)', 'FontSize', fontsize, 'FontWeight', 'bold');
% title('GP Regression vs. Baseline Comparison', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 16);
% Ideal axes dimensions based on pbaspect [2 1.3 1.3]
idealWidth = 800;
idealHeight = idealWidth / (2/1.3);  % ~520 pixels

% Add a small margin (50 pixels on left/right and top/bottom)
margin = 40;
figureWidth = idealWidth + 2 * margin;
figureHeight = idealHeight + 2 * margin;

% Set the figure size in pixels
set(gcf, 'Units', 'pixels', 'Position', [100, 100, figureWidth, figureHeight]);

% Set the axes to fill most of the figure (using normalized units)
set(gca, 'Units', 'normalized', 'Position', [margin/figureWidth, margin/figureHeight, idealWidth/figureWidth, idealHeight/figureHeight]);

% Now set the desired aspect ratio
pbaspect([2 1.3 1.3]);

grid on
% axis tight
view(3)  % 3D view

% Set custom camera parameters
% campos([0.289403013612007, -8.078223274184722, 0.429934340274663]);
% camva(7.190042297051135);
% view(6.299999630701684, 4.763211779162734);  % view(azimuth, elevation)
% Set custom camera parameters similar to before
campos([0.32, -8.05, 0.45]);    % slightly shift the camera position
camva(7.0);                    % narrow the field-of-view just a bit
view(7, 5.5);                  % adjust azimuth and elevation slightly


% Enhance visualization with lighting
lighting gouraud
camlight headlight

% Add legend
lgd = legend('Observed Data','Train/Test Split','Ground Truth','PnP-PIML (Ours)', 'Location', 'best');
lgd.FontSize = 16;
% exportgraphics(gcf, 'GPdPHS_result.eps', 'ContentType', 'vector');




%% Plot NN result:
% uncomment for use
% 
% NN_time = readmatrix("NN_adjusted_time_test.csv");
% NN_spatial = readmatrix("NN_spatial_test.csv");
% NN_result = readmatrix("NN_test_predictions.csv");
% 
% % Example scattered data vectors
% 
% % Create a grid for interpolation
% xi = linspace(min(NN_time), max(NN_time), 100);
% yi = linspace(min(NN_spatial), max(NN_spatial), 100);
% [NN_time_test, NN_spatial_test] = meshgrid(xi, yi);
% 
% % Interpolate scattered data onto the grid
% NN_result = griddata(NN_time, NN_spatial, NN_result, NN_time_test, NN_spatial_test, 'cubic');
% 
% figure(3)
% clf
% hold on
% 
% % Plot raw data points as red plus markers
% plot3(time, spatial, defl, 'g+', 'MarkerSize', 3, 'LineWidth', 0.7, 'DisplayName', 'Data');
% 
% % Plot ground truth surface in blue with high transparency and visible grid ("web cover")
% surf(T_test, S_test, D_pred, ...
%     'FaceColor', 'blue', ...
%     'FaceAlpha', 0.5, ...         % High transparency
%     'EdgeColor', 'black', ...       % Web cover edges
%     'LineStyle', '-', ...
%     'DisplayName', 'GP Regression on GT data');
% 
% % Ensure baseline_result is reshaped correctly
% baseline_result = reshape(baseline_result, size(T_test));
% 
% % Plot GP model prediction surface in red (opaque) with web cover
% surf(NN_time_test, NN_spatial_test, NN_result, ...
%     'FaceColor', 'red', ...
%     'FaceAlpha', 0.8, ...           % Fully opaque
%     'EdgeColor', 'black', ...       % Web cover edges
%     'LineStyle', '-', ...
%     'DisplayName', 'GP Baseline');
% 
% % Customize axes labels and title
% xlabel('Time (t)', 'FontSize', fontsize, 'FontWeight', 'bold');
% ylabel('Spatial (z)', 'FontSize', fontsize, 'FontWeight', 'bold');
% zlabel('Deflection (s)', 'FontSize', fontsize, 'FontWeight', 'bold');
% % title('GP Regression vs. Baseline Comparison', 'FontSize', 16, 'FontWeight', 'bold');
% 
% % Set fixed axis ranges
% xlim([0 0.2])
% ylim([0 1])
% zlim([zmin zmax])
% % axis manual;
% 
% 
% grid on
% % axis tight
% view(3)  % 3D view
% 
% % Set custom camera parameters
% campos([0.289403013612007, -8.078223274184722, 0.429934340274663]);
% camva(7.190042297051135);
% view(6.299999630701684, 4.763211779162734);  % view(azimuth, elevation)
% % Enhance visualization with lighting
% lighting gouraud
% camlight headlight
% 
% % Add legend
% lgd = legend('Observed Data','Ground Truth','Neural Network', 'Location', 'best');
% lgd.FontSize = 16;
% exportgraphics(gcf, 'NN_result.eps', 'ContentType', 'vector');
% 
