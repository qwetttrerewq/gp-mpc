%--------------------------------------------------------------------------
% Pre-train GP Model for Single Track Vehicle Dynamics
%
% Collects training data by simulating true (Pacejka) and nominal (linear)
% models across diverse operating conditions, then trains GP on dynamics
% residuals with properly optimized hyperparameters.
%
% State:  x = [I_x; I_y; psi; V_vx; V_vy; psi_dot] (6D)
% Input:  u = [delta; T] (2D)
% GP in:  z = [V_vx; V_vy; psi_dot; delta; T] (5D)
% GP out: d = [ΔV_vx_dot; ΔV_vy_dot; Δpsi_dot_dot] (3D)
%
% Saves: simresults/pretrained_GP_singletrack_<timestamp>.mat
%   - Contains variable 'd_GP' (GP object)
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))

fprintf('\n========================================\n');
fprintf('GP Pre-training for Single Track Dynamics\n');
fprintf('========================================\n\n');

%% ========================================================================
%  CONFIGURATION
%  ========================================================================

% GP dictionary size
GP_Nmax = 300;

% Data collection parameters
dt = 0.1;              % timestep [s]
tf_total = 10;          % total duration [s]

% Hyperparameter optimization
optimize_hyper = true;
opt_method = 'fmincon';

fprintf('Configuration:\n');
fprintf('  Dictionary size: %d\n', GP_Nmax);
fprintf('  Total duration: %.1f s\n', tf_total);
fprintf('  Timestep: %.2f s\n', dt);
fprintf('  Optimize hyperparameters: %s\n', string(optimize_hyper));
fprintf('  Optimization method: %s\n\n', opt_method);

%% ========================================================================
%  CREATE MODELS
%  ========================================================================

% Process noise (small for clean training)
var_w = [];

% True model (Pacejka tires)
trueModel = MotionModelGP_true([], var_w);

% Nominal model (Linear tires)
nomModel = MotionModelGP_nominal([], []);

fprintf('Models created:\n');
fprintf('  True: Pacejka tires (Df=%.0f N, Dr=%.0f N)\n', trueModel.D_f, trueModel.D_r);
fprintf('  Nominal: Linear tires (Cf=%.0f N/rad, Cr=%.0f N/rad)\n', nomModel.c_f, nomModel.c_r);
fprintf('  State: [I_x; I_y; psi; V_vx; V_vy; psi_dot] (6D)\n');
fprintf('  Input: [delta; T] (2D)\n');
fprintf('  GP input z: [V_vx; V_vy; psi_dot; delta; T] (5D)\n');
fprintf('  GP output d: [ΔV_vx_dot; ΔV_vy_dot; Δpsi_dot_dot] (3D)\n\n');

%% ========================================================================
%  COLLECT TRAINING DATA
%  ========================================================================

fprintf('Collecting training data...\n');

% Define comprehensive scenarios
% [v_ref, radius, duration_fraction, description]
scenarios = {
    % Low speed scenarios
    10, 20, 0.10, 'Very low speed, very tight turn'
    10, 40, 0.10, 'Very low speed, medium turn'
    10, 60, 0.08, 'Very low speed, gentle turn'
    
    % Medium-low speed
    12, 25, 0.10, 'Low-medium speed, tight turn'
    12, 35, 0.10, 'Low-medium speed, medium turn'
    12, 50, 0.08, 'Low-medium speed, gentle turn'
    
    % % Medium speed
    % 15, 25, 0.10, 'Medium speed, tight turn'
    % 15, 35, 0.10, 'Medium speed, medium-tight turn'
    % 15, 50, 0.12, 'CRITICAL: Medium speed, medium turn (TEST CONDITION)'
    % 15, 70, 0.08, 'Medium speed, gentle turn'
    };

n = nomModel.n;  % 6
m = nomModel.m;  % 2

all_z = [];  % GP inputs [5 x N]
all_d = [];  % GP outputs (dynamics residuals) [N x 3]

for s = 1:size(scenarios, 1)
    v_ref = scenarios{s, 1};
    radius = scenarios{s, 2};
    duration_frac = scenarios{s, 3};
    description = scenarios{s, 4};
    
    scenario_duration = tf_total * duration_frac;
    r_ref = v_ref / radius;
    
    fprintf('  [%d/%d] %s\n', s, size(scenarios,1), description);
    fprintf('        v=%.1f m/s, R=%.1f m, t=%.1fs, r_ref=%.3f rad/s\n', ...
        v_ref, radius, scenario_duration, r_ref);
    
    % Time vector
    t_vec = 0:dt:scenario_duration;
    kmax = length(t_vec) - 1;
    
    % Initial state [I_x; I_y; psi; V_vx; V_vy; psi_dot]
    x0 = [0; 0; 0; v_ref; 0; 0];
    
    % Storage
    x_true = [x0 zeros(n, kmax)];
    x_nom = [x0 zeros(n, kmax)];
    u_sim = zeros(m, kmax);
    
    % Simple proportional controller for yaw rate tracking
    K_r = 0.8;
    K_T = 0.5;  % Torque controller for velocity tracking
    
    for k = 1:kmax
        % Control law
        r_error = r_ref - x_true(6, k);  % yaw rate error
        v_error = v_ref - x_true(4, k);  % velocity error
        
        delta_cmd = K_r * r_error;
        T_cmd = K_T * v_error;
        
        % Saturate inputs
        u_sim(1, k) = max(min(delta_cmd, deg2rad(30)), -deg2rad(30));
        u_sim(2, k) = max(min(T_cmd, 1), -1);
        
        % Propagate true model
        [x_true(:, k+1), ~] = trueModel.xkp1(x_true(:, k), zeros(n,n), u_sim(:, k), dt);
        
        % Propagate nominal model (not needed for tire force residual)
        % [x_nom(:, k+1), ~] = nomModel.xkp1(x_nom(:, k), zeros(n,n), u_sim(:, k), dt);
        
        % Calculate tire force residual: d = [ΔF_y_f; ΔF_y_r; ΔM_z]
        % Get lateral tire forces from both models
        [F_y_f_true, F_y_r_true] = trueModel.getTireForces(x_true(:, k), u_sim(:, k));
        [F_y_f_nom, F_y_r_nom] = nomModel.getTireForces(x_true(:, k), u_sim(:, k));
        
        % Tire force residuals
        delta_F_y_f = F_y_f_true - F_y_f_nom;
        delta_F_y_r = F_y_r_true - F_y_r_nom;
        
        % Calculate moment residual: ΔM_z = l_f*ΔF_y_f - l_r*ΔF_y_r
        % (assuming delta is small, cos(delta)≈1)
        delta_M_z = nomModel.l_f * delta_F_y_f - nomModel.l_r * delta_F_y_r;
        
        % GP output: d = [ΔF_y_f; ΔF_y_r; ΔM_z]
        d_k = [delta_F_y_f; delta_F_y_r; delta_M_z];
        
        % GP input: z = [V_vx; V_vy; psi_dot; delta; T]
        z_k = [nomModel.Bz_x * x_true(:, k);  % [V_vx; V_vy; psi_dot]
            nomModel.Bz_u * u_sim(:, k)];  % [delta; T]
        
        % Store
        all_z = [all_z z_k];
        all_d = [all_d; d_k'];
    end
    
    fprintf('        Collected %d points (avg residual: ΔFyf=%.2f N, ΔFyr=%.2f N, ΔMz=%.2f Nm)\n', ...
        kmax, mean(abs(all_d(end-kmax+1:end, 1))), ...
        mean(abs(all_d(end-kmax+1:end, 2))), ...
        mean(abs(all_d(end-kmax+1:end, 3))));
end

fprintf('\nTotal data: %d points\n', size(all_z, 2));

% Subsample if too many points
max_training_points = 800;
if size(all_z, 2) > max_training_points
    fprintf('Subsampling from %d to %d points...\n', size(all_z, 2), max_training_points);
    idx_sample = randperm(size(all_z, 2), max_training_points);
    all_z = all_z(:, idx_sample);
    all_d = all_d(idx_sample, :);
    fprintf('  Subsampled!\n\n');
end

% Compute data statistics
z_range = max(all_z, [], 2) - min(all_z, [], 2);
d_range = max(all_d) - min(all_d);

fprintf('Data ranges:\n');
fprintf('  V_vx:      [%.3f, %.3f] m/s (range: %.3f)\n', min(all_z(1,:)), max(all_z(1,:)), z_range(1));
fprintf('  V_vy:      [%.3f, %.3f] m/s (range: %.3f)\n', min(all_z(2,:)), max(all_z(2,:)), z_range(2));
fprintf('  psi_dot:   [%.3f, %.3f] rad/s (range: %.3f)\n', min(all_z(3,:)), max(all_z(3,:)), z_range(3));
fprintf('  delta:     [%.3f, %.3f] rad (range: %.3f)\n', min(all_z(4,:)), max(all_z(4,:)), z_range(4));
fprintf('  T:         [%.3f, %.3f] [-] (range: %.3f)\n', min(all_z(5,:)), max(all_z(5,:)), z_range(5));
fprintf('  ΔF_y_f:    [%.2f, %.2f] N (range: %.2f)\n', min(all_d(:,1)), max(all_d(:,1)), d_range(1));
fprintf('  ΔF_y_r:    [%.2f, %.2f] N (range: %.2f)\n', min(all_d(:,2)), max(all_d(:,2)), d_range(2));
fprintf('  ΔM_z:      [%.2f, %.2f] Nm (range: %.2f)\n\n', min(all_d(:,3)), max(all_d(:,3)), d_range(3));

%% ========================================================================
%  INITIALIZE GP
%  ========================================================================

fprintf('Initializing GP with data-informed hyperparameters...\n');

gp_n = nomModel.nz;  % 5D
gp_p = nomModel.nd;  % 3D

% Initial hyperparameters
ell_vx = 2.0;        % V_vx length scale [m/s]
ell_vy = 0.5;        % V_vy length scale [m/s]
ell_r = 0.2;         % psi_dot length scale [rad/s]
ell_delta = 0.15;    % delta length scale [rad]
ell_T = 0.3;         % T length scale [-]

% Signal variance (based on typical tire force residuals)
var_f_Fyf = 2000^2;    % ΔF_y_f signal variance [N²]
var_f_Fyr = 1500^2;    % ΔF_y_r signal variance [N²]
var_f_Mz = 3000^2;     % ΔM_z signal variance [Nm²]

% Noise variance
var_n_Fyf = var_f_Fyf * 0.2;
var_n_Fyr = var_f_Fyr * 0.2;
var_n_Mz = var_f_Mz * 0.2;

fprintf('Initial hyperparameters:\n');
fprintf('  Length scales: [%.2f, %.2f, %.2f, %.2f, %.2f] (vx, vy, r, delta, T)\n', ...
    ell_vx, ell_vy, ell_r, ell_delta, ell_T);
fprintf('  Signal variance: [%.0f, %.0f, %.0f] (ΔFyf, ΔFyr, ΔMz)\n', ...
    var_f_Fyf, var_f_Fyr, var_f_Mz);
fprintf('  Noise variance: [%.0f, %.0f, %.0f]\n\n', var_n_Fyf, var_n_Fyr, var_n_Mz);

% Set hyperparameters
var_f = [var_f_Fyf; var_f_Fyr; var_f_Mz];
var_n = diag([var_n_Fyf, var_n_Fyr, var_n_Mz]);

% Length scale matrix (diagonal ARD kernel for each output)
M = zeros(gp_n, gp_n, gp_p);
M(:,:,1) = diag([ell_vx^2, ell_vy^2, ell_r^2, ell_delta^2, ell_T^2]);  % ΔF_y_f
M(:,:,2) = diag([ell_vx^2, ell_vy^2, ell_r^2, ell_delta^2, ell_T^2]);  % ΔF_y_r
M(:,:,3) = diag([ell_vx^2, ell_vy^2, ell_r^2, ell_delta^2, ell_T^2]);  % ΔM_z

% Create GP
d_GP = GP(gp_n, gp_p, var_f, var_n, M, GP_Nmax);

% Add training data in batches
batch_size = 50;
num_batches = ceil(size(all_z, 2) / batch_size);

fprintf('Adding %d training points to GP in %d batches...\n', size(all_z, 2), num_batches);

for b = 1:num_batches
    idx_start = (b-1)*batch_size + 1;
    idx_end = min(b*batch_size, size(all_z, 2));
    
    d_GP.add(all_z(:, idx_start:idx_end), all_d(idx_start:idx_end, :));
    
    if mod(b, 5) == 0 || b == num_batches
        fprintf('  Batch %d/%d: Dictionary size = %d\n', b, num_batches, size(d_GP.X, 2));
    end
end

fprintf('  Final dictionary size: %d/%d\n\n', size(d_GP.X, 2), GP_Nmax);

%% ========================================================================
%  OPTIMIZE HYPERPARAMETERS
%  ========================================================================

if optimize_hyper
    fprintf('========================================\n');
    fprintf('Optimizing hyperparameters...\n');
    fprintf('========================================\n\n');
    
    try
        d_GP.optimizeHyperParams(opt_method);
        
        fprintf('\nOptimization successful!\n');
        fprintf('Optimized hyperparameters:\n');
        fprintf('  Length scales (ΔF_y_f): [%.4f, %.4f, %.4f, %.4f, %.4f]\n', sqrt(diag(d_GP.M(:,:,1))));
        fprintf('  Length scales (ΔF_y_r): [%.4f, %.4f, %.4f, %.4f, %.4f]\n', sqrt(diag(d_GP.M(:,:,2))));
        fprintf('  Length scales (ΔM_z):  [%.4f, %.4f, %.4f, %.4f, %.4f]\n', sqrt(diag(d_GP.M(:,:,3))));
        fprintf('  Signal variance: [%.4f, %.4f, %.6f]\n', d_GP.var_f);
        fprintf('  Noise variance: [%.4f, %.4f, %.6f]\n\n', diag(d_GP.var_n));
        
    catch ME
        warning('GPMPC:OptimizationFailed', 'Hyperparameter optimization failed: %s', ME.message);
        fprintf('Continuing with initial hyperparameters...\n\n');
    end
end

%% ========================================================================
%  VALIDATE
%  ========================================================================

fprintf('========================================\n');
fprintf('Validating GP predictions...\n');
fprintf('========================================\n\n');

[mu_pred, var_pred] = d_GP.eval(all_z, false);

% Check dimensions
fprintf('Debug: size(all_z) = [%d, %d]\n', size(all_z));
fprintf('Debug: size(all_d) = [%d, %d]\n', size(all_d));
fprintf('Debug: size(mu_pred) = [%d, %d]\n', size(mu_pred));
fprintf('Debug: size(var_pred) = [%d, %d]\n\n', size(var_pred));

% For multi-output GP, var_pred is typically [p x p x N] or just diagonal variances [p x N]
% Check the actual format
if ndims(var_pred) == 3
    % var_pred is [p x p x N] - extract diagonal for each test point
    fprintf('var_pred format: [p x p x N] covariance tensor\n');
    N_test = size(var_pred, 3);
    sigma_all = zeros(3, N_test);
    for i = 1:N_test
        sigma_all(:, i) = sqrt(diag(var_pred(:,:,i)));
    end
elseif size(var_pred, 1) == size(var_pred, 2) && size(var_pred, 1) == nomModel.nd
    % var_pred is [p x p] - single covariance matrix (not per-point)
    fprintf('var_pred format: [p x p] single covariance matrix\n');
    fprintf('WARNING: GP returns single covariance matrix, not per-point variances.\n');
    fprintf('Cannot compute proper coverage statistics.\n\n');
    % Use diagonal as rough estimate
    sigma_all = repmat(sqrt(diag(var_pred)), 1, size(all_z, 2));
else
    % var_pred is [p x N] - individual variances per output per point
    fprintf('var_pred format: [p x N] individual variances\n');
    sigma_all = sqrt(var_pred);
end

% Ensure mu_pred and var_pred are properly sized
error_vx = all_d(:, 1) - mu_pred(1, :)';
error_vy = all_d(:, 2) - mu_pred(2, :)';
error_r = all_d(:, 3) - mu_pred(3, :)';

rmse_vx = sqrt(mean(error_vx.^2));
rmse_vy = sqrt(mean(error_vy.^2));
rmse_r = sqrt(mean(error_r.^2));

fprintf('Training Set Performance:\n');
fprintf('  RMSE (ΔF_y_f): %.2f N\n', rmse_vx);
fprintf('  RMSE (ΔF_y_r): %.2f N\n', rmse_vy);
fprintf('  RMSE (ΔM_z): %.2f Nm\n', rmse_r);

% 2-sigma coverage
sigma_vx = sigma_all(1, :)';  % [N x 1]
sigma_vy = sigma_all(2, :)';  % [N x 1]
sigma_r = sigma_all(3, :)';   % [N x 1]

fprintf('Debug: size(error_vx) = [%d, %d]\n', size(error_vx));
fprintf('Debug: size(sigma_vx) = [%d, %d]\n\n', size(sigma_vx));

% Check if sizes match
if length(error_vx) == length(sigma_vx)
    coverage_vx = mean(abs(error_vx) <= 2*sigma_vx) * 100;
    coverage_vy = mean(abs(error_vy) <= 2*sigma_vy) * 100;
    coverage_r = mean(abs(error_r) <= 2*sigma_r) * 100;
    
    fprintf('  2σ coverage:\n');
    fprintf('    Δvx_dot: %.1f%%\n', coverage_vx);
    fprintf('    Δvy_dot: %.1f%%\n', coverage_vy);
    fprintf('    Δr_dot:  %.1f%%\n\n', coverage_r);
else
    warning('Size mismatch: error_vx has %d elements but sigma_vx has %d elements. Skipping coverage.', ...
        length(error_vx), length(sigma_vx));
    coverage_vx = NaN;
    coverage_vy = NaN;
    coverage_r = NaN;
end

%% ========================================================================
%  SAVE
%  ========================================================================

if ~exist('simresults', 'dir')
    mkdir('simresults');
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('simresults/pretrained_GP_singletrack_%s.mat', timestamp);

save(filename, 'd_GP');

fprintf('========================================\n');
fprintf('GP Training Complete!\n');
fprintf('========================================\n');
fprintf('Saved to:\n  %s\n\n', filename);
fprintf('Dictionary size: %d/%d\n', size(d_GP.X, 2), GP_Nmax);
fprintf('Performance summary:\n');
fprintf('  ΔF_y_f RMSE: %.2f N\n', rmse_vx);
fprintf('  ΔF_y_r RMSE: %.2f N\n', rmse_vy);
fprintf('  ΔM_z RMSE: %.2f Nm\n', rmse_r);
if ~isnan(coverage_vx)
    fprintf('  Coverage: vx %.1f%%, vy %.1f%%, r %.1f%%\n\n', coverage_vx, coverage_vy, coverage_r);
else
    fprintf('  Coverage: N/A (single covariance matrix)\n\n');
end

%% ========================================================================
%  VISUALIZATION
%  ========================================================================

fprintf('Generating validation plots...\n');

figure('Name', 'GP Pre-training Validation', 'Color', 'w', 'Position', [50 50 1600 1000]);

% ΔF_y_f
subplot(3,3,1)
plot(all_d(:,1), 'b-', 'LineWidth', 1.5); hold on;
plot(mu_pred(1,:), 'r--', 'LineWidth', 1.5);
plot(mu_pred(1,:) + 2*sigma_all(1,:), 'k:', 'LineWidth', 0.5);
plot(mu_pred(1,:) - 2*sigma_all(1,:), 'k:', 'LineWidth', 0.5);
grid on; xlabel('Sample'); ylabel('\DeltaF_{y,f} [N]');
title('Front Tire Force Residual');
legend('True', 'GP Mean', '2\sigma', 'Location', 'best');

subplot(3,3,4)
plot(error_vx, 'b-', 'LineWidth', 1); hold on;
if ~isnan(coverage_vx)
    plot(2*sigma_vx, 'r--', 'LineWidth', 1);
    plot(-2*sigma_vx, 'r--', 'LineWidth', 1);
end
yline(0, 'k:', 'LineWidth', 0.5);
grid on; xlabel('Sample'); ylabel('Error [N]');
if ~isnan(coverage_vx)
    title(sprintf('ΔFyf Error (%.1f%% in 2\\sigma)', coverage_vx));
else
    title('ΔFyf Error');
end

subplot(3,3,7)
scatter(all_d(:,1), mu_pred(1,:), 10, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
plot([min(all_d(:,1)) max(all_d(:,1))], [min(all_d(:,1)) max(all_d(:,1))], 'k--', 'LineWidth', 1.5);
grid on; xlabel('True [N]'); ylabel('Predicted [N]');
title(sprintf('ΔFyf (RMSE=%.2f N)', rmse_vx));
axis equal; axis tight;

% ΔF_y_r
subplot(3,3,2)
plot(all_d(:,2), 'b-', 'LineWidth', 1.5); hold on;
plot(mu_pred(2,:), 'r--', 'LineWidth', 1.5);
plot(mu_pred(2,:) + 2*sigma_all(2,:), 'k:', 'LineWidth', 0.5);
plot(mu_pred(2,:) - 2*sigma_all(2,:), 'k:', 'LineWidth', 0.5);
grid on; xlabel('Sample'); ylabel('\DeltaF_{y,r} [N]');
title('Rear Tire Force Residual');
legend('True', 'GP Mean', '2\sigma', 'Location', 'best');

subplot(3,3,5)
plot(error_vy, 'b-', 'LineWidth', 1); hold on;
if ~isnan(coverage_vy)
    plot(2*sigma_vy, 'r--', 'LineWidth', 1);
    plot(-2*sigma_vy, 'r--', 'LineWidth', 1);
end
yline(0, 'k:', 'LineWidth', 0.5);
grid on; xlabel('Sample'); ylabel('Error [N]');
if ~isnan(coverage_vy)
    title(sprintf('ΔFyr Error (%.1f%% in 2\\sigma)', coverage_vy));
else
    title('ΔFyr Error');
end

subplot(3,3,8)
scatter(all_d(:,2), mu_pred(2,:), 10, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
plot([min(all_d(:,2)) max(all_d(:,2))], [min(all_d(:,2)) max(all_d(:,2))], 'k--', 'LineWidth', 1.5);
grid on; xlabel('True [N]'); ylabel('Predicted [N]');
title(sprintf('ΔFyr (RMSE=%.2f N)', rmse_vy));
axis equal; axis tight;

% ΔM_z
subplot(3,3,3)
plot(all_d(:,3), 'b-', 'LineWidth', 1.5); hold on;
plot(mu_pred(3,:), 'r--', 'LineWidth', 1.5);
plot(mu_pred(3,:) + 2*sigma_all(3,:), 'k:', 'LineWidth', 0.5);
plot(mu_pred(3,:) - 2*sigma_all(3,:), 'k:', 'LineWidth', 0.5);
grid on; xlabel('Sample'); ylabel('\DeltaM_z [Nm]');
title('Yaw Moment Residual');
legend('True', 'GP Mean', '2\sigma', 'Location', 'best');

subplot(3,3,6)
plot(error_r, 'b-', 'LineWidth', 1); hold on;
if ~isnan(coverage_r)
    plot(2*sigma_r, 'r--', 'LineWidth', 1);
    plot(-2*sigma_r, 'r--', 'LineWidth', 1);
end
yline(0, 'k:', 'LineWidth', 0.5);
grid on; xlabel('Sample'); ylabel('Error [Nm]');
if ~isnan(coverage_r)
    title(sprintf('ΔMz Error (%.1f%% in 2\\sigma)', coverage_r));
else
    title('ΔMz Error');
end

subplot(3,3,9)
scatter(all_d(:,3), mu_pred(3,:), 10, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
plot([min(all_d(:,3)) max(all_d(:,3))], [min(all_d(:,3)) max(all_d(:,3))], 'k--', 'LineWidth', 1.5);
grid on; xlabel('True [Nm]'); ylabel('Predicted [Nm]');
title(sprintf('ΔMz (RMSE=%.2f Nm)', rmse_r));
axis equal; axis tight;

sgtitle(sprintf('GP Pre-training Validation: Tire Force Residuals (N=%d/%d)', ...
    size(d_GP.X,2), GP_Nmax));

fprintf('Plots generated!\n');
fprintf('========================================\n\n');
