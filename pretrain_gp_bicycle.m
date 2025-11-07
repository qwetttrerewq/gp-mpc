%--------------------------------------------------------------------------
% Pre-train GP Model for Bicycle Vehicle Dynamics (IMPROVED VERSION)
%
% Collects training data by simulating true (Pacejka) and nominal (linear)
% models across diverse operating conditions, then trains GP on tire force
% residuals with properly optimized hyperparameters.
%
% Saves: simresults/pretrained_GP_bicycle_<timestamp>.mat
%   - Contains variable 'd_GP' (GP object)
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))

fprintf('\n========================================\n');
fprintf('GP Pre-training for Bicycle Dynamics\n');
fprintf('(IMPROVED VERSION with Hyperparameter Optimization)\n');
fprintf('========================================\n\n');

%% ========================================================================
%  CONFIGURATION
%  ========================================================================

% GP dictionary size
GP_Nmax = 300;  % Increased for better coverage

% Data collection parameters
dt = 0.05;              % Shorter timestep for finer granularity
tf_total = 60;          % Longer duration for more data

% Hyperparameter optimization
optimize_hyper = true;  % ENABLE optimization
opt_method = 'fmincon'; % 'fmincon' or 'ga'

fprintf('Configuration:\n');
fprintf('  Dictionary size: %d\n', GP_Nmax);
fprintf('  Total duration: %.1f s\n', tf_total);
fprintf('  Timestep: %.2f s\n', dt);
fprintf('  Optimize hyperparameters: %s\n', string(optimize_hyper));
fprintf('  Optimization method: %s\n\n', opt_method);

%% ========================================================================
%  CREATE MODELS
%  ========================================================================

% Process noise (relatively small for clean training)
% var_w = diag([(0.05)^2 (deg2rad(0.5))^2]);
var_w = [];

% True model (Pacejka tires)
trueModel = MotionModelGP_true([], var_w);

% Nominal model (Linear tires)
nomModel = MotionModelGP_nominal([], []);

fprintf('Models created:\n');
fprintf('  True: Pacejka tires (Df=%.0f N, Dr=%.0f N)\n', trueModel.D_f, trueModel.D_r);
fprintf('  Nominal: Linear tires (Cf=%.0f N/rad, Cr=%.0f N/rad)\n', nomModel.c_f, nomModel.c_r);
fprintf('  GP input: [vy; r; delta] (3D)\n');
fprintf('  GP output: [ΔF_y_f; ΔF_y_r] (2D tire force residuals)\n\n');

%% ========================================================================
%  COLLECT TRAINING DATA (COMPREHENSIVE SCENARIOS)
%  ========================================================================

fprintf('Collecting training data...\n');

% Define comprehensive scenarios covering the operating range
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
    
    % Medium speed (includes test condition: 15 m/s, 50m radius)
    15, 25, 0.10, 'Medium speed, tight turn'
    15, 35, 0.10, 'Medium speed, medium-tight turn'
    15, 50, 0.12, 'CRITICAL: Medium speed, medium turn (TEST CONDITION)'
    15, 70, 0.08, 'Medium speed, gentle turn'
    
    % % Medium-high speed
    % 17, 40, 0.08, 'Medium-high speed, medium turn'
    % 17, 55, 0.08, 'Medium-high speed, gentle turn'
    % 17, 80, 0.06, 'Medium-high speed, very gentle turn'
    };

n = nomModel.n;
m = nomModel.m;

all_z = [];  % GP inputs
all_d = [];  % GP outputs (tire force residuals)

% Data normalization storage
z_mean = [];
z_std = [];
d_mean = [];
d_std = [];

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
    
    % Initial state [vx; vy; psi; r]
    x0 = [v_ref; 0; 0; 0];
    
    % Storage
    x_true = [x0 zeros(n, kmax)];
    u_sim = zeros(m, kmax);
    
    % Simple proportional controller for yaw rate tracking
    K_r = 0.8;  % Increased gain for faster convergence
    
    for k = 1:kmax
        % Control law (yaw rate tracking)
        r_error = r_ref - x_true(4, k);
        u_sim(:, k) = K_r * r_error;
        u_sim(:, k) = max(min(u_sim(:, k), deg2rad(30)), -deg2rad(30));
        
        % Propagate true model
        [x_true(:, k+1), ~] = trueModel.xkp1(x_true(:, k), zeros(n,n), u_sim(:, k), dt);
        
        % Calculate tire force residual: d = [ΔF_y_f; ΔF_y_r]
        [F_y_f_true, F_y_r_true] = trueModel.getTireForces(x_true(:, k), u_sim(:, k));
        [F_y_f_nom, F_y_r_nom] = nomModel.getTireForces(x_true(:, k), u_sim(:, k));
        d_k = [F_y_f_true - F_y_f_nom; F_y_r_true - F_y_r_nom];
        
        % GP input: z = [vy; r; delta]
        z_k = [nomModel.Bz_x * x_true(:, k); nomModel.Bz_u * u_sim(:, k)];
        
        % Store
        all_z = [all_z z_k];
        all_d = [all_d; d_k'];
    end
    
    fprintf('        Collected %d points (avg residual: Fyf=%.1f N, Fyr=%.1f N)\n', ...
        kmax, mean(abs(all_d(end-kmax+1:end, 1))), mean(abs(all_d(end-kmax+1:end, 2))));
end

fprintf('\nTotal data: %d points\n', size(all_z, 2));

% Subsample if too many points (to avoid numerical issues)
max_training_points = 800;
if size(all_z, 2) > max_training_points
    fprintf('Subsampling from %d to %d points...\n', size(all_z, 2), max_training_points);
    idx_sample = randperm(size(all_z, 2), max_training_points);
    all_z = all_z(:, idx_sample);
    all_d = all_d(idx_sample, :);
    fprintf('  Subsampled!\n\n');
end

% Compute data statistics for better hyperparameter initialization
z_range = max(all_z, [], 2) - min(all_z, [], 2);
d_range = max(all_d) - min(all_d);

fprintf('Data ranges:\n');
fprintf('  vy:    [%.3f, %.3f] m/s (range: %.3f)\n', min(all_z(1,:)), max(all_z(1,:)), z_range(1));
fprintf('  r:     [%.3f, %.3f] rad/s (range: %.3f)\n', min(all_z(2,:)), max(all_z(2,:)), z_range(2));
fprintf('  delta: [%.3f, %.3f] rad (range: %.3f)\n', min(all_z(3,:)), max(all_z(3,:)), z_range(3));
fprintf('  ΔF_y_f: [%.1f, %.1f] N (range: %.1f)\n', min(all_d(:,1)), max(all_d(:,1)), d_range(1));
fprintf('  ΔF_y_r: [%.1f, %.1f] N (range: %.1f)\n\n', min(all_d(:,2)), max(all_d(:,2)), d_range(2));

%% ========================================================================
%  INITIALIZE GP WITH INFORMED HYPERPARAMETERS
%  ========================================================================

fprintf('Initializing GP with data-informed hyperparameters...\n');

gp_n = nomModel.nz;  % 3D
gp_p = nomModel.nd;  % 2D

% Initial hyperparameters - USE CONSERVATIVE VALUES for numerical stability
% Length scales based on typical operating ranges
ell_vy = 0.5;          % vy length scale [m/s]
ell_r = 0.2;           % r length scale [rad/s]
ell_delta = 0.15;      % delta length scale [rad]

% Signal variance: use reasonable values (not data variance!)
% Typical residual is ~1000-2000 N, so variance ~(1500)^2
var_f_Fyf = 2000^2;   % Front tire signal variance [N^2]
var_f_Fyr = 1500^2;   % Rear tire signal variance [N^2]

% Noise variance: 10% of signal variance for stability
var_n_Fyf = var_f_Fyf * 0.1;  % 10% for numerical stability
var_n_Fyr = var_f_Fyr * 0.1;

fprintf('Initial hyperparameters:\n');
fprintf('  Length scales: [%.4f, %.4f, %.4f] (vy, r, delta)\n', ell_vy, ell_r, ell_delta);
fprintf('  Signal variance: [%.2f, %.2f] (Fyf, Fyr)\n', var_f_Fyf, var_f_Fyr);
fprintf('  Noise variance: [%.2f, %.2f]\n\n', var_n_Fyf, var_n_Fyr);

% Set initial hyperparameters
var_f = [var_f_Fyf; var_f_Fyr];
var_n = [var_n_Fyf, 0; 0, var_n_Fyr];

% Length scale matrix (diagonal ARD kernel for each output)
M = zeros(gp_n, gp_n, gp_p);
M(:,:,1) = diag([ell_vy^2, ell_r^2, ell_delta^2]);  % Front tire
M(:,:,2) = diag([ell_vy^2, ell_r^2, ell_delta^2]);  % Rear tire

% Create GP
d_GP = GP(gp_n, gp_p, var_f, var_n, M, GP_Nmax);

% Add training data incrementally in batches (for numerical stability)
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
%  OPTIMIZE HYPERPARAMETERS (WITH ERROR HANDLING)
%  ========================================================================

if optimize_hyper
    fprintf('========================================\n');
    fprintf('Optimizing hyperparameters...\n');
    fprintf('========================================\n\n');
    
    try
        % Optimize
        d_GP.optimizeHyperParams(opt_method);
        
        fprintf('\nOptimization successful!\n');
        fprintf('Optimized hyperparameters:\n');
        fprintf('  Length scales (Fyf): [%.4f, %.4f, %.4f]\n', sqrt(diag(d_GP.M(:,:,1))));
        fprintf('  Length scales (Fyr): [%.4f, %.4f, %.4f]\n', sqrt(diag(d_GP.M(:,:,2))));
        fprintf('  Signal variance: [%.2f, %.2f]\n', d_GP.var_f);
        fprintf('  Noise variance: [%.2f, %.2f]\n\n', diag(d_GP.var_n));
        
    catch ME
        warning('GPMPC:OptimizationFailed', 'Hyperparameter optimization failed: %s', ME.message);
        fprintf('Continuing with initial hyperparameters...\n\n');
    end
end

%% ========================================================================
%  VALIDATE ON TRAINING DATA
%  ========================================================================

fprintf('========================================\n');
fprintf('Validating GP predictions on training data...\n');
fprintf('========================================\n\n');

[mu_pred, var_pred] = d_GP.eval(all_z, false);

error_Fyf = all_d(:, 1) - mu_pred(1, :)';
error_Fyr = all_d(:, 2) - mu_pred(2, :)';

rmse_Fyf = sqrt(mean(error_Fyf.^2));
rmse_Fyr = sqrt(mean(error_Fyr.^2));

% RMSE as percentage of data range
rmse_pct_Fyf = rmse_Fyf / d_range(1) * 100;
rmse_pct_Fyr = rmse_Fyr / d_range(2) * 100;

fprintf('Training Set Performance:\n');
fprintf('  RMSE (ΔF_y_f): %.2f N (%.1f%% of range)\n', rmse_Fyf, rmse_pct_Fyf);
fprintf('  RMSE (ΔF_y_r): %.2f N (%.1f%% of range)\n', rmse_Fyr, rmse_pct_Fyr);

% Relative improvement
rmse_baseline_Fyf = std(all_d(:,1));
rmse_baseline_Fyr = std(all_d(:,2));
improvement_Fyf = (1 - rmse_Fyf / rmse_baseline_Fyf) * 100;
improvement_Fyr = (1 - rmse_Fyr / rmse_baseline_Fyr) * 100;

fprintf('  Improvement vs. zero prediction:\n');
fprintf('    Front: %.1f%%\n', improvement_Fyf);
fprintf('    Rear:  %.1f%%\n', improvement_Fyr);

% 2-sigma coverage
sigma_Fyf = sqrt(var_pred(1, :));
sigma_Fyr = sqrt(var_pred(2, :));

coverage_Fyf = mean(abs(error_Fyf) <= 2*sigma_Fyf') * 100;
coverage_Fyr = mean(abs(error_Fyr) <= 2*sigma_Fyr') * 100;

fprintf('  2σ coverage (target: 95%%):\n');
fprintf('    Front: %.1f%%\n', coverage_Fyf);
fprintf('    Rear:  %.1f%%\n\n', coverage_Fyr);

% Check if GP is learning
if improvement_Fyf < 50 || improvement_Fyr < 50
    warning('GP improvement is less than 50%% - may need more data or better hyperparameters!');
end

if coverage_Fyf < 80 || coverage_Fyr < 80
    warning('2-sigma coverage is less than 80%% - GP uncertainty may be underestimated!');
end

%% ========================================================================
%  SAVE
%  ========================================================================

if ~exist('simresults', 'dir')
    mkdir('simresults');
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('simresults/pretrained_GP_bicycle_%s.mat', timestamp);

% Save GP (variable name must be 'd_GP' for compatibility)
save(filename, 'd_GP');

fprintf('========================================\n');
fprintf('GP Training Complete!\n');
fprintf('========================================\n');
fprintf('Saved to:\n  %s\n\n', filename);
fprintf('Dictionary size: %d/%d\n', size(d_GP.X, 2), GP_Nmax);
fprintf('Performance summary:\n');
fprintf('  Front tire RMSE: %.2f N (%.1f%% improvement)\n', rmse_Fyf, improvement_Fyf);
fprintf('  Rear tire RMSE:  %.2f N (%.1f%% improvement)\n', rmse_Fyr, improvement_Fyr);
fprintf('  Coverage: Front %.1f%%, Rear %.1f%%\n\n', coverage_Fyf, coverage_Fyr);

fprintf('Usage:\n');
fprintf('  main_cornering_gp(''circular'', ''GPfile'', ''%s'')\n\n', filename);

%% ========================================================================
%  VISUALIZATION
%  ========================================================================

fprintf('Generating validation plots...\n');

figure('Name', 'GP Pre-training Validation', 'Color', 'w', 'Position', [50 50 1400 900]);

% ΔF_y_f prediction time series
subplot(3,3,1)
plot(all_d(:,1), 'b-', 'LineWidth', 1.5); hold on;
plot(mu_pred(1,:), 'r--', 'LineWidth', 1.5);
plot(mu_pred(1,:) + 2*sigma_Fyf, 'k:', 'LineWidth', 0.5);
plot(mu_pred(1,:) - 2*sigma_Fyf, 'k:', 'LineWidth', 0.5);
grid on;
xlabel('Sample');
ylabel('\DeltaF_{y,f} [N]');
title('Front Tire Force Residual');
legend('True', 'GP Mean', '2\sigma bounds', 'Location', 'best');

% ΔF_y_r prediction time series
subplot(3,3,2)
plot(all_d(:,2), 'b-', 'LineWidth', 1.5); hold on;
plot(mu_pred(2,:), 'r--', 'LineWidth', 1.5);
plot(mu_pred(2,:) + 2*sigma_Fyr, 'k:', 'LineWidth', 0.5);
plot(mu_pred(2,:) - 2*sigma_Fyr, 'k:', 'LineWidth', 0.5);
grid on;
xlabel('Sample');
ylabel('\DeltaF_{y,r} [N]');
title('Rear Tire Force Residual');
legend('True', 'GP Mean', '2\sigma bounds', 'Location', 'best');

% Prediction vs True scatter (Front)
subplot(3,3,3)
scatter(all_d(:,1), mu_pred(1,:), 10, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
plot([min(all_d(:,1)) max(all_d(:,1))], [min(all_d(:,1)) max(all_d(:,1))], 'k--', 'LineWidth', 1.5);
grid on;
xlabel('True \DeltaF_{y,f} [N]');
ylabel('Predicted \DeltaF_{y,f} [N]');
title(sprintf('Front Tire (RMSE=%.1f N)', rmse_Fyf));
axis equal; axis tight;

% Error time series (Front)
subplot(3,3,4)
plot(error_Fyf, 'b-', 'LineWidth', 1); hold on;
plot(2*sigma_Fyf, 'r--', 'LineWidth', 1);
plot(-2*sigma_Fyf, 'r--', 'LineWidth', 1);
yline(0, 'k:', 'LineWidth', 0.5);
grid on;
xlabel('Sample');
ylabel('Error [N]');
title(sprintf('Front Tire Error (%.1f%% in 2\\sigma)', coverage_Fyf));

% Error time series (Rear)
subplot(3,3,5)
plot(error_Fyr, 'b-', 'LineWidth', 1); hold on;
plot(2*sigma_Fyr, 'r--', 'LineWidth', 1);
plot(-2*sigma_Fyr, 'r--', 'LineWidth', 1);
yline(0, 'k:', 'LineWidth', 0.5);
grid on;
xlabel('Sample');
ylabel('Error [N]');
title(sprintf('Rear Tire Error (%.1f%% in 2\\sigma)', coverage_Fyr));

% Prediction vs True scatter (Rear)
subplot(3,3,6)
scatter(all_d(:,2), mu_pred(2,:), 10, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
plot([min(all_d(:,2)) max(all_d(:,2))], [min(all_d(:,2)) max(all_d(:,2))], 'k--', 'LineWidth', 1.5);
grid on;
xlabel('True \DeltaF_{y,r} [N]');
ylabel('Predicted \DeltaF_{y,r} [N]');
title(sprintf('Rear Tire (RMSE=%.1f N)', rmse_Fyr));
axis equal; axis tight;

% Error histogram (Front)
subplot(3,3,7)
histogram(error_Fyf, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
hold on;
x_range = linspace(min(error_Fyf), max(error_Fyf), 100);
plot(x_range, normpdf(x_range, 0, std(error_Fyf)), 'r-', 'LineWidth', 2);
grid on;
xlabel('Prediction Error [N]');
ylabel('Probability Density');
title(sprintf('\\DeltaF_{y,f} Error Distribution'));
legend('Empirical', 'Gaussian', 'Location', 'best');

% Error histogram (Rear)
subplot(3,3,8)
histogram(error_Fyr, 50, 'Normalization', 'pdf', 'FaceAlpha', 0.7);
hold on;
x_range = linspace(min(error_Fyr), max(error_Fyr), 100);
plot(x_range, normpdf(x_range, 0, std(error_Fyr)), 'r-', 'LineWidth', 2);
grid on;
xlabel('Prediction Error [N]');
ylabel('Probability Density');
title(sprintf('\\DeltaF_{y,r} Error Distribution'));
legend('Empirical', 'Gaussian', 'Location', 'best');

% Hyperparameter visualization
subplot(3,3,9)
bar_data = [sqrt(diag(d_GP.M(:,:,1)))'; sqrt(diag(d_GP.M(:,:,2)))'];
b = bar(bar_data);
set(gca, 'XTickLabel', {'Front', 'Rear'});
ylabel('Length Scale');
title('Optimized Length Scales');
legend('v_y', 'r', '\delta', 'Location', 'best');
grid on;

sgtitle(sprintf('GP Pre-training Validation (N=%d/%d, Optimization: %s)', ...
    size(d_GP.X,2), GP_Nmax, string(optimize_hyper)));

fprintf('Plots generated!\n');
fprintf('========================================\n\n');
