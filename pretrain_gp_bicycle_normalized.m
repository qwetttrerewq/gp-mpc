%--------------------------------------------------------------------------
% Pre-train GP Model for Bicycle Vehicle Dynamics (NORMALIZED VERSION)
%
% This version normalizes the residuals to zero mean and unit variance
% to avoid numerical conditioning problems in GP training
%
% Saves: simresults/pretrained_GP_bicycle_normalized_<timestamp>.mat
%   - Contains variables: 'd_GP', 'd_mean', 'd_std' for denormalization
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))

fprintf('\n========================================\n');
fprintf('GP Pre-training (NORMALIZED VERSION)\n');
fprintf('========================================\n\n');

%% Configuration
GP_Nmax = 200;
dt = 0.1;
tf_total = 20;
optimize_hyper = false;  % Disable optimization for now

fprintf('Configuration:\n');
fprintf('  Dictionary size: %d\n', GP_Nmax);
fprintf('  Total duration: %.1f s\n', tf_total);
fprintf('  Optimize hyperparameters: %s\n\n', string(optimize_hyper));

%% Create Models
var_w = diag([(0.1)^2 (deg2rad(1))^2]);
trueModel = MotionModelGP_Bicycle_true([], var_w);
nomModel = MotionModelGP_Bicycle_nominal([], []);

fprintf('Models created\n');
fprintf('  GP input: [vy; r; delta] (3D)\n');
fprintf('  GP output: [ΔF_y_f; ΔF_y_r] (2D, NORMALIZED)\n\n');

%% Collect Training Data
fprintf('Collecting training data...\n');

scenarios = {
    12, 30, 0.25, 'Low speed'
    15, 50, 0.40, 'Medium speed (TEST CONDITION)'
    17, 55, 0.35, 'High speed'
    };

all_z = [];
all_d = [];

for s = 1:size(scenarios, 1)
    v_ref = scenarios{s, 1};
    radius = scenarios{s, 2};
    duration_frac = scenarios{s, 3};

    scenario_duration = tf_total * duration_frac;
    r_ref = v_ref / radius;

    fprintf('  [%d/%d] v=%.1f m/s, R=%.1f m\n', s, size(scenarios,1), v_ref, radius);

    t_vec = 0:dt:scenario_duration;
    kmax = length(t_vec) - 1;
    x0 = [v_ref; 0; 0; 0];

    x_true = [x0 zeros(4, kmax)];
    u_sim = zeros(1, kmax);
    K_r = 0.5;

    for k = 1:kmax
        r_error = r_ref - x_true(4, k);
        u_sim(:, k) = max(min(K_r * r_error, deg2rad(30)), -deg2rad(30));
        [x_true(:, k+1), ~] = trueModel.xkp1(x_true(:, k), zeros(4,4), u_sim(:, k), dt);

        [F_y_f_true, F_y_r_true] = trueModel.getTireForces(x_true(:, k), u_sim(:, k));
        [F_y_f_nom, F_y_r_nom] = nomModel.getTireForces(x_true(:, k), u_sim(:, k));
        d_k = [F_y_f_true - F_y_f_nom; F_y_r_true - F_y_r_nom];

        z_k = [nomModel.Bz_x * x_true(:, k); nomModel.Bz_u * u_sim(:, k)];

        all_z = [all_z z_k];
        all_d = [all_d; d_k'];
    end
end

fprintf('\nTotal data: %d points\n', size(all_z, 2));
fprintf('  Raw residual range: Fyf [%.1f, %.1f] N, Fyr [%.1f, %.1f] N\n', ...
    min(all_d(:,1)), max(all_d(:,1)), min(all_d(:,2)), max(all_d(:,2)));

%% Normalize Data
d_mean = mean(all_d, 1);  % [1x2]
d_std = std(all_d, 1);     % [1x2]

all_d_norm = (all_d - repmat(d_mean, size(all_d,1), 1)) ./ repmat(d_std, size(all_d,1), 1);

fprintf('\nNormalization parameters:\n');
fprintf('  Front: mean=%.1f N, std=%.1f N\n', d_mean(1), d_std(1));
fprintf('  Rear:  mean=%.1f N, std=%.1f N\n', d_mean(2), d_std(2));
fprintf('  Normalized range: [%.2f, %.2f]\n\n', min(all_d_norm(:)), max(all_d_norm(:)));

%% Initialize GP with UNIT variance
gp_n = 3;
gp_p = 2;

% Conservative hyperparameters for normalized data
ell = 0.5;  % Length scale
var_f = [1.0; 1.0];  % Signal variance (unit for normalized data)
var_n = [0.01, 0; 0, 0.01];  % Small noise variance
M = zeros(gp_n, gp_n, gp_p);
M(:,:,1) = ell^2 * eye(gp_n);
M(:,:,2) = ell^2 * eye(gp_n);

d_GP = GP(gp_n, gp_p, var_f, var_n, M, GP_Nmax);

%% Add Data
fprintf('Adding %d points to GP...\n', size(all_z, 2));
d_GP.add(all_z, all_d_norm);
fprintf('  Dictionary size: %d\n\n', size(d_GP.X, 2));

%% Validate
[mu_pred_norm, var_pred] = d_GP.eval(all_z, false);

% Denormalize predictions
mu_pred = mu_pred_norm .* repmat(d_std', 1, size(mu_pred_norm,2)) + repmat(d_mean', 1, size(mu_pred_norm,2));

error_Fyf = all_d(:, 1) - mu_pred(1, :)';
error_Fyr = all_d(:, 2) - mu_pred(2, :)';

rmse_Fyf = sqrt(mean(error_Fyf.^2));
rmse_Fyr = sqrt(mean(error_Fyr.^2));

fprintf('Validation (denormalized):\n');
fprintf('  RMSE Front: %.2f N\n', rmse_Fyf);
fprintf('  RMSE Rear:  %.2f N\n\n', rmse_Fyr);

%% Save
if ~exist('simresults', 'dir')
    mkdir('simresults');
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('simresults/pretrained_GP_bicycle_normalized_%s.mat', timestamp);

save(filename, 'd_GP', 'd_mean', 'd_std');

fprintf('========================================\n');
fprintf('GP Training Complete!\n');
fprintf('========================================\n');
fprintf('Saved to:\n  %s\n\n', filename);
fprintf('IMPORTANT: This GP outputs NORMALIZED residuals.\n');
fprintf('Use d_mean and d_std to denormalize:\n');
fprintf('  d_real = d_GP_output .* d_std + d_mean\n\n');
