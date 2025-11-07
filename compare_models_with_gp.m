%--------------------------------------------------------------------------
% Compare True vs Nominal vs (Nominal+GP) Models
%
% This script demonstrates that Nominal+GP model better approximates
% the True model compared to Nominal model alone.
%
% Approach:
%   1. Use MPC with Nominal model to generate control inputs
%   2. Apply same control inputs to: True, Nominal, and (Nominal+GP) models
%   3. Compare state trajectories and tire forces
%   4. Show quantitative improvement with GP
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))
addpath(fullfile(pwd, 'functions'))
addpath(fullfile(pwd, 'utils'))

fprintf('\n========================================\n');
fprintf('Model Comparison: True vs Nominal vs (Nominal+GP)\n');
fprintf('========================================\n\n');

%% ========================================================================
%  LOAD PRE-TRAINED GP
%  ========================================================================

% Find most recent GP file
gp_files = dir('simresults/pretrained_GP_bicycle_*.mat');
if isempty(gp_files)
    error('Pre-trained GP not found! Please run pretrain_gp_bicycle.m first.');
end

% Sort by date and get most recent
[~, idx] = max([gp_files.datenum]);
GP_file = fullfile(gp_files(idx).folder, gp_files(idx).name);

fprintf('Loading pre-trained GP from:\n  %s\n', GP_file);
load(GP_file, 'd_GP');
fprintf('  Dictionary size: %d\n', d_GP.N);
fprintf('  Input dimension: %d\n', d_GP.n);
fprintf('  Output dimension: %d\n\n', d_GP.p);

%% ========================================================================
%  SIMULATION PARAMETERS
%  ========================================================================

dt = 0.1;           % timestep [s]
tf = 10;            % simulation time [s]
N = 20;             % MPC prediction horizon
maxiter = 20;       % max iterations per MPC solve

fprintf('Simulation parameters:\n');
fprintf('  Duration: %.1f s\n', tf);
fprintf('  Timestep: %.2f s\n', dt);
fprintf('  MPC horizon: %d steps\n', N);
fprintf('  Max iterations: %d\n\n', maxiter);

%% ========================================================================
%  REFERENCE TRAJECTORY (Circular Path)
%  ========================================================================

radius = 50;        % [m] circle radius
v_ref = 15;         % [m/s] target velocity
r_ref = v_ref / radius;  % [rad/s] yaw rate for circular path

t_sim_vec = 0:dt:tf;
psi_ref_vec = r_ref * t_sim_vec;

fprintf('Reference path: Circular\n');
fprintf('  Radius: %.1f m\n', radius);
fprintf('  Velocity: %.1f m/s\n', v_ref);
fprintf('  Yaw rate: %.3f rad/s (%.1f deg/s)\n\n', r_ref, rad2deg(r_ref));

%% ========================================================================
%  CREATE MODELS
%  ========================================================================

% Process noise
var_w = diag([(0.1)^2 (deg2rad(1))^2]);

% 1. True model (Pacejka tires)
trueModel = MotionModelGP_Bicycle_true([], var_w);

% 2. Nominal model (Linear tires, no GP)
nominalModel = MotionModelGP_Bicycle_nominal([], []);

% 3. GP-augmented model (Linear tires + GP correction)
gpModel = MotionModelGP_Bicycle_nominal(@d_GP.eval, var_w);
d_GP.isActive = true;

n = nominalModel.n;  % 4 states
m = nominalModel.m;  % 1 input

fprintf('Models created:\n');
fprintf('  1. True model: Pacejka tires\n');
fprintf('  2. Nominal model: Linear tires\n');
fprintf('  3. GP-augmented model: Linear tires + GP correction\n');
fprintf('  States: n = %d [vx, vy, psi, r]\n', n);
fprintf('  Inputs: m = %d [delta]\n\n', m);

%% ========================================================================
%  COST FUNCTION WEIGHTS
%  ========================================================================

weights.q_vx = 1;       % longitudinal velocity tracking
weights.q_vy = 5;       % minimize lateral velocity (stability)
weights.q_psi = 50;     % heading angle tracking
weights.q_r = 50;       % yaw rate tracking (path following)
weights.q_delta = 0.1;  % steering effort

fprintf('Cost function weights:\n');
fprintf('  q_vx = %.1f, q_vy = %.1f, q_psi = %.1f\n', ...
    weights.q_vx, weights.q_vy, weights.q_psi);
fprintf('  q_r = %.1f, q_delta = %.2f\n\n', weights.q_r, weights.q_delta);

%% ========================================================================
%  NMPC SETUP (using Nominal model for control)
%  ========================================================================

ne = 0;  % no extra variables

% Cost function
fo = @(t,mu_x,var_x,u,e,r) costFcn(mu_x, u, v_ref, r_ref, t, weights);
fend = @(t,mu_x,var_x,e,r) 2 * costFcn(mu_x, zeros(m,1), v_ref, r_ref, t, weights);

% Dynamics (using nominal model for MPC)
f = @(mu_x,var_x,u) nominalModel.xkp1(mu_x, var_x, u, dt);

% Constraints
h = @(x,u,e) [];  % no equality constraints

% Inequality: sideslip angle limit
beta_max = deg2rad(12);  % max 12 deg sideslip
g = @(x,u,e) constraintFcn(x, beta_max);

% Input bounds
u_lb = -deg2rad(30);  % delta_min
u_ub = deg2rad(30);   % delta_max

% Create NMPC
mpc = NMPC(f, h, g, u_lb, u_ub, n, m, ne, fo, fend, N, dt);
mpc.tol = 1e-4;
mpc.maxiter = maxiter;

fprintf('NMPC initialized:\n');
fprintf('  Using Nominal model for control\n');
fprintf('  Horizon: N = %d (%.2f s)\n', N, N*dt);
fprintf('  Sideslip limit: %.1f deg\n\n', rad2deg(beta_max));

%% ========================================================================
%  GENERATE CONTROL INPUTS (using MPC with Nominal model)
%  ========================================================================

fprintf('========================================\n');
fprintf('PHASE 1: Generating Control Inputs\n');
fprintf('========================================\n\n');

x0 = [v_ref; 0; 0; 0];  % Initial states

kmax = length(t_sim_vec) - 1;
u_control = NaN(m, kmax);
x_mpc = [x0 NaN(n, kmax)];

fprintf('Running MPC to generate control sequence...\n');

for k = 1:kmax
    if mod(k, 20) == 0
        fprintf('  Progress: %d/%d steps (%.1f%%)\n', k, kmax, k/kmax*100);
    end

    % Solve NMPC
    try
        [u_opt, ~] = mpc.optimize(x_mpc(:,k), t_sim_vec(k), 0, false);
        u_control(:,k) = u_opt(:,1);
    catch ME
        warning('MPC failed at k=%d: %s', k, ME.message);
        if k > 1
            u_control(:,k) = u_control(:,k-1);
        else
            u_control(:,k) = 0;
        end
    end

    % Propagate nominal model (just for reference)
    [mu_xkp1, ~] = nominalModel.xkp1(x_mpc(:,k), zeros(n,n), u_control(:,k), dt);
    x_mpc(:,k+1) = mu_xkp1;
end

fprintf('\nControl sequence generated!\n');
fprintf('  Total control inputs: %d\n\n', kmax);

%% ========================================================================
%  APPLY SAME CONTROLS TO ALL THREE MODELS
%  ========================================================================

fprintf('========================================\n');
fprintf('PHASE 2: Simulating All Models\n');
fprintf('========================================\n\n');

% Storage for all three models
x_true = [x0 NaN(n, kmax)];
x_nom = [x0 NaN(n, kmax)];
x_gp = [x0 NaN(n, kmax)];

% Tire forces storage
F_yf_true = NaN(1, kmax);
F_yr_true = NaN(1, kmax);
F_yf_nom = NaN(1, kmax);
F_yr_nom = NaN(1, kmax);
F_yf_gp = NaN(1, kmax);
F_yr_gp = NaN(1, kmax);

% GP residual prediction storage
gp_residual_pred = NaN(2, kmax);  % GP prediction
gp_residual_true = NaN(2, kmax);  % True residual
gp_residual_var = NaN(2, kmax);   % GP variance

fprintf('Applying control inputs to:\n');
fprintf('  1. True model (Pacejka)\n');
fprintf('  2. Nominal model (Linear)\n');
fprintf('  3. GP-augmented model (Linear + GP)\n\n');

for k = 1:kmax
    if mod(k, 20) == 0
        fprintf('  Progress: %d/%d steps (%.1f%%)\n', k, kmax, k/kmax*100);
    end

    % Get tire forces at current state (before propagation)
    [F_yf_true(k), F_yr_true(k)] = trueModel.getTireForces(x_true(:,k), u_control(:,k));
    [F_yf_nom(k), F_yr_nom(k)] = nominalModel.getTireForces(x_nom(:,k), u_control(:,k));
    [F_yf_gp(k), F_yr_gp(k)] = gpModel.getTireForces(x_gp(:,k), u_control(:,k));

    % Calculate TRUE tire force residual
    gp_residual_true(:,k) = [F_yf_true(k) - F_yf_nom(k); F_yr_true(k) - F_yr_nom(k)];

    % Get GP PREDICTION of residual
    % NOTE: GP uses x_true state for fair comparison (not x_gp which may diverge)
    z_gp = [nominalModel.Bz_x * x_true(:,k); nominalModel.Bz_u * u_control(:,k)];
    [gp_mu, gp_var] = d_GP.eval(z_gp, false);
    gp_residual_pred(:,k) = gp_mu;
    gp_residual_var(:,k) = diag(gp_var);

    % Propagate all three models with SAME control input
    [x_true(:,k+1), ~] = trueModel.xkp1(x_true(:,k), zeros(n,n), u_control(:,k), dt);
    [x_nom(:,k+1), ~] = nominalModel.xkp1(x_nom(:,k), zeros(n,n), u_control(:,k), dt);
    [x_gp(:,k+1), ~] = gpModel.xkp1(x_gp(:,k), zeros(n,n), u_control(:,k), dt);
end

fprintf('\nAll simulations completed!\n\n');

%% ========================================================================
%  COMPUTE COMPARISON METRICS
%  ========================================================================

fprintf('========================================\n');
fprintf('PHASE 3: Computing Metrics\n');
fprintf('========================================\n\n');

% State errors (compared to True model)
error_nom_vx = x_nom(1,:) - x_true(1,:);
error_nom_vy = x_nom(2,:) - x_true(2,:);
error_nom_psi = x_nom(3,:) - x_true(3,:);
error_nom_r = x_nom(4,:) - x_true(4,:);

error_gp_vx = x_gp(1,:) - x_true(1,:);
error_gp_vy = x_gp(2,:) - x_true(2,:);
error_gp_psi = x_gp(3,:) - x_true(3,:);
error_gp_r = x_gp(4,:) - x_true(4,:);

% Tire force errors
error_nom_Fyf = F_yf_nom - F_yf_true;
error_nom_Fyr = F_yr_nom - F_yr_true;
error_gp_Fyf = F_yf_gp - F_yf_true;
error_gp_Fyr = F_yr_gp - F_yr_true;

% RMSE for states
rmse_nom_vx = rms(error_nom_vx(1:kmax));
rmse_nom_vy = rms(error_nom_vy(1:kmax));
rmse_nom_psi = rms(error_nom_psi(1:kmax));
rmse_nom_r = rms(error_nom_r(1:kmax));

rmse_gp_vx = rms(error_gp_vx(1:kmax));
rmse_gp_vy = rms(error_gp_vy(1:kmax));
rmse_gp_psi = rms(error_gp_psi(1:kmax));
rmse_gp_r = rms(error_gp_r(1:kmax));

% RMSE for tire forces
rmse_nom_Fyf = rms(error_nom_Fyf);
rmse_nom_Fyr = rms(error_nom_Fyr);
rmse_gp_Fyf = rms(error_gp_Fyf);
rmse_gp_Fyr = rms(error_gp_Fyr);

% GP Residual Prediction Errors
gp_resid_error_Fyf = gp_residual_true(1,:) - gp_residual_pred(1,:);
gp_resid_error_Fyr = gp_residual_true(2,:) - gp_residual_pred(2,:);
rmse_gp_resid_Fyf = rms(gp_resid_error_Fyf);
rmse_gp_resid_Fyr = rms(gp_resid_error_Fyr);

% GP prediction accuracy (how well GP predicts the residual)
gp_accuracy_Fyf = (1 - rmse_gp_resid_Fyf / rms(gp_residual_true(1,:))) * 100;
gp_accuracy_Fyr = (1 - rmse_gp_resid_Fyr / rms(gp_residual_true(2,:))) * 100;

% Improvement percentages
improve_vx = (1 - rmse_gp_vx/rmse_nom_vx) * 100;
improve_vy = (1 - rmse_gp_vy/rmse_nom_vy) * 100;
improve_psi = (1 - rmse_gp_psi/rmse_nom_psi) * 100;
improve_r = (1 - rmse_gp_r/rmse_nom_r) * 100;
improve_Fyf = (1 - rmse_gp_Fyf/rmse_nom_Fyf) * 100;
improve_Fyr = (1 - rmse_gp_Fyr/rmse_nom_Fyr) * 100;

%% ========================================================================
%  DISPLAY RESULTS
%  ========================================================================

fprintf('COMPARISON RESULTS\n');
fprintf('==========================================\n\n');

fprintf('STATE TRACKING (vs True Model)\n');
fprintf('------------------------------------------\n');
fprintf('Variable    | Nominal RMSE | GP RMSE   | Improvement\n');
fprintf('------------------------------------------\n');
fprintf('vx [m/s]    | %12.4f | %9.4f | %9.1f%%\n', rmse_nom_vx, rmse_gp_vx, improve_vx);
fprintf('vy [m/s]    | %12.4f | %9.4f | %9.1f%%\n', rmse_nom_vy, rmse_gp_vy, improve_vy);
fprintf('psi [rad]   | %12.4f | %9.4f | %9.1f%%\n', rmse_nom_psi, rmse_gp_psi, improve_psi);
fprintf('r [rad/s]   | %12.4f | %9.4f | %9.1f%%\n', rmse_nom_r, rmse_gp_r, improve_r);
fprintf('------------------------------------------\n\n');

fprintf('TIRE FORCE TRACKING (vs True Model)\n');
fprintf('------------------------------------------\n');
fprintf('Variable    | Nominal RMSE | GP RMSE   | Improvement\n');
fprintf('------------------------------------------\n');
fprintf('F_y_f [N]   | %12.2f | %9.2f | %9.1f%%\n', rmse_nom_Fyf, rmse_gp_Fyf, improve_Fyf);
fprintf('F_y_r [N]   | %12.2f | %9.2f | %9.1f%%\n', rmse_nom_Fyr, rmse_gp_Fyr, improve_Fyr);
fprintf('------------------------------------------\n\n');

fprintf('GP RESIDUAL PREDICTION PERFORMANCE\n');
fprintf('------------------------------------------\n');
fprintf('This measures how accurately GP predicts\n');
fprintf('the tire force residual (True - Nominal)\n');
fprintf('------------------------------------------\n');
fprintf('Variable          | True RMS | Pred Error | Accuracy\n');
fprintf('------------------------------------------\n');
fprintf('ΔF_y_f residual   | %8.2f | %10.2f | %7.1f%%\n', ...
    rms(gp_residual_true(1,:)), rmse_gp_resid_Fyf, gp_accuracy_Fyf);
fprintf('ΔF_y_r residual   | %8.2f | %10.2f | %7.1f%%\n', ...
    rms(gp_residual_true(2,:)), rmse_gp_resid_Fyr, gp_accuracy_Fyr);
fprintf('------------------------------------------\n');
fprintf('*** If accuracy is low (<50%%), GP is not\n');
fprintf('    learning the residual correctly! ***\n');
fprintf('------------------------------------------\n\n');

fprintf('AVERAGE IMPROVEMENT: %.1f%%\n', ...
    mean([improve_vx, improve_vy, improve_psi, improve_r, improve_Fyf, improve_Fyr]));
fprintf('==========================================\n\n');

%% ========================================================================
%  VISUALIZATION
%  ========================================================================

fprintf('Generating comparison plots...\n');

% First figure: GP Residual Prediction Analysis
figure('Name', 'GP Residual Prediction Analysis', ...
    'Color', 'w', 'Position', [50 50 1400 800]);

% Front tire residual
subplot(2,2,1)
plot(t_sim_vec(1:kmax), gp_residual_true(1,:), 'k-', 'LineWidth', 2, 'DisplayName', 'True Residual'); hold on;
plot(t_sim_vec(1:kmax), gp_residual_pred(1,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'GP Prediction');
grid on;
xlabel('Time [s]'); ylabel('\DeltaF_{y,f} [N]');
title(sprintf('Front Tire Residual (GP Accuracy: %.1f%%)', gp_accuracy_Fyf));
legend('Location', 'best');

% Rear tire residual
subplot(2,2,2)
plot(t_sim_vec(1:kmax), gp_residual_true(2,:), 'k-', 'LineWidth', 2, 'DisplayName', 'True Residual'); hold on;
plot(t_sim_vec(1:kmax), gp_residual_pred(2,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'GP Prediction');
grid on;
xlabel('Time [s]'); ylabel('\DeltaF_{y,r} [N]');
title(sprintf('Rear Tire Residual (GP Accuracy: %.1f%%)', gp_accuracy_Fyr));
legend('Location', 'best');

% Front tire residual prediction error
subplot(2,2,3)
plot(t_sim_vec(1:kmax), gp_resid_error_Fyf, 'b-', 'LineWidth', 1.5); hold on;
sigma_Fyf = sqrt(gp_residual_var(1,:));
plot(t_sim_vec(1:kmax), 2*sigma_Fyf, 'k--', 'LineWidth', 1, 'DisplayName', '2\sigma bounds');
plot(t_sim_vec(1:kmax), -2*sigma_Fyf, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
yline(0, 'k:', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]'); ylabel('Prediction Error [N]');
title(sprintf('Front Residual Error (RMSE: %.2f N)', rmse_gp_resid_Fyf));
legend('Error', '2\sigma', 'Location', 'best');

% Rear tire residual prediction error
subplot(2,2,4)
plot(t_sim_vec(1:kmax), gp_resid_error_Fyr, 'b-', 'LineWidth', 1.5); hold on;
sigma_Fyr = sqrt(gp_residual_var(2,:));
plot(t_sim_vec(1:kmax), 2*sigma_Fyr, 'k--', 'LineWidth', 1, 'DisplayName', '2\sigma bounds');
plot(t_sim_vec(1:kmax), -2*sigma_Fyr, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
yline(0, 'k:', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]'); ylabel('Prediction Error [N]');
title(sprintf('Rear Residual Error (RMSE: %.2f N)', rmse_gp_resid_Fyr));
legend('Error', '2\sigma', 'Location', 'best');

sgtitle('GP Residual Prediction: How well does GP predict tire force residuals?');

% Second figure: Model Comparison
figure('Name', 'Model Comparison: True vs Nominal vs (Nominal+GP)', ...
    'Color', 'w', 'Position', [50 50 1600 1000]);

% --- State Trajectories ---
% vx
subplot(4,3,1)
plot(t_sim_vec, x_true(1,:), 'k-', 'LineWidth', 2, 'DisplayName', 'True'); hold on;
plot(t_sim_vec, x_nom(1,:), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nominal');
plot(t_sim_vec, x_gp(1,:), 'b-.', 'LineWidth', 1.5, 'DisplayName', 'Nominal+GP');
grid on; ylabel('v_x [m/s]'); title('Longitudinal Velocity');
legend('Location', 'best');

% vy
subplot(4,3,2)
plot(t_sim_vec, x_true(2,:), 'k-', 'LineWidth', 2); hold on;
plot(t_sim_vec, x_nom(2,:), 'r--', 'LineWidth', 1.5);
plot(t_sim_vec, x_gp(2,:), 'b-.', 'LineWidth', 1.5);
grid on; ylabel('v_y [m/s]'); title('Lateral Velocity');

% psi
subplot(4,3,3)
plot(t_sim_vec, rad2deg(x_true(3,:)), 'k-', 'LineWidth', 2); hold on;
plot(t_sim_vec, rad2deg(x_nom(3,:)), 'r--', 'LineWidth', 1.5);
plot(t_sim_vec, rad2deg(x_gp(3,:)), 'b-.', 'LineWidth', 1.5);
grid on; ylabel('\psi [deg]'); title('Heading Angle');

% r
subplot(4,3,4)
plot(t_sim_vec, rad2deg(x_true(4,:)), 'k-', 'LineWidth', 2); hold on;
plot(t_sim_vec, rad2deg(x_nom(4,:)), 'r--', 'LineWidth', 1.5);
plot(t_sim_vec, rad2deg(x_gp(4,:)), 'b-.', 'LineWidth', 1.5);
grid on; ylabel('r [deg/s]'); title('Yaw Rate');

% --- State Errors ---
subplot(4,3,5)
plot(t_sim_vec(1:kmax), error_nom_vx(1:kmax), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Nominal'); hold on;
plot(t_sim_vec(1:kmax), error_gp_vx(1:kmax), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Nominal+GP');
yline(0, 'k--', 'LineWidth', 0.5);
grid on; ylabel('\Deltav_x [m/s]');
title(sprintf('v_x Error (GP: %.1f%% better)', improve_vx));
legend('Location', 'best');

subplot(4,3,6)
plot(t_sim_vec(1:kmax), error_nom_vy(1:kmax), 'r-', 'LineWidth', 1.5); hold on;
plot(t_sim_vec(1:kmax), error_gp_vy(1:kmax), 'b-', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on; ylabel('\Deltav_y [m/s]');
title(sprintf('v_y Error (GP: %.1f%% better)', improve_vy));

subplot(4,3,7)
plot(t_sim_vec(1:kmax), rad2deg(error_nom_psi(1:kmax)), 'r-', 'LineWidth', 1.5); hold on;
plot(t_sim_vec(1:kmax), rad2deg(error_gp_psi(1:kmax)), 'b-', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on; ylabel('\Delta\psi [deg]');
title(sprintf('\\psi Error (GP: %.1f%% better)', improve_psi));

subplot(4,3,8)
plot(t_sim_vec(1:kmax), rad2deg(error_nom_r(1:kmax)), 'r-', 'LineWidth', 1.5); hold on;
plot(t_sim_vec(1:kmax), rad2deg(error_gp_r(1:kmax)), 'b-', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on; ylabel('\Deltar [deg/s]'); xlabel('Time [s]');
title(sprintf('r Error (GP: %.1f%% better)', improve_r));

% --- Tire Forces ---
subplot(4,3,9)
plot(t_sim_vec(1:kmax), F_yf_true, 'k-', 'LineWidth', 2); hold on;
plot(t_sim_vec(1:kmax), F_yf_nom, 'r--', 'LineWidth', 1.5);
plot(t_sim_vec(1:kmax), F_yf_gp, 'b-.', 'LineWidth', 1.5);
grid on; ylabel('F_{y,f} [N]'); title('Front Tire Lateral Force');

subplot(4,3,10)
plot(t_sim_vec(1:kmax), F_yr_true, 'k-', 'LineWidth', 2); hold on;
plot(t_sim_vec(1:kmax), F_yr_nom, 'r--', 'LineWidth', 1.5);
plot(t_sim_vec(1:kmax), F_yr_gp, 'b-.', 'LineWidth', 1.5);
grid on; ylabel('F_{y,r} [N]'); xlabel('Time [s]'); title('Rear Tire Lateral Force');

% --- Tire Force Errors ---
subplot(4,3,11)
plot(t_sim_vec(1:kmax), error_nom_Fyf, 'r-', 'LineWidth', 1.5); hold on;
plot(t_sim_vec(1:kmax), error_gp_Fyf, 'b-', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on; ylabel('\DeltaF_{y,f} [N]'); xlabel('Time [s]');
title(sprintf('F_{y,f} Error (GP: %.1f%% better)', improve_Fyf));

subplot(4,3,12)
plot(t_sim_vec(1:kmax), error_nom_Fyr, 'r-', 'LineWidth', 1.5); hold on;
plot(t_sim_vec(1:kmax), error_gp_Fyr, 'b-', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on; ylabel('\DeltaF_{y,r} [N]'); xlabel('Time [s]');
title(sprintf('F_{y,r} Error (GP: %.1f%% better)', improve_Fyr));

sgtitle('Model Comparison: GP-augmented model approximates True model better than Nominal alone');

fprintf('  Done!\n\n');

%% ========================================================================
%  SUMMARY BAR CHART
%  ========================================================================

figure('Name', 'RMSE Comparison Summary', 'Color', 'w', 'Position', [100 100 1200 600]);

subplot(1,2,1)
categories = {'v_x', 'v_y', '\psi', 'r'};
rmse_nom_states = [rmse_nom_vx, rmse_nom_vy, rmse_nom_psi, rmse_nom_r];
rmse_gp_states = [rmse_gp_vx, rmse_gp_vy, rmse_gp_psi, rmse_gp_r];
b = bar([rmse_nom_states; rmse_gp_states]');
b(1).FaceColor = [0.8 0.2 0.2];
b(2).FaceColor = [0.2 0.6 0.2];
set(gca, 'XTickLabel', categories);
ylabel('RMSE');
title('State Tracking Error');
legend('Nominal', 'Nominal+GP', 'Location', 'best');
grid on;

subplot(1,2,2)
categories2 = {'F_{y,f}', 'F_{y,r}'};
rmse_nom_forces = [rmse_nom_Fyf, rmse_nom_Fyr];
rmse_gp_forces = [rmse_gp_Fyf, rmse_gp_Fyr];
b2 = bar([rmse_nom_forces; rmse_gp_forces]');
b2(1).FaceColor = [0.8 0.2 0.2];
b2(2).FaceColor = [0.2 0.6 0.2];
set(gca, 'XTickLabel', categories2);
ylabel('RMSE [N]');
title('Tire Force Tracking Error');
legend('Nominal', 'Nominal+GP', 'Location', 'best');
grid on;

sgtitle('RMSE Comparison: Nominal vs (Nominal+GP) approximation of True model');

fprintf('========================================\n');
fprintf('Comparison Complete!\n');
fprintf('========================================\n\n');

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function cost = costFcn(x, u, v_ref, r_ref, t, w)
% Cost function for bicycle model
vx = x(1);
vy = x(2);
psi = x(3);
r = x(4);
delta = u(1);

psi_ref_k = r_ref * t;

cost = w.q_vx * (vx - v_ref)^2 + ...
    w.q_vy * vy^2 + ...
    w.q_psi * (psi - psi_ref_k)^2 + ...
    w.q_r * (r - r_ref)^2 + ...
    w.q_delta * delta^2;
end

function g = constraintFcn(x, beta_max)
vx = x(1);
vy = x(2);
beta = atan2(vy, vx);

% |beta| <= beta_max
g = [beta - beta_max;
    -beta - beta_max];
end
