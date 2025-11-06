%--------------------------------------------------------------------------
% GP-MPC for Vehicle Cornering Performance
%
% Simplified cornering scenario demonstrating GP-augmented MPC
% State:  x = [vx; vy; psi; r]  (4D)
% Input:  u = [delta]      (1D)
% GP in:  z = [vx; vy; r; delta]  (4D)
% GP out: d = [Δvy_dot; Δr_dot]  (2D)
%
% Reference: CLAUDE.md specification
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))
addpath(fullfile(pwd, 'utils'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'CODEGEN'))


%% ========================================================================
%  SIMULATION PARAMETERS
%  ========================================================================

dt = 0.1;          % timestep [s] (finer control)
tf = 10;            % simulation time [s]
N = 10;             % MPC prediction horizon (longer lookahead)
maxiter = 20;       % max iterations per MPC solve (more iterations for convergence)

% GP configuration
loadPreTrainedGP = false;
useGP = false;               % use GP in MPC prediction
trainGPonline = false;       % update GP during simulation

% Display info
lookahead = dt * N;
fprintf('\n========================================\n');
fprintf('GP-MPC Cornering Performance Simulation\n');
fprintf('========================================\n');
fprintf('Prediction horizon: %d steps (%.2f s)\n', N, lookahead);
fprintf('Timestep: %.3f s\n', dt);
fprintf('Total time: %.1f s\n', tf);
fprintf('GP active: %s\n', string(useGP));
fprintf('========================================\n\n');


%% ========================================================================
%  REFERENCE PATH SELECTION
%  ========================================================================

% Choose reference type: 'circular', 'figure8', or 'constant'
path_type = 'circular';

switch path_type
    case 'circular'
        refPath = CircularPath.createCircular(30, 12);  % R=40m, v=16m/s (balanced: ~0.65g lateral)
    case 'figure8'
        refPath = CircularPath.createFigure8(40, 12);   % R=40m, v=12m/s
    case 'constant'
        refPath = CircularPath.createConstant(20);      % v=20m/s straight
end

fprintf('Reference path: %s\n', path_type);
fprintf('Target velocity: %.1f m/s\n\n', refPath.v_ref);


%% ========================================================================
%  CREATE TRUE DYNAMICS MODEL (Pacejka tires)
%  ========================================================================

% Process noise for true system
var_w = diag([(0.1)^2 (deg2rad(1))^2]);  % [vy_dot; r_dot] noise

% Create true model with Pacejka tires
trueModel = MotionModelGP_Bicycle_true([], var_w);


%% ========================================================================
%  CREATE NOMINAL MODEL (Linear tires)
%  ========================================================================

% Nominal model without disturbances (for residual computation)
nomModel = MotionModelGP_Bicycle_nominal([], []);


%% ========================================================================
%  CREATE GP MODEL
%  ========================================================================

if ~loadPreTrainedGP
    % GP dimensions
    gp_n = MotionModelGP_Bicycle_nominal.nz;  % input dim = 4
    gp_p = MotionModelGP_Bicycle_nominal.nd;  % output dim = 2
    
    % GP hyperparameters
    var_f = repmat(0.1, [gp_p, 1]);           % output variance
    var_n = diag(var_w) / 3;                   % measurement noise
    M = repmat(diag([1e0, 1e0, 1e0, 1e0].^2), [1, 1, gp_p]);  % length scales
    maxsize = 200;  % max dictionary size
    
    % Create GP object
    d_GP = GP(gp_n, gp_p, var_f, var_n, M, maxsize);
    fprintf('GP model created (max dictionary size: %d)\n\n', maxsize);
else
    load('simresults/GP_cornering_pretrained.mat', 'd_GP');
    fprintf('Pre-trained GP model loaded\n\n');
end

% Create estimation model (nominal + GP)
estModel = MotionModelGP_Bicycle_nominal(@d_GP.eval, var_w);


%% ========================================================================
%  CREATE NMPC CONTROLLER
%  ========================================================================

n = estModel.n;  % 4 states [vx; vy; psi; r]
m = estModel.m;  % 1 input
ne = 0;          % no extra variables

% Cost function weights
weights.q_vy = 1;          % lateral velocity penalty
weights.q_r = 20;          % yaw rate tracking
weights.q_vx = 10;         % longitudinal velocity tracking
weights.q_delta = 0.5;     % steering penalty
weights.q_ddelta = 5;      % steering rate penalty
weights.q_beta = 100;      % sideslip penalty (safety)

% Chance constraint parameters
beta_max = deg2rad(15);    % max sideslip angle (relaxed for aggressive cornering)
epsilon = 0.05;            % violation probability (5%)
lambda_eps = norminv(1 - epsilon);  % ~ 1.645 for 95% confidence

% Define cost functions
fo = @(t, mu_x, var_x, u, e, r) costFunction(t, mu_x, var_x, u, refPath, weights);
fend = @(t, mu_x, var_x, e, r) 2 * costFunction(t, mu_x, var_x, zeros(m,1), refPath, weights);

% Define dynamics
f = @(mu_x, var_x, u) estModel.xkp1(mu_x, var_x, u, dt);

% Define constraints
h = @(x, u, e) [];  % equality constraints (none)

% Inequality constraints: sideslip and steering limits
% g(x,u) <= 0
g = @(x, u, e) constraintFunction(x, u, beta_max, lambda_eps);

% Input bounds
u_lb = -deg2rad(30);   % min steering angle
u_ub = deg2rad(30);    % max steering angle

% Create NMPC object
mpc = NMPC(f, h, g, u_lb, u_ub, n, m, ne, fo, fend, N, dt);
mpc.tol = 5e-2;       % relaxed tolerance for difficult scenarios
mpc.maxiter = maxiter;

fprintf('NMPC controller initialized\n');
fprintf('  Horizon N = %d\n', N);
fprintf('  Max iterations = %d\n', maxiter);
fprintf('  Sideslip limit = %.1f deg\n', rad2deg(beta_max));
fprintf('  Chance constraint ε = %.1f%%\n', epsilon*100);
fprintf('========================================\n\n');


%% ========================================================================
%  INITIALIZE SIMULATION
%  ========================================================================

% Initial state [vx; vy; psi; r]
x0 = [refPath.v_ref; 0; 0; 0];  % start at reference velocity, zero slip, zero yaw

% Time vector
out.t = 0:dt:tf;
kmax = length(out.t) - 1;

% Storage arrays
out.x = [x0 NaN(n, kmax)];              % true states
out.xhat = [x0 NaN(n, kmax)];           % state estimates
out.xnom = [x0 NaN(n, kmax)];           % nominal predictions
out.u = NaN(m, kmax);                   % applied inputs
out.u_opt = NaN(m, N, kmax);            % optimal input sequences
out.mu_x_pred = NaN(n, N+1, kmax);      % predicted state means
out.var_x_pred = NaN(n, n, N+1, kmax);  % predicted state covariances
out.cost = NaN(1, kmax);                % MPC cost values
out.beta = NaN(1, kmax);                % sideslip angles

% Reference tracking storage
out.r_ref = NaN(1, kmax);
out.psi_ref = NaN(1, kmax);
out.v_ref = NaN(1, kmax);

% GP training data
out.d_est = NaN(2, kmax);               % estimated residuals
out.d_gp = NaN(2, kmax);                % GP predictions
out.z_train = NaN(4, kmax);             % GP input data

% Activate GP
d_GP.isActive = useGP;

fprintf('Simulation initialized\n');
fprintf('  Initial state: vx=%.1f m/s, vy=%.1f m/s, psi=%.3f rad, r=%.3f rad/s\n', x0(1), x0(2), x0(3), x0(4));
fprintf('  Starting simulation...\n\n');


%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================

for k = 1:kmax
    fprintf('----------------------------------------\n');
    fprintf('Time: %.2f / %.2f s (step %d/%d)\n', out.t(k), tf, k, kmax);
    
    % ---------------------------------------------------------------------
    % Get reference trajectory at current time
    % ---------------------------------------------------------------------
    [psi_ref, r_ref, v_ref] = refPath.getReference(out.t(k));
    out.psi_ref(k) = psi_ref;
    out.r_ref(k) = r_ref;
    out.v_ref(k) = v_ref;
    
    % ---------------------------------------------------------------------
    % Solve NMPC optimization
    % ---------------------------------------------------------------------
    tic;
    try
        [u_opt, e_opt] = mpc.optimize(out.xhat(:,k), out.t(k), 0, false);
        out.u(:,k) = u_opt(:,1);
        out.u_opt(:,:,k) = u_opt;
        
        % Get predicted trajectory
        [out.mu_x_pred(:,:,k), out.var_x_pred(:,:,:,k)] = ...
            mpc.predictStateSequence(out.xhat(:,k), zeros(n,n), u_opt);
        
        solve_time = toc;
        fprintf('  MPC solved in %.3f s\n', solve_time);
        fprintf('  Steering: %.2f deg\n', rad2deg(out.u(1,k)));
        
    catch ME
        warning('MPC optimization failed: %s', ME.message);
        if k > 1
            out.u(:,k) = out.u(:,k-1);  % Use previous input
        else
            out.u(:,k) = 0;
        end
    end
    
    % ---------------------------------------------------------------------
    % Apply control to TRUE system
    % ---------------------------------------------------------------------
    [mu_xkp1, var_xkp1] = trueModel.xkp1(out.x(:,k), zeros(n,n), out.u(:,k), dt);
    out.x(:,k+1) = mu_xkp1;  % Use mean (or could sample: mvnrnd(mu, var))
    
    % ---------------------------------------------------------------------
    % State estimation (perfect observer for now)
    % ---------------------------------------------------------------------
    out.xhat(:,k+1) = out.x(:,k+1);
    
    % Calculate sideslip angle
    out.beta(k) = atan2(out.xhat(2,k), out.xhat(1,k));
    fprintf('  Sideslip: %.2f deg\n', rad2deg(out.beta(k)));
    
    % ---------------------------------------------------------------------
    % Compute nominal prediction (for residual calculation)
    % ---------------------------------------------------------------------
    out.xnom(:,k+1) = nomModel.xkp1(out.xhat(:,k), zeros(n,n), out.u(:,k), dt);
    
    % ---------------------------------------------------------------------
    % Calculate residual and update GP
    % ---------------------------------------------------------------------
    if trainGPonline
        % Residual: d = Bd^-1 * (x_true - x_nom)
        d_est = estModel.Bd \ (out.xhat(:,k+1) - out.xnom(:,k+1));
        out.d_est(:,k) = d_est;
        
        % GP input
        z_k = [estModel.Bz_x * out.xhat(:,k); estModel.Bz_u * out.u(:,k)];
        out.z_train(:,k) = z_k;
        
        % GP prediction
        [d_gp, ~] = d_GP.eval(z_k, true);
        out.d_gp(:,k) = d_gp;
        
        % Add to GP dictionary
        d_GP.add(z_k, d_est');
        d_GP.updateModel();
        
        % Error analysis
        error_without_gp = norm(d_est);
        error_with_gp = norm(d_est - d_gp);
        fprintf('  Residual |d|: %.4f\n', error_without_gp);
        fprintf('  GP error |d-d_gp|: %.4f (reduction: %.1f%%)\n', ...
            error_with_gp, (1 - error_with_gp/error_without_gp)*100);
        fprintf('  GP dict size: %d/%d\n', size(d_GP.X,2), d_GP.Nmax);
    end
    
    % ---------------------------------------------------------------------
    % Safety checks
    % ---------------------------------------------------------------------
    if abs(out.beta(k)) > deg2rad(20)
        warning('Large sideslip angle detected: %.1f deg', rad2deg(out.beta(k)));
    end
    
    if out.xhat(1,k+1) < 5
        warning('Low longitudinal velocity: %.1f m/s', out.xhat(1,k+1));
    end
    
    fprintf('\n');
end

fprintf('========================================\n');
fprintf('Simulation completed successfully!\n');
fprintf('========================================\n\n');


%% ========================================================================
%  POST-PROCESSING AND VISUALIZATION
%  ========================================================================

fprintf('Generating plots...\n');

% Find valid data range
k_valid = find(~isnan(out.x(1,:)), 1, 'last');
k_valid_ref = min(k_valid, length(out.r_ref));  % Reference arrays are size kmax

% ---------------------------------------------------------------------
% Figure 1: State trajectories
% ---------------------------------------------------------------------
figure('Name', 'State Trajectories', 'Color', 'w', 'Position', [100 100 1000 700]);

subplot(3,1,1)
plot(out.t(1:k_valid), out.x(1,1:k_valid), 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid), refPath.v_ref*ones(1,k_valid), 'r--', 'LineWidth', 1);
grid on;
ylabel('v_x [m/s]');
title('Longitudinal Velocity');
legend('Actual', 'Reference', 'Location', 'best');

subplot(3,1,2)
plot(out.t(1:k_valid), out.x(2,1:k_valid), 'b-', 'LineWidth', 1.5);
grid on;
ylabel('v_y [m/s]');
title('Lateral Velocity');

subplot(3,1,3)
plot(out.t(1:k_valid), rad2deg(out.x(3,1:k_valid)), 'b-', 'LineWidth', 1.5);
grid on;
ylabel('\psi [deg]');
title('Yaw Angle');

sgtitle('Vehicle State Trajectories');

% ---------------------------------------------------------------------
% Figure 1b: Yaw Rate
% ---------------------------------------------------------------------
figure('Name', 'Yaw Rate Tracking', 'Color', 'w', 'Position', [120 120 800 400]);
plot(out.t(1:k_valid), rad2deg(out.x(4,1:k_valid)), 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid_ref), rad2deg(out.r_ref(1:k_valid_ref)), 'r--', 'LineWidth', 1);
grid on;
ylabel('r [deg/s]');
xlabel('Time [s]');
title('Yaw Rate Tracking');
legend('Actual', 'Reference', 'Location', 'best');

% ---------------------------------------------------------------------
% Figure 2: Control input and sideslip
% ---------------------------------------------------------------------
figure('Name', 'Control and Safety', 'Color', 'w', 'Position', [150 150 1000 600]);

subplot(2,1,1)
stairs(out.t(1:k_valid_ref), rad2deg(out.u(1,1:k_valid_ref)), 'b-', 'LineWidth', 1.5);
hold on;
yline(rad2deg(u_ub), 'r--', 'LineWidth', 1);
yline(rad2deg(u_lb), 'r--', 'LineWidth', 1);
grid on;
ylabel('\delta [deg]');
title('Steering Angle');
legend('Applied', 'Limits', 'Location', 'best');

subplot(2,1,2)
plot(out.t(1:k_valid_ref), rad2deg(out.beta(1:k_valid_ref)), 'b-', 'LineWidth', 1.5);
hold on;
yline(rad2deg(beta_max), 'r--', 'LineWidth', 1);
yline(-rad2deg(beta_max), 'r--', 'LineWidth', 1);
grid on;
ylabel('\beta [deg]');
xlabel('Time [s]');
title('Sideslip Angle');
legend('Actual', 'Limits', 'Location', 'best');

sgtitle('Control Input and Safety Constraints');

% ---------------------------------------------------------------------
% Figure 3: GP learning performance
% ---------------------------------------------------------------------
if trainGPonline
    figure('Name', 'GP Learning', 'Color', 'w', 'Position', [200 200 1000 700]);
    
    subplot(3,1,1)
    plot(out.t(1:k_valid_ref), out.d_est(1,1:k_valid_ref), 'b-', 'LineWidth', 1.5); hold on;
    plot(out.t(1:k_valid_ref), out.d_gp(1,1:k_valid_ref), 'r--', 'LineWidth', 1.5);
    grid on;
    ylabel('d_{vy} [m/s^2]', 'Interpreter', 'tex');
    title('Lateral Acceleration Residual');
    legend('True residual', 'GP prediction', 'Location', 'best');
    
    subplot(3,1,2)
    plot(out.t(1:k_valid_ref), out.d_est(2,1:k_valid_ref), 'b-', 'LineWidth', 1.5); hold on;
    plot(out.t(1:k_valid_ref), out.d_gp(2,1:k_valid_ref), 'r--', 'LineWidth', 1.5);
    grid on;
    ylabel('d_r [rad/s^2]', 'Interpreter', 'tex');
    title('Yaw Acceleration Residual');
    legend('True residual', 'GP prediction', 'Location', 'best');
    
    subplot(3,1,3)
    error_norm = vecnorm(out.d_est(:,1:k_valid_ref) - out.d_gp(:,1:k_valid_ref));
    plot(out.t(1:k_valid_ref), error_norm, 'b-', 'LineWidth', 1.5);
    grid on;
    ylabel('Error norm');
    xlabel('Time [s]');
    title('GP Prediction Error (convergence)');
    
    sgtitle('GP Learning Performance');
    
    % Compute statistics
    fprintf('\n=== GP Learning Statistics ===\n');
    fprintf('Final dictionary size: %d / %d\n', size(d_GP.X, 2), d_GP.Nmax);
    fprintf('Mean residual norm (without GP): %.4f\n', mean(vecnorm(out.d_est(:,1:k_valid_ref))));
    fprintf('Mean prediction error (with GP): %.4f\n', mean(error_norm));
    fprintf('Improvement: %.1f%%\n', (1 - mean(error_norm)/mean(vecnorm(out.d_est(:,1:k_valid_ref))))*100);
end

% ---------------------------------------------------------------------
% Figure 4: Reference path tracking (if applicable)
% ---------------------------------------------------------------------
if ~strcmp(path_type, 'constant')
    figure('Name', 'Path Tracking', 'Color', 'w', 'Position', [250 250 700 700]);
    refPath.plotPath(gcf);
    % Could integrate x,y positions here if extended state
    title(sprintf('%s Path Reference', path_type));
end

fprintf('Plots generated successfully!\n\n');


%% ========================================================================
%  SAVE RESULTS
%  ========================================================================

% Create results directory if it doesn't exist
if ~exist('simresults', 'dir')
    mkdir('simresults');
end

% Save simulation results
save_filename = sprintf('simresults/cornering_%s_GP%s_%s.mat', ...
    path_type, string(useGP), datestr(now, 'yyyymmdd_HHMMSS'));
save(save_filename, 'out', 'd_GP', 'refPath', 'weights');
fprintf('Results saved to: %s\n', save_filename);

fprintf('\n========================================\n');
fprintf('Simulation complete!\n');
fprintf('========================================\n\n');


%% ========================================================================
%  RUN DIAGNOSTIC ANALYSIS
%  ========================================================================

fprintf('Running diagnostic analysis...\n\n');
diagnose_GP_MPC(out, d_GP, trueModel, nomModel, estModel, refPath, beta_max);


%% ========================================================================
%  COST FUNCTION
%  ========================================================================

function cost = costFunction(t, mu_x, var_x, u, refPath, w)
% Extract states
vx = mu_x(1);
vy = mu_x(2);
psi = mu_x(3);
r = mu_x(4);

% Extract input
delta = u(1);

% Get reference
[~, r_ref, v_ref] = refPath.getReference(t);

% Sideslip angle
beta = atan2(vy, vx);

% State penalties
cost_vy = w.q_vy * vy^2;
cost_r = w.q_r * (r - r_ref)^2;
cost_vx = w.q_vx * (vx - v_ref)^2;
cost_beta = w.q_beta * beta^2;

% Input penalties
cost_delta = w.q_delta * delta^2;

% Total cost
% cost = cost_vx + cost_vy + cost_r + cost_delta + cost_beta;
cost = cost_vx + cost_r + cost_delta;
end


%% ========================================================================
%  CONSTRAINT FUNCTION
%  ========================================================================

function g = constraintFunction(x, u, beta_max, lambda_eps)
% Inequality constraints: g(x,u) <= 0
%
% Conservative chance constraint:
%   |μ_β| + λ_ε * σ_β <= β_max
%
% For now, deterministic constraint (can be extended with variance)

vx = x(1);
vy = x(2);

% Sideslip angle
beta = atan2(vy, vx);

% Constraint: |beta| <= beta_max
g = [beta - beta_max;      % beta <= beta_max
    -beta - beta_max];     % -beta <= beta_max  =>  beta >= -beta_max

% Future extension: add variance term
% if size(x,2) > 3  % if variance is provided
%     sigma_beta = sqrt(var_beta_from_covariance);
%     g = g + lambda_eps * sigma_beta;
% end
end
