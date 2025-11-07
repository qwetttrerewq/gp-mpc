%--------------------------------------------------------------------------
% Simple NMPC Test for 4-State Vehicle Dynamics (TRUE MODEL - Pacejka)
% State:  x = [vx; vy; psi; r]
% Input:  u = [delta]
%
% This script uses MotionModelGP_Bicycle_true (Pacejka tires)
% All parameters are IDENTICAL to tracking_nominal.m for fair comparison
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'functions'))

fprintf('\n========================================\n');
fprintf('4-State Vehicle Dynamics NMPC Test\n');
fprintf('(TRUE Model - Pacejka Tires)\n');
fprintf('========================================\n\n');

%% Simulation Parameters (SAME as nominal)
dt = 0.1;          % timestep [s]
tf = 10;            % simulation time [s]
N = 20;            % MPC prediction horizon
maxiter = 20;      % max iterations per MPC solve

%% Reference Circular Path (SAME as nominal)
% Simple circular motion: constant yaw rate for circular path
radius = 10;        % [m] circle radius
v_ref = 15;         % [m/s] target velocity
r_ref = v_ref / radius;  % [rad/s] yaw rate for circular path

t_sim_vec = 0:dt:tf;
psi_ref_vec = r_ref * t_sim_vec;

fprintf('Reference path: Circular\n');
fprintf('  Radius: %.1f m\n', radius);
fprintf('  Velocity: %.1f m/s\n', v_ref);
fprintf('  Yaw rate: %.3f rad/s (%.1f deg/s)\n\n', r_ref, rad2deg(r_ref));

%% Define True Model (Pacejka tires)
trueModel = MotionModelGP_Bicycle_true([], []);

n = trueModel.n;  % 4
m = trueModel.m;  % 1 (only delta)

fprintf('True model created (Pacejka tires)\n');
fprintf('  States: n = %d (vx, vy, psi, r)\n', n);
fprintf('  Inputs: m = %d (delta)\n\n', m);

%% Cost Function Weights (SAME as nominal)
weights.q_vx = 1;      % longitudinal velocity tracking
weights.q_vy = 5;      % minimize lateral velocity (stability)
weights.q_psi = 50;    % heading angle tracking
weights.q_r = 50;      % yaw rate tracking (path following)
weights.q_delta = 0.1; % steering effort

%% NMPC Setup
ne = 0;  % no extra variables

% Cost function
fo = @(t,mu_x,var_x,u,e,r) costFcn(mu_x, u, v_ref, r_ref, t, weights);
fend = @(t,mu_x,var_x,e,r) 2 * costFcn(mu_x, zeros(m,1), v_ref, r_ref, t, weights);
% Dynamics (using TRUE model with Pacejka tires)
f = @(mu_x,var_x,u) trueModel.xkp1(mu_x, var_x, u, dt);

% Constraints
h = @(x,u,e) [];  % no equality constraints

% Inequality: sideslip angle limit (SAME as nominal)
beta_max = deg2rad(12);  % max 12 deg sideslip
g = @(x,u,e) constraintFcn(x, beta_max);

% Input bounds (SAME as nominal)
u_lb = -deg2rad(30);  % delta_min
u_ub = deg2rad(30);   % delta_max

% Create NMPC
mpc = NMPC(f, h, g, u_lb, u_ub, n, m, ne, fo, fend, N, dt);
mpc.tol = 1e-4;
mpc.maxiter = maxiter;

fprintf('NMPC initialized\n');
fprintf('  Horizon: N = %d (%.2f s)\n', N, N*dt);
fprintf('  Max iterations: %d\n', maxiter);
fprintf('  Sideslip limit: %.1f deg\n\n', rad2deg(beta_max));

%% Initialize Simulation (SAME as nominal)
x0 = [v_ref; 0; 0; 0];  % Initial states

out.t = 0:dt:tf;
kmax = length(out.t) - 1;

out.x = [x0 NaN(n, kmax)];
out.u = NaN(m, kmax);
out.beta = NaN(1, kmax);
out.cost = NaN(1, kmax);

fprintf('Starting simulation...\n');
fprintf('  Initial state: [%.1f, %.1f, %.3f, %.3f]\n\n', x0);

%% Main Simulation Loop
for k = 1:kmax
    if mod(k, 20) == 0
        fprintf('Time: %.2f / %.2f s\n', out.t(k), tf);
    end
    
    % Solve NMPC
    try
        [u_opt, result] = mpc.optimize(out.x(:,k), out.t(k), 0, false);
        out.u(:,k) = u_opt(:,1);
    catch ME
        warning('MPC failed at k=%d: %s', k, ME.message);
        if k > 1
            out.u(:,k) = out.u(:,k-1);
        else
            out.u(:,k) = 0;
        end
    end
    
    % Apply to TRUE system (Pacejka)
    [mu_xkp1, ~] = trueModel.xkp1(out.x(:,k), zeros(n,n), out.u(:,k), dt);
    out.x(:,k+1) = mu_xkp1;
    
    % Calculate sideslip
    out.beta(k) = atan2(out.x(2,k), out.x(1,k));
end

fprintf('\nSimulation completed!\n');
fprintf('========================================\n\n');

%% Post-Processing
k_valid = find(~isnan(out.x(1,:)), 1, 'last');

% Calculate tracking errors
% State order: x = [vx, vy, psi, r]
vx_error = rms(out.x(1,1:k_valid-1) - v_ref);
psi_error = rms(out.x(3,1:k_valid-1) - psi_ref_vec(1:k_valid-1));  % psi is x(3)
r_error = rms(out.x(4,1:k_valid-1) - r_ref);                        % r is x(4)
max_beta = max(abs(rad2deg(out.beta)));

fprintf('Performance Metrics:\n');
fprintf('  Velocity RMS error: %.3f m/s\n', vx_error);
fprintf('  Heading RMS error: %.4f rad (%.2f deg)\n', psi_error, rad2deg(psi_error));
fprintf('  Yaw rate RMS error: %.4f rad/s\n', r_error);
fprintf('  Max sideslip: %.2f deg\n\n', max_beta);

%% Visualization
figure('Name', 'NMPC Test Results (Pacejka)', 'Color', 'w', 'Position', [100 100 1400 900]);

% States
subplot(4,2,1)
plot(out.t(1:k_valid), out.x(1,1:k_valid), 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid), v_ref*ones(1,k_valid), 'r--', 'LineWidth', 1);
grid on; ylabel('V_{vx} [m/s]'); title('Longitudinal Velocity');
legend('Actual', 'Reference', 'Location', 'best');

subplot(4,2,3)
plot(out.t(1:k_valid), out.x(2,1:k_valid), 'b-', 'LineWidth', 1.5);
grid on; ylabel('V_{vy} [m/s]'); title('Lateral Velocity');

subplot(4,2,5)
plot(out.t(1:k_valid), rad2deg(out.x(3,1:k_valid)), 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid), rad2deg(psi_ref_vec(1:k_valid)), 'r--', 'LineWidth', 1);
grid on; ylabel('\psi [deg]'); title('Heading Angle');
legend('Actual', 'Reference', 'Location', 'best');

subplot(4,2,7)
plot(out.t(1:k_valid), rad2deg(out.x(4,1:k_valid)), 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid), rad2deg(r_ref)*ones(1,k_valid), 'r--', 'LineWidth', 1);
grid on; ylabel('r [deg/s]'); xlabel('Time [s]'); title('Yaw Rate');
legend('Actual', 'Reference', 'Location', 'best');

% Input (only steering)
subplot(4,2,2)
stairs(out.t(1:kmax), rad2deg(out.u), 'b-', 'LineWidth', 1.5); hold on;
yline(rad2deg(u_ub), 'r--', 'LineWidth', 1);
yline(rad2deg(u_lb), 'r--', 'LineWidth', 1);
grid on; ylabel('\delta [deg]'); title('Steering Angle');
legend('Command', 'Bounds', 'Location', 'best');

% Tracking errors
subplot(4,2,4)
plot(out.t(1:k_valid-1), out.x(1,1:k_valid-1) - v_ref, 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid-1), rad2deg(out.x(4,1:k_valid-1) - r_ref), 'r-', 'LineWidth', 1.5);
grid on; ylabel('Error'); title('Velocity & Yaw Rate Errors');
legend('v_x error [m/s]', 'r error [deg/s]', 'Location', 'best');

% Heading tracking error
subplot(4,2,6)
plot(out.t(1:k_valid-1), rad2deg(out.x(3,1:k_valid-1) - psi_ref_vec(1:k_valid-1)), 'b-', 'LineWidth', 1.5);
grid on; ylabel('\psi error [deg]'); title('Heading Tracking Error');
legend('Error', 'Location', 'best');

% Sideslip
subplot(4,2,8)
plot(out.t(1:kmax), rad2deg(out.beta), 'b-', 'LineWidth', 1.5); hold on;
yline(rad2deg(beta_max), 'r--'); yline(-rad2deg(beta_max), 'r--');
grid on; ylabel('\beta [deg]'); xlabel('Time [s]'); title('Sideslip Angle');
legend('Sideslip', 'Bounds', 'Location', 'best');

sgtitle('4-State NMPC Test: Circular Path Tracking (TRUE Model - Pacejka Tires)');

fprintf('Plots generated\n\n');

%% Save results for comparison
save('simresults/tracking_pacejka_results.mat', 'out', 'vx_error', 'psi_error', 'r_error', 'max_beta');
fprintf('Results saved to simresults/tracking_pacejka_results.mat\n\n');

%% Helper Functions

function cost = costFcn(x, u, v_ref, r_ref, t, w)
% Cost function for bicycle model (1 input: delta)
% CRITICAL: State order is x = [vx; vy; psi; r]
vx = x(1);
vy = x(2);
psi = x(3);  % Heading angle
r = x(4);    % Yaw rate
delta = u(1);

psi_ref_k = r_ref * t;

% Full cost function - ALL TERMS NEEDED for proper tracking
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
