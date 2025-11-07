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

dt = 0.1;          % timestep [s]
tf = 3;           % simulation time [s]
N = 20;            % MPC prediction horizon
maxiter = 20;      % max iterations per MPC solve

fprintf('\n========================================\n');
fprintf('4-State Vehicle NMPC Test (TRUE Model)\n');
fprintf('========================================\n\n');


%% ========================================================================
%  REFERENCE CIRCULAR PATH
%  ========================================================================

% Simple circular motion: constant yaw rate for circular path
radius = 10;        % [m] circle radius (SAME as nominal)
v_ref = 15;         % [m/s] target velocity (SAME as nominal)
r_ref = v_ref / radius;  % [rad/s] yaw rate for circular path

t_sim_vec = 0:dt:tf;
psi_ref_vec = r_ref * t_sim_vec;

fprintf('Reference path: Circular\n');
fprintf('  Radius: %.1f m\n', radius);
fprintf('  Velocity: %.1f m/s\n', v_ref);
fprintf('  Yaw rate: %.3f rad/s (%.1f deg/s)\n\n', r_ref, rad2deg(r_ref));


%% ========================================================================
%  CREATE DYNAMICS MODEL (Pacejka tires ONLY)
%  ========================================================================

% Create true model with Pacejka tires (no process noise)
trueModel = MotionModelGP_true([], []);

fprintf('True model created (Pacejka tires)\n');
fprintf('  States: n = %d (vx, vy, psi, r)\n', trueModel.n);
fprintf('  Inputs: m = %d (delta)\n\n', trueModel.m);


%% ========================================================================
%  CREATE NMPC CONTROLLER
%  ========================================================================

n = trueModel.n;  % 4 states [vx; vy; psi; r]
m = trueModel.m;  % 1 input [delta] (vx is constant)
ne = 0;          % no extra variables

% Cost function weights (SAME as nominal)
weights.q_vx = 1;      % longitudinal velocity tracking (constant vx)
weights.q_vy = 5;      % minimize lateral velocity (stability)
weights.q_psi = 50;    % heading angle tracking
weights.q_r = 50;      % yaw rate tracking (path following)
weights.q_delta = 0.1; % steering effort

% Sideslip constraint
beta_max = deg2rad(12);  % max 12 deg sideslip

% Define cost functions
fo = @(t, mu_x, var_x, u, e, r) costFcn(mu_x, u, v_ref, r_ref, t, weights);
fend = @(t, mu_x, var_x, e, r) 2 * costFcn(mu_x, zeros(m,1), v_ref, r_ref, t, weights);

% Define dynamics (using TRUE model with Pacejka tires)
f = @(mu_x, var_x, u) trueModel.xkp1(mu_x, var_x, u, dt);

% Define constraints
h = @(x, u, e) [];  % equality constraints (none)

% Inequality constraints: sideslip limit
g = @(x, u, e) constraintFcn(x, beta_max);

% Input bounds [delta]
u_lb = -deg2rad(30);  % min steering angle [rad]
u_ub = deg2rad(30);   % max steering angle [rad]

% Create NMPC object
mpc = NMPC(f, h, g, u_lb, u_ub, n, m, ne, fo, fend, N, dt);
mpc.tol = 1e-4;
mpc.maxiter = maxiter;

fprintf('NMPC initialized (using TRUE Pacejka model)\n');
fprintf('  Horizon: N = %d (%.2f s)\n', N, N*dt);
fprintf('  Max iterations: %d\n', maxiter);
fprintf('  Sideslip limit: %.1f deg\n\n', rad2deg(beta_max));


%% ========================================================================
%  INITIALIZE SIMULATION
%  ========================================================================

% Initial state [vx; vy; psi; r]
x0 = [v_ref; 0; 0; 0];  % start at reference velocity, zero slip, zero yaw

% Time vector
out.t = 0:dt:tf;
kmax = length(out.t) - 1;

% Storage arrays
out.x = [x0 NaN(n, kmax)];              % true states
out.u = NaN(m, kmax);                   % applied inputs
out.beta = NaN(1, kmax);                % sideslip angles

fprintf('Starting simulation...\n');
fprintf('  Initial state: [%.1f, %.1f, %.3f]\n\n', x0);


%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================

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
    
    % Apply control to system (TRUE Pacejka model)
    [mu_xkp1, ~] = trueModel.xkp1(out.x(:,k), zeros(n,n), out.u(:,k), dt);
    out.x(:,k+1) = mu_xkp1;
    
    % Calculate sideslip
    out.beta(k) = atan2(out.x(2,k), out.x(1,k));
end

fprintf('\nSimulation completed!\n');
fprintf('========================================\n\n');


%% ========================================================================
%  POST-PROCESSING
%  ========================================================================

% Find valid data range
k_valid = find(~isnan(out.x(1,:)), 1, 'last');

% Calculate tracking errors
vx_error = rms(out.x(1,1:k_valid-1) - v_ref);
psi_error = rms(out.x(3,1:k_valid-1) - psi_ref_vec(1:k_valid-1));
r_error = rms(out.x(4,1:k_valid-1) - r_ref);
max_beta = max(abs(rad2deg(out.beta)));

fprintf('Performance Metrics:\n');
fprintf('  Velocity RMS error: %.3f m/s\n', vx_error);
fprintf('  Heading RMS error: %.4f rad (%.2f deg)\n', psi_error, rad2deg(psi_error));
fprintf('  Yaw rate RMS error: %.4f rad/s\n', r_error);
fprintf('  Max sideslip: %.2f deg\n\n', max_beta);

%% ========================================================================
%  VISUALIZATION
%  ========================================================================

figure('Name', 'NMPC Test Results', 'Color', 'w', 'Position', [100 100 1400 900]);

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

% Input: Steering angle
subplot(4,2,2)
stairs(out.t(1:kmax), rad2deg(out.u), 'b-', 'LineWidth', 1.5); hold on;
yline(rad2deg(u_ub), 'r--', 'LineWidth', 1);
yline(rad2deg(u_lb), 'r--', 'LineWidth', 1);
grid on; ylabel('\delta [deg]'); title('Steering Angle');
legend('Command', 'Bounds', 'Location', 'best');

% Velocity tracking (replaces torque plot)
subplot(4,2,4)
plot(out.t(1:k_valid), out.x(1,1:k_valid), 'b-', 'LineWidth', 1.5); hold on;
plot(out.t(1:k_valid), v_ref*ones(1,k_valid), 'r--', 'LineWidth', 1);
grid on; ylabel('v_x [m/s]'); title('Longitudinal Velocity (Constant)');
legend('Actual', 'Reference', 'Location', 'best');

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


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function cost = costFcn(x, u, v_ref, r_ref, t, w)
% Cost function for bicycle model (2 inputs: delta, T)
% CRITICAL: State order is x = [vx; vy; psi; r]
%           Input order is u = [delta; T]
vx = x(1);
vy = x(2);
psi = x(3);  % Heading angle
r = x(4);    % Yaw rate
delta = u(1);  % Steering angle
T = u(2);      % Torque gain [-1, 1]

psi_ref_k = r_ref * t;

% Full cost function - ALL TERMS NEEDED for proper tracking
cost = w.q_vx * (vx - v_ref)^2 + ...
    w.q_vy * vy^2 + ...
    w.q_psi * (psi - psi_ref_k)^2 + ...
    w.q_r * (r - r_ref)^2 + ...
    w.q_delta * delta^2 + ...
    w.q_T * T^2;
end

function g = constraintFcn(x, beta_max)
vx = x(1);
vy = x(2);
beta = atan2(vy, vx);

% |beta| <= beta_max
g = [beta - beta_max;
    -beta - beta_max];
end
