function main_cornering_gp(scenario, varargin)
%--------------------------------------------------------------------------
% GP-MPC for Vehicle Cornering Performance
%
% Demonstrates GP-augmented MPC for vehicle cornering control
% Combines nominal model (linear tires), true model (Pacejka tires),
% and GP learning of residual dynamics
%
% Usage:
%   main_cornering_gp('circular')          % Circular path tracking
%   main_cornering_gp('track')             % Track following
%   main_cornering_gp('circular', 'GP', false)  % Disable GP
%
% State:  x = [vx; vy; psi; r]  (4D)
% Input:  u = [delta]           (1D)
% GP in:  z = [vy; r; delta]    (3D)
% GP out: d = [ΔF_y_f; ΔF_y_r]  (2D tire force residuals)
%
% Reference: CLAUDE.md specification
%--------------------------------------------------------------------------

% Store input arguments before clearing
if nargin < 1
    scenario_arg = 'circular';
else
    scenario_arg = scenario;
end
varargin_copy = varargin;

close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))
addpath(fullfile(pwd, 'utils'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'CODEGEN'))

%% ========================================================================
%  PARSE INPUT ARGUMENTS
%  ========================================================================

% Set default scenario if not provided
scenario = scenario_arg;

% Parse optional arguments
p = inputParser;
addParameter(p, 'GP', true, @islogical);           % Use GP or not
addParameter(p, 'GPfile', '', @ischar);             % Pre-trained GP file path
addParameter(p, 'Nmax', 100, @isnumeric);           % GP dictionary size
parse(p, varargin_copy{:});

useGP = p.Results.GP;
GPfile = p.Results.GPfile;
GP_Nmax = p.Results.Nmax;

fprintf('\n========================================\n');
fprintf('GP-MPC for Vehicle Cornering\n');
fprintf('========================================\n');
fprintf('Scenario: %s\n', scenario);
fprintf('GP enabled: %s\n', string(useGP));
fprintf('GP dictionary size: %d\n\n', GP_Nmax);

%% ========================================================================
%  SIMULATION PARAMETERS
%  ========================================================================

dt = 0.1;          % timestep [s]
tf = 10;           % simulation time [s]
N = 20;            % MPC prediction horizon
maxiter = 20;      % max iterations per MPC solve

%% ========================================================================
%  REFERENCE TRAJECTORY SETUP
%  ========================================================================

switch lower(scenario)
    case 'circular'
        % Simple circular motion: constant yaw rate for circular path
        radius = 50;        % [m] circle radius
        v_ref = 15;         % [m/s] target velocity
        r_ref = v_ref / radius;  % [rad/s] yaw rate for circular path

        t_sim_vec = 0:dt:tf;
        psi_ref_vec = r_ref * t_sim_vec;

        fprintf('Reference path: Circular\n');
        fprintf('  Radius: %.1f m\n', radius);
        fprintf('  Velocity: %.1f m/s\n', v_ref);
        fprintf('  Yaw rate: %.3f rad/s (%.1f deg/s)\n\n', r_ref, rad2deg(r_ref));

        refType = 'circular';

    case 'track'
        % Load race track
        [trackdata, x0_track, th0, w] = RaceTrack.loadTrack_02();
        track = RaceTrack(trackdata, x0_track, th0, w);

        fprintf('Reference path: Race track loaded\n');
        fprintf('  Track points: %d\n\n', size(trackdata, 2));

        refType = 'track';
        v_ref = 15;  % default velocity for track

    otherwise
        error('Unknown scenario: %s. Use ''circular'' or ''track''.', scenario);
end

%% ========================================================================
%  CREATE DYNAMICS MODELS
%  ========================================================================

% Process noise for true model (small for realistic simulation)
var_w = diag([(0.1)^2 (deg2rad(1))^2]);  % tire force noise [F_y_f; F_y_r] in [N^2]

% True model with Pacejka tires
trueModel = MotionModelGP_Bicycle_true([], var_w);

% Nominal model with linear tires (no GP, no noise)
nomModel = MotionModelGP_Bicycle_nominal([], []);

fprintf('Models created:\n');
fprintf('  True model: Pacejka tires (nonlinear)\n');
fprintf('  Nominal model: Linear tires\n');
fprintf('  States: n = %d (vx, vy, psi, r)\n', nomModel.n);
fprintf('  Inputs: m = %d (delta)\n\n', nomModel.m);

%% ========================================================================
%  CREATE GAUSSIAN PROCESS (GP) MODEL
%  ========================================================================

% GP input/output dimensions
gp_n = nomModel.nz;  % 3D: [vy; r; delta]
gp_p = nomModel.nd;  % 2D: [ΔF_y_f; ΔF_y_r] tire force residuals

if ~isempty(GPfile) && exist(GPfile, 'file')
    % Load pre-trained GP
    fprintf('Loading pre-trained GP from: %s\n', GPfile);
    load(GPfile, 'd_GP');
    fprintf('GP loaded successfully\n\n');
else
    % Initialize new GP
    fprintf('Initializing new GP model\n');

    % GP hyperparameters
    var_f = repmat(0.01, [gp_p, 1]);           % output variance
    var_n = diag(var_w / 3);                   % measurement noise variance
    M = repmat(diag([1e0, 1e0, 1e0].^2), [1, 1, gp_p]);  % length scale for [vy; r; delta]

    % Create GP object
    d_GP = GP(gp_n, gp_p, var_f, var_n, M, GP_Nmax);

    fprintf('  Input dimension: %d [vy; r; delta]\n', gp_n);
    fprintf('  Output dimension: %d [ΔF_y_f; ΔF_y_r]\n', gp_p);
    fprintf('  Dictionary size: %d\n\n', GP_Nmax);
end

% Create estimation model (nominal + GP)
estModel = MotionModelGP_Bicycle_nominal(@d_GP.eval, var_w);

% Enable/disable GP
d_GP.isActive = useGP;
fprintf('GP active: %s\n\n', string(d_GP.isActive));

%% ========================================================================
%  COST FUNCTION WEIGHTS
%  ========================================================================

weights.q_vx = 1;      % longitudinal velocity tracking
weights.q_vy = 5;      % minimize lateral velocity (stability)
weights.q_psi = 50;    % heading angle tracking
weights.q_r = 50;      % yaw rate tracking (path following)
weights.q_delta = 0.1; % steering effort

fprintf('Cost function weights:\n');
fprintf('  q_vx = %.1f (velocity tracking)\n', weights.q_vx);
fprintf('  q_vy = %.1f (lateral stability)\n', weights.q_vy);
fprintf('  q_psi = %.1f (heading tracking)\n', weights.q_psi);
fprintf('  q_r = %.1f (yaw rate tracking)\n', weights.q_r);
fprintf('  q_delta = %.2f (steering effort)\n\n', weights.q_delta);

%% ========================================================================
%  NMPC SETUP
%  ========================================================================

n = estModel.n;  % 4 states
m = estModel.m;  % 1 input
ne = 0;          % no extra variables

% Sideslip constraint
beta_max = deg2rad(12);  % max 12 deg sideslip

% Define cost functions
if strcmp(refType, 'circular')
    % Circular path cost
    fo = @(t, mu_x, var_x, u, e, r) costFcn_circular(mu_x, u, v_ref, r_ref, t, weights);
    fend = @(t, mu_x, var_x, e, r) 2 * costFcn_circular(mu_x, zeros(m,1), v_ref, r_ref, t, weights);
else
    % Track-based cost
    fo = @(t, mu_x, var_x, u, e, r) costFcn_track(mu_x, var_x, u, track);
    fend = @(t, mu_x, var_x, e, r) 2 * costFcn_track(mu_x, var_x, zeros(m,1), track);
end

% Define dynamics (using estimation model with GP)
f = @(mu_x, var_x, u) estModel.xkp1(mu_x, var_x, u, dt);

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

fprintf('NMPC initialized:\n');
fprintf('  Horizon: N = %d (%.2f s)\n', N, N*dt);
fprintf('  Max iterations: %d\n', maxiter);
fprintf('  Sideslip limit: %.1f deg\n\n', rad2deg(beta_max));

%% ========================================================================
%  INITIALIZE SIMULATION
%  ========================================================================

% Initial state [vx; vy; psi; r]
if strcmp(refType, 'track')
    x0 = [v_ref; 0; 0; 0];
else
    x0 = [v_ref; 0; 0; 0];  % start at reference velocity
end

% Time vector
out.t = 0:dt:tf;
kmax = length(out.t) - 1;

% Storage arrays
out.x = [x0 NaN(n, kmax)];              % true states
out.xhat = [x0 NaN(n, kmax)];           % state estimation (perfect observer)
out.u = NaN(m, kmax);                   % applied inputs
out.beta = NaN(1, kmax);                % sideslip angles
out.d_est = NaN(gp_p, kmax);            % tire force residuals (measured)
out.d_gp = NaN(gp_p, kmax);             % GP predictions of tire force residuals

% Reference trajectory storage
if strcmp(refType, 'circular')
    out.psi_ref = psi_ref_vec;
    out.r_ref = r_ref;
    out.v_ref = v_ref;
end

fprintf('Simulation initialized:\n');
fprintf('  Duration: %.1f s\n', tf);
fprintf('  Time steps: %d\n', kmax);
fprintf('  Initial state: x0 = [%.1f, %.3f, %.3f, %.3f]\n\n', x0);

%% ========================================================================
%  MAIN SIMULATION LOOP
%  ========================================================================

fprintf('Starting simulation...\n');
fprintf('----------------------------------------------\n');

for k = 1:kmax
    if mod(k, 20) == 0 || k == 1
        fprintf('Time: %.2f / %.2f s (step %d/%d)\n', out.t(k), tf, k, kmax);
    end

    % ---------------------------------------------------------------------
    % NMPC Optimization
    % ---------------------------------------------------------------------
    try
        [u_opt, result] = mpc.optimize(out.xhat(:,k), out.t(k), 0, false);
        out.u(:,k) = u_opt(:,1);
    catch ME
        warning('MPC failed at k=%d: %s', k, ME.message);
        if k > 1
            out.u(:,k) = out.u(:,k-1);
        else
            out.u(:,k) = 0;
        end
    end

    % ---------------------------------------------------------------------
    % Simulate True System (Pacejka tires)
    % ---------------------------------------------------------------------
    [mu_xkp1, ~] = trueModel.xkp1(out.x(:,k), zeros(n,n), out.u(:,k), dt);
    out.x(:,k+1) = mu_xkp1;

    % ---------------------------------------------------------------------
    % Perfect State Observer
    % ---------------------------------------------------------------------
    out.xhat(:,k+1) = out.x(:,k+1);

    % ---------------------------------------------------------------------
    % Estimate Disturbance (Tire Force Residual)
    % ---------------------------------------------------------------------
    % Calculate tire forces from both true (via measured states) and nominal models
    [F_y_f_true, F_y_r_true] = trueModel.getTireForces(out.xhat(:,k), out.u(:,k));
    [F_y_f_nom, F_y_r_nom] = nomModel.getTireForces(out.xhat(:,k), out.u(:,k));

    % Tire force residual: d = [ΔF_y_f; ΔF_y_r]
    d_est = [F_y_f_true - F_y_f_nom; F_y_r_true - F_y_r_nom];
    out.d_est(:,k) = d_est;

    % GP input: z = [vy; r; delta]
    zhat = [estModel.Bz_x * out.xhat(:,k); estModel.Bz_u * out.u(:,k)];

    % GP prediction
    [mu_gp, ~] = d_GP.eval(zhat, true);
    out.d_gp(:,k) = mu_gp;

    % Add data to GP (always collect, but update controlled by useGP)
    if useGP && mod(k-1, 1) == 0  % Add every step
        d_GP.add(zhat, d_est');
        d_GP.updateModel();
    end

    % Calculate sideslip
    out.beta(k) = atan2(out.x(2,k), out.x(1,k));

    % ---------------------------------------------------------------------
    % Safety Check
    % ---------------------------------------------------------------------
    if out.x(1,k+1) < 0
        warning('Vehicle driving backwards at k=%d. Stopping simulation.', k);
        break;
    end
    if abs(rad2deg(out.beta(k))) > 80
        warning('Excessive sideslip (%.1f deg) at k=%d. Stopping.', rad2deg(out.beta(k)), k);
        break;
    end
end

fprintf('----------------------------------------------\n');
fprintf('Simulation completed!\n\n');

%% ========================================================================
%  POST-PROCESSING
%  ========================================================================

% Find valid data range
k_valid = find(~isnan(out.x(1,:)), 1, 'last');

fprintf('Post-processing results...\n');

% Calculate tracking errors (circular only)
if strcmp(refType, 'circular')
    vx_error = rms(out.x(1,1:k_valid-1) - v_ref);
    psi_error = rms(out.x(3,1:k_valid-1) - psi_ref_vec(1:k_valid-1));
    r_error = rms(out.x(4,1:k_valid-1) - r_ref);

    fprintf('\nPerformance Metrics:\n');
    fprintf('  Velocity RMS error: %.3f m/s\n', vx_error);
    fprintf('  Heading RMS error: %.4f rad (%.2f deg)\n', psi_error, rad2deg(psi_error));
    fprintf('  Yaw rate RMS error: %.4f rad/s\n', r_error);
end

% Constraint violations
max_beta = max(abs(rad2deg(out.beta)));
fprintf('  Max sideslip: %.2f deg (limit: %.1f deg)\n', max_beta, rad2deg(beta_max));

% GP performance
if useGP
    predErrorNOgp = out.d_est(:, 1:k_valid-1);
    predErrorWITHgp = out.d_est(:, 1:k_valid-1) - out.d_gp(:, 1:k_valid-1);

    rms_no_gp = sqrt(mean(predErrorNOgp.^2, 2));
    rms_with_gp = sqrt(mean(predErrorWITHgp.^2, 2));

    fprintf('\nGP Prediction Performance:\n');
    fprintf('  RMS error WITHOUT GP: [%.4f, %.4f]\n', rms_no_gp);
    fprintf('  RMS error WITH GP:    [%.4f, %.4f]\n', rms_with_gp);
    fprintf('  Improvement: [%.1f%%, %.1f%%]\n', ...
        (1 - rms_with_gp(1)/rms_no_gp(1))*100, ...
        (1 - rms_with_gp(2)/rms_no_gp(2))*100);
end

fprintf('\n');

%% ========================================================================
%  VISUALIZATION
%  ========================================================================

fprintf('Generating plots...\n');

figure('Name', 'GP-MPC Cornering Results', 'Color', 'w', 'Position', [100 100 1400 900]);

% States
subplot(4,2,1)
plot(out.t(1:k_valid), out.x(1,1:k_valid), 'b-', 'LineWidth', 1.5); hold on;
if strcmp(refType, 'circular')
    plot(out.t(1:k_valid), v_ref*ones(1,k_valid), 'r--', 'LineWidth', 1);
    legend('Actual', 'Reference', 'Location', 'best');
end
grid on; ylabel('v_x [m/s]'); title('Longitudinal Velocity');

subplot(4,2,3)
plot(out.t(1:k_valid), out.x(2,1:k_valid), 'b-', 'LineWidth', 1.5);
grid on; ylabel('v_y [m/s]'); title('Lateral Velocity');

subplot(4,2,5)
plot(out.t(1:k_valid), rad2deg(out.x(3,1:k_valid)), 'b-', 'LineWidth', 1.5); hold on;
if strcmp(refType, 'circular')
    plot(out.t(1:k_valid), rad2deg(psi_ref_vec(1:k_valid)), 'r--', 'LineWidth', 1);
    legend('Actual', 'Reference', 'Location', 'best');
end
grid on; ylabel('\psi [deg]'); title('Heading Angle');

subplot(4,2,7)
plot(out.t(1:k_valid), rad2deg(out.x(4,1:k_valid)), 'b-', 'LineWidth', 1.5); hold on;
if strcmp(refType, 'circular')
    plot(out.t(1:k_valid), rad2deg(r_ref)*ones(1,k_valid), 'r--', 'LineWidth', 1);
    legend('Actual', 'Reference', 'Location', 'best');
end
grid on; ylabel('r [deg/s]'); xlabel('Time [s]'); title('Yaw Rate');

% Input
subplot(4,2,2)
stairs(out.t(1:k_valid-1), rad2deg(out.u), 'b-', 'LineWidth', 1.5); hold on;
yline(rad2deg(u_ub), 'r--', 'LineWidth', 1);
yline(rad2deg(u_lb), 'r--', 'LineWidth', 1);
grid on; ylabel('\delta [deg]'); title('Steering Angle');
legend('Command', 'Bounds', 'Location', 'best');

% Sideslip
subplot(4,2,4)
plot(out.t(1:k_valid-1), rad2deg(out.beta), 'b-', 'LineWidth', 1.5); hold on;
yline(rad2deg(beta_max), 'r--'); yline(-rad2deg(beta_max), 'r--');
grid on; ylabel('\beta [deg]'); title('Sideslip Angle');
legend('Sideslip', 'Bounds', 'Location', 'best');

% GP Predictions (if enabled)
if useGP
    subplot(4,2,6)
    plot(out.t(1:k_valid-1), out.d_est(1,1:k_valid-1), 'b-', 'LineWidth', 1.5); hold on;
    plot(out.t(1:k_valid-1), out.d_gp(1,1:k_valid-1), 'r--', 'LineWidth', 1.5);
    grid on; ylabel('\DeltaF_{y,f} [N]'); title('GP Prediction: Front Tire Force Residual');
    legend('True Residual', 'GP Prediction', 'Location', 'best');

    subplot(4,2,8)
    plot(out.t(1:k_valid-1), out.d_est(2,1:k_valid-1), 'b-', 'LineWidth', 1.5); hold on;
    plot(out.t(1:k_valid-1), out.d_gp(2,1:k_valid-1), 'r--', 'LineWidth', 1.5);
    grid on; ylabel('\DeltaF_{y,r} [N]'); xlabel('Time [s]'); title('GP Prediction: Rear Tire Force Residual');
    legend('True Residual', 'GP Prediction', 'Location', 'best');
else
    subplot(4,2,6)
    plot(out.t(1:k_valid-1), out.d_est(1,1:k_valid-1), 'b-', 'LineWidth', 1.5);
    grid on; ylabel('\DeltaF_{y,f} [N]'); title('Model Mismatch: Front Tire Force');

    subplot(4,2,8)
    plot(out.t(1:k_valid-1), out.d_est(2,1:k_valid-1), 'b-', 'LineWidth', 1.5);
    grid on; ylabel('\DeltaF_{y,r} [N]'); xlabel('Time [s]'); title('Model Mismatch: Rear Tire Force');
end

if useGP
    sgtitle(sprintf('GP-MPC Cornering (%s path) - GP ENABLED', scenario));
else
    sgtitle(sprintf('Nominal MPC Cornering (%s path) - GP DISABLED', scenario));
end

fprintf('Plots generated\n\n');

%% ========================================================================
%  SAVE RESULTS
%  ========================================================================

fprintf('Saving simulation results...\n');
if ~exist('simresults', 'dir')
    mkdir('simresults');
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
if useGP
    filename = sprintf('simresults/cornering_%s_GPtrue_%s.mat', scenario, timestamp);
else
    filename = sprintf('simresults/cornering_%s_GPfalse_%s.mat', scenario, timestamp);
end

save(filename, 'out', 'weights', 'useGP', 'd_GP', 'scenario');
fprintf('Results saved to: %s\n\n', filename);

end  % main function


%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function cost = costFcn_circular(x, u, v_ref, r_ref, t, w)
% Cost function for circular path tracking
% State order: x = [vx; vy; psi; r]
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


function cost = costFcn_track(mu_x, var_x, u, track)
% Cost function for track following (adapted from main_singletrack.m)
% Track oriented penalization
q_l = 50;     % penalization of lag error
q_c = 20;     % penalization of contouring error
q_o = 5;      % penalization for orientation error
q_d = -3;     % reward high track centerline velocites
q_r = 100;    % penalization when vehicle is outside track

% State and input penalization
q_v = 0;      % reward high absolute velocities
q_st = 0.1;   % penalization of steering
q_psidot = 8; % penalize high yaw rates

% Extract states
I_x = mu_x(1);
I_y = mu_x(2);
psi = mu_x(3);
V_vx = mu_x(4);
V_vy = mu_x(5);
psidot = mu_x(6);
track_dist = mu_x(7);

delta = u(1);
track_vel = u(3);  % for track scenario

% Get errors from track
[lag_error, contour_error, offroad_error, orientation_error] = ...
    track.getVehicleDeviation([I_x; I_y], psi, track_dist);

cost_contour = q_c * contour_error^2;
cost_lag = q_l * lag_error^2;
cost_orientation = q_o * orientation_error^2;

% Off-road penalty
gamma = 1000;
lambda = -0.1;
offroad_error = 5*(sqrt((4+gamma*(lambda-offroad_error).^2)/gamma) - (lambda-offroad_error));
cost_outside = q_r * offroad_error^2;

% Velocity cost
cost_vel = q_v * norm([V_vx; V_vy]);

% Yaw rate cost
cost_psidot = q_psidot * psidot^2;

% Track velocity cost
cost_dist = q_d * track_vel;

% Input cost
cost_inputs = q_st * delta^2;

cost = cost_contour + cost_lag + cost_orientation + cost_dist + ...
    cost_outside + cost_inputs + cost_vel + cost_psidot;
end


function g = constraintFcn(x, beta_max)
% Sideslip angle constraint
vx = x(1);
vy = x(2);
beta = atan2(vy, vx);

% |beta| <= beta_max  =>  beta - beta_max <= 0  and  -beta - beta_max <= 0
g = [beta - beta_max;
    -beta - beta_max];
end
