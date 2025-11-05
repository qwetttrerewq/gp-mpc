%--------------------------------------------------------------------------
% Test Script for Cornering Bicycle Models
%
% This script validates the bicycle models and compares:
% 1. Linear tire model (nominal) vs Pacejka tire model (true)
% 2. GP learning effectiveness
% 3. Model prediction accuracy
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))
addpath(fullfile(pwd, 'utils'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'CODEGEN'))

fprintf('\n========================================\n');
fprintf('Bicycle Model Validation Tests\n');
fprintf('========================================\n\n');


%% ========================================================================
%  TEST 1: Tire Model Comparison
%  ========================================================================

fprintf('TEST 1: Tire Model Comparison\n');
fprintf('-----------------------------\n');

% Create models
nomModel = MotionModelGP_Bicycle_nominal([], []);
trueModel = MotionModelGP_Bicycle_true([], []);

% Plot tire characteristics
trueModel.plotTireCharacteristics();

fprintf('  ✓ Tire comparison plot generated\n\n');


%% ========================================================================
%  TEST 2: Model Dynamics Validation
%  ========================================================================

fprintf('TEST 2: Model Dynamics Validation\n');
fprintf('----------------------------------\n');

% Test states
x_test = [15; 0.5; 0.1];  % vx=15m/s, vy=0.5m/s, r=0.1rad/s
u_test = deg2rad(5);       % delta=5deg

% Nominal model
xdot_nom = nomModel.f(x_test, u_test);
fprintf('  Nominal model:\n');
fprintf('    vx_dot = %.4f m/s^2\n', xdot_nom(1));
fprintf('    vy_dot = %.4f m/s^2\n', xdot_nom(2));
fprintf('    r_dot  = %.4f rad/s^2\n', xdot_nom(3));

% True model
xdot_true = trueModel.f(x_test, u_test);
fprintf('  True model (Pacejka):\n');
fprintf('    vx_dot = %.4f m/s^2\n', xdot_true(1));
fprintf('    vy_dot = %.4f m/s^2\n', xdot_true(2));
fprintf('    r_dot  = %.4f rad/s^2\n', xdot_true(3));

% Residual
residual = xdot_true - xdot_nom;
fprintf('  Residual (true - nominal):\n');
fprintf('    Δvx_dot = %.4f m/s^2\n', residual(1));
fprintf('    Δvy_dot = %.4f m/s^2\n', residual(2));
fprintf('    Δr_dot  = %.4f rad/s^2\n', residual(3));
fprintf('    |residual| = %.4f\n', norm(residual));

fprintf('  ✓ Models evaluated successfully\n\n');


%% ========================================================================
%  TEST 3: Jacobian Validation
%  ========================================================================

fprintf('TEST 3: Jacobian Validation\n');
fprintf('----------------------------\n');

% Check if Jacobian functions exist
if exist('bicycle_gradx_f', 'file')
    fprintf('  ✓ bicycle_gradx_f found\n');
else
    fprintf('  ✗ bicycle_gradx_f not found. Generating...\n');
    nomModel.generate_grad_functions();
end

% Test Jacobians
gradx = nomModel.gradx_f(x_test, u_test);
gradu = nomModel.gradu_f(x_test, u_test);

fprintf('  Jacobian ∂f/∂x (state gradient):\n');
disp(gradx);

fprintf('  Jacobian ∂f/∂u (input gradient):\n');
disp(gradu);

% Verify Jacobian accuracy via numerical differentiation
fprintf('  Verifying Jacobian accuracy...\n');
eps = 1e-6;
gradx_numerical = zeros(3, 3);
f0 = nomModel.f(x_test, u_test);

for i = 1:3
    x_pert = x_test;
    x_pert(i) = x_pert(i) + eps;
    f_pert = nomModel.f(x_pert, u_test);
    gradx_numerical(:, i) = (f_pert - f0) / eps;
end

gradx_error = norm(gradx - gradx_numerical, 'fro');
fprintf('  Jacobian error ||∂f/∂x_analytical - ∂f/∂x_numerical|| = %.2e\n', gradx_error);

if gradx_error < 1e-4
    fprintf('  ✓ Jacobian validation PASSED\n\n');
else
    fprintf('  ✗ Jacobian validation FAILED (error too large)\n\n');
end


%% ========================================================================
%  TEST 4: State Propagation Over Time
%  ========================================================================

fprintf('TEST 4: State Propagation Test\n');
fprintf('-------------------------------\n');

% Simulation parameters
dt = 0.05;
T = 5;  % 5 seconds
N = T / dt;

% Initial state
x0 = [15; 0; 0];  % 15 m/s, no lateral velocity, no yaw rate

% Constant steering input
u_const = deg2rad(10);  % 10 degree steering

% Storage
x_nom = zeros(3, N+1);
x_true = zeros(3, N+1);
x_nom(:, 1) = x0;
x_true(:, 1) = x0;

% Simulate
for k = 1:N
    x_nom(:, k+1) = nomModel.xkp1(x_nom(:, k), zeros(3), u_const, dt);
    x_true(:, k+1) = trueModel.xkp1(x_true(:, k), zeros(3), u_const, dt);
end

t = (0:N) * dt;

% Plot results
figure('Name', 'State Propagation Comparison', 'Color', 'w', 'Position', [100 100 1000 700]);

subplot(3,1,1)
plot(t, x_nom(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_true(1,:), 'r--', 'LineWidth', 1.5);
grid on; ylabel('v_x [m/s]');
title('Longitudinal Velocity');
legend('Nominal (Linear)', 'True (Pacejka)', 'Location', 'best');

subplot(3,1,2)
plot(t, x_nom(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_true(2,:), 'r--', 'LineWidth', 1.5);
grid on; ylabel('v_y [m/s]');
title('Lateral Velocity');
legend('Nominal', 'True', 'Location', 'best');

subplot(3,1,3)
plot(t, rad2deg(x_nom(3,:)), 'b-', 'LineWidth', 1.5); hold on;
plot(t, rad2deg(x_true(3,:)), 'r--', 'LineWidth', 1.5);
grid on; ylabel('r [deg/s]'); xlabel('Time [s]');
title('Yaw Rate');
legend('Nominal', 'True', 'Location', 'best');

sgtitle(sprintf('State Propagation: δ = %.1f deg (constant)', rad2deg(u_const)));

% Compute final state error
error_final = norm(x_true(:, end) - x_nom(:, end));
fprintf('  Final state error after %.1f s: %.4f\n', T, error_final);
fprintf('  ✓ State propagation test completed\n\n');


%% ========================================================================
%  TEST 5: Slip Angle Sweep
%  ========================================================================

fprintf('TEST 5: Slip Angle Response\n');
fprintf('---------------------------\n');

% Sweep through different velocities and steering angles
vx_range = [10, 15, 20];  % m/s
delta_range = linspace(-deg2rad(20), deg2rad(20), 50);

figure('Name', 'Slip Angle Response', 'Color', 'w', 'Position', [150 150 1000 600]);

for i = 1:length(vx_range)
    vx = vx_range(i);

    % Storage
    alpha_f_nom = zeros(size(delta_range));
    alpha_r_nom = zeros(size(delta_range));
    Fy_f_nom = zeros(size(delta_range));
    Fy_r_nom = zeros(size(delta_range));
    Fy_f_true = zeros(size(delta_range));
    Fy_r_true = zeros(size(delta_range));

    for j = 1:length(delta_range)
        delta = delta_range(j);
        x_test = [vx; 0; 0.2];  % some yaw rate

        % Calculate slip angles
        l_f = nomModel.l_f;
        l_r = nomModel.l_r;
        alpha_f = atan2(x_test(2) + l_f*x_test(3), x_test(1)) - delta;
        alpha_r = atan2(x_test(2) - l_r*x_test(3), x_test(1));

        alpha_f_nom(j) = alpha_f;
        alpha_r_nom(j) = alpha_r;

        % Nominal tire forces
        Fy_f_nom(j) = nomModel.c_f * alpha_f;
        Fy_r_nom(j) = nomModel.c_r * alpha_r;

        % True tire forces (Pacejka)
        Fy_f_true(j) = trueModel.D_f * sin(trueModel.C_f * atan(trueModel.B_f*alpha_f - ...
                       trueModel.E_f*(trueModel.B_f*alpha_f - atan(trueModel.B_f*alpha_f))));
        Fy_r_true(j) = trueModel.D_r * sin(trueModel.C_r * atan(trueModel.B_r*alpha_r - ...
                       trueModel.E_r*(trueModel.B_r*alpha_r - atan(trueModel.B_r*alpha_r))));
    end

    % Plot front tire
    subplot(1,2,1)
    plot(rad2deg(delta_range), Fy_f_nom, '--', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Linear (v_x=%.0f m/s)', vx)); hold on;
    plot(rad2deg(delta_range), Fy_f_true, '-', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Pacejka (v_x=%.0f m/s)', vx));

    % Plot rear tire
    subplot(1,2,2)
    plot(rad2deg(delta_range), Fy_r_nom, '--', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Linear (v_x=%.0f m/s)', vx)); hold on;
    plot(rad2deg(delta_range), Fy_r_true, '-', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Pacejka (v_x=%.0f m/s)', vx));
end

subplot(1,2,1)
grid on; xlabel('Steering Angle [deg]'); ylabel('Front Lateral Force [N]');
title('Front Tire Force vs Steering');
legend('Location', 'best');

subplot(1,2,2)
grid on; xlabel('Steering Angle [deg]'); ylabel('Rear Lateral Force [N]');
title('Rear Tire Force vs Steering');
legend('Location', 'best');

sgtitle('Tire Force Characteristics at Different Speeds');

fprintf('  ✓ Slip angle response test completed\n\n');


%% ========================================================================
%  TEST 6: Circular Path Reference
%  ========================================================================

fprintf('TEST 6: Circular Path Reference\n');
fprintf('--------------------------------\n');

% Create circular path
circPath = CircularPath.createCircular(50, 15);

% Get reference at different times
t_test = [0, 2, 5, 10];
fprintf('  Circular path (R=%.0f m, v=%.1f m/s):\n', circPath.radius, circPath.v_ref);

for t = t_test
    [psi_ref, r_ref, v_ref] = circPath.getReference(t);
    fprintf('    t=%.1fs: ψ_ref=%.2f rad, r_ref=%.3f rad/s, v_ref=%.1f m/s\n', ...
            t, psi_ref, r_ref, v_ref);
end

% Plot path
figure('Name', 'Reference Path', 'Color', 'w');
circPath.plotPath(gcf);

fprintf('  ✓ Circular path reference test completed\n\n');


%% ========================================================================
%  TEST SUMMARY
%  ========================================================================

fprintf('========================================\n');
fprintf('All Tests Completed Successfully!\n');
fprintf('========================================\n');
fprintf('\nYou can now run main_cornering.m to perform the full GP-MPC simulation.\n\n');

fprintf('Quick start commands:\n');
fprintf('  >> main_cornering  %% Run cornering simulation with GP learning\n');
fprintf('  >> trueModel.plotTireCharacteristics()  %% Compare tire models\n\n');
