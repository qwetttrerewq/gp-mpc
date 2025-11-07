%--------------------------------------------------------------------------
% Diagnostic Script: Compare MotionModelGP_test vs MotionModelGP_Bicycle_nominal
%
% Purpose: Identify why the Pacejka-based test model performs poorly
%          compared to the linear tire nominal model
%
% Tests:
%   1. Tire force comparison (linear vs Pacejka)
%   2. Open-loop response comparison
%   3. State derivative magnitude analysis
%   4. Slip angle behavior
%   5. Model parameter differences
%--------------------------------------------------------------------------

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))
addpath(fullfile(pwd, 'utils'))

fprintf('\n========================================\n');
fprintf('MODEL DIAGNOSTIC COMPARISON\n');
fprintf('========================================\n\n');

%% ========================================================================
%  CREATE BOTH MODELS
%  ========================================================================

% Nominal model (linear tires)
nomModel = MotionModelGP_Bicycle_nominal([], []);

% Test model (Pacejka tires)
testModel = MotionModelGP_test([], []);

fprintf('Models created:\n');
fprintf('  Nominal: %d states, %d inputs (linear tires)\n', nomModel.n, nomModel.m);
fprintf('  Test:    %d states, %d inputs (Pacejka tires)\n\n', testModel.n, testModel.m);

%% ========================================================================
%  1. PARAMETER COMPARISON
%  ========================================================================

fprintf('1. VEHICLE PARAMETER COMPARISON\n');
fprintf('   ========================================\n');
fprintf('   Parameter      | Nominal   | Test      | Match?\n');
fprintf('   ----------------------------------------\n');
fprintf('   Mass (M)       | %6.1f    | %6.1f    | %s\n', nomModel.M, testModel.M, check_match(nomModel.M, testModel.M));
fprintf('   Inertia (Iz)   | %6.1f    | %6.1f    | %s\n', nomModel.I_z, testModel.I_z, check_match(nomModel.I_z, testModel.I_z));
fprintf('   l_f            | %6.2f    | %6.2f    | %s\n', nomModel.l_f, testModel.l_f, check_match(nomModel.l_f, testModel.l_f));
fprintf('   l_r            | %6.2f    | %6.2f    | %s\n', nomModel.l_r, testModel.l_r, check_match(nomModel.l_r, testModel.l_r));
fprintf('   deltamax       | %6.2f    | %6.2f    | %s\n', nomModel.deltamax, testModel.deltamax, check_match(nomModel.deltamax, testModel.deltamax));
fprintf('   ========================================\n\n');

%% ========================================================================
%  2. TIRE FORCE COMPARISON
%  ========================================================================

fprintf('2. TIRE FORCE COMPARISON\n');
fprintf('   ========================================\n');

% Test range of slip angles
alpha_range = linspace(-deg2rad(15), deg2rad(15), 100);

% Nominal linear tire forces
Fy_f_nom = nomModel.c_f * alpha_range;
Fy_r_nom = nomModel.c_r * alpha_range;

% Pacejka tire forces (front)
Fy_f_test = testModel.D_f * sin(testModel.C_f * atan(...
    testModel.B_f * alpha_range - testModel.E_f * (testModel.B_f * alpha_range - atan(testModel.B_f * alpha_range))));

% Pacejka tire forces (rear)
Fy_r_test = testModel.D_r * sin(testModel.C_r * atan(...
    testModel.B_r * alpha_range - testModel.E_r * (testModel.B_r * alpha_range - atan(testModel.B_r * alpha_range))));

% Plot comparison
figure('Name', 'Tire Force Comparison', 'Position', [100 100 1400 600]);

subplot(1,2,1)
plot(rad2deg(alpha_range), Fy_f_nom, 'b-', 'LineWidth', 2); hold on;
plot(rad2deg(alpha_range), Fy_f_test, 'r--', 'LineWidth', 2);
grid on; xlabel('Slip Angle [deg]'); ylabel('Lateral Force [N]');
title('Front Tire Force Comparison');
legend('Nominal (Linear)', 'Test (Pacejka)', 'Location', 'best');

subplot(1,2,2)
plot(rad2deg(alpha_range), Fy_r_nom, 'b-', 'LineWidth', 2); hold on;
plot(rad2deg(alpha_range), Fy_r_test, 'r--', 'LineWidth', 2);
grid on; xlabel('Slip Angle [deg]'); ylabel('Lateral Force [N]');
title('Rear Tire Force Comparison');
legend('Nominal (Linear)', 'Test (Pacejka)', 'Location', 'best');

% Find peak forces
[max_Fy_f_test, idx_f] = max(abs(Fy_f_test));
[max_Fy_r_test, idx_r] = max(abs(Fy_r_test));

fprintf('   Front tire:\n');
fprintf('     Linear stiffness:   %.1f N/rad\n', nomModel.c_f);
fprintf('     Pacejka peak force: %.1f N at %.2f deg\n', max_Fy_f_test, rad2deg(alpha_range(idx_f)));
fprintf('   Rear tire:\n');
fprintf('     Linear stiffness:   %.1f N/rad\n', nomModel.c_r);
fprintf('     Pacejka peak force: %.1f N at %.2f deg\n\n', max_Fy_r_test, rad2deg(alpha_range(idx_r)));

%% ========================================================================
%  3. STATE DIMENSIONS MISMATCH
%  ========================================================================

fprintf('3. STATE VECTOR ANALYSIS\n');
fprintf('   ========================================\n');
fprintf('   Nominal state: [vx; vy; psi; r]  (n=%d)\n', nomModel.n);
fprintf('   Test state:    [vx; vy; psi; r]  (n=%d)\n', testModel.n);

if nomModel.n ~= testModel.n
    fprintf('   >>> WARNING: State dimensions MISMATCH! <<<\n');
    fprintf('   >>> This will cause MPC prediction errors! <<<\n\n');
else
    fprintf('   State dimensions match: OK\n\n');
end

fprintf('   GP Input dimensions:\n');
fprintf('     Nominal: nz=%d, z=[vx;vy;r;delta]\n', nomModel.nz);
fprintf('     Test:    nz=%d, z=[vx;vy;psi;r;delta;T]\n', testModel.nz);
if nomModel.nz ~= testModel.nz
    fprintf('     >>> WARNING: GP input dimensions differ! <<<\n\n');
end

fprintf('   GP Output dimensions:\n');
fprintf('     Nominal: nd=%d, d=[Δvy_dot; Δr_dot]\n', nomModel.nd);
fprintf('     Test:    nd=%d, d=[Δvx_dot; Δvy_dot; Δpsi_dot; Δr_dot]\n\n', testModel.nd);

%% ========================================================================
%  4. OPEN-LOOP RESPONSE COMPARISON
%  ========================================================================

fprintf('4. OPEN-LOOP STEP RESPONSE\n');
fprintf('   ========================================\n');

% Simulation parameters
dt = 0.05;
tf = 3.0;
t_vec = 0:dt:tf;
N_steps = length(t_vec);

% Initial condition: [vx; vy; psi; r]
x0 = [20; 0; 0; 0];  % 20 m/s, no lateral motion

% Step input: constant steering angle
delta_step = deg2rad(5);  % 5 deg steering
u_nom = delta_step;       % nominal uses 1 input
u_test = [delta_step; 0]; % test uses 2 inputs [delta; T]

% Simulate both models
x_nom = zeros(nomModel.n, N_steps);
x_test = zeros(testModel.n, N_steps);
x_nom(:,1) = x0;
x_test(:,1) = x0;

for k = 1:N_steps-1
    % Nominal model step
    [x_nom(:,k+1), ~] = nomModel.xkp1(x_nom(:,k), zeros(nomModel.n), u_nom, dt);

    % Test model step (suppress Fy_f printout temporarily)
    diary off;
    [x_test(:,k+1), ~] = testModel.xkp1(x_test(:,k), zeros(testModel.n), u_test, dt);
    diary on;
end

% Plot comparison
figure('Name', 'Open-Loop Response Comparison', 'Position', [100 100 1400 800]);

subplot(4,1,1)
plot(t_vec, x_nom(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t_vec, x_test(1,:), 'r--', 'LineWidth', 1.5);
grid on; ylabel('v_x [m/s]'); title('Longitudinal Velocity');
legend('Nominal', 'Test', 'Location', 'best');

subplot(4,1,2)
plot(t_vec, x_nom(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t_vec, x_test(2,:), 'r--', 'LineWidth', 1.5);
grid on; ylabel('v_y [m/s]'); title('Lateral Velocity');
legend('Nominal', 'Test', 'Location', 'best');

subplot(4,1,3)
plot(t_vec, rad2deg(x_nom(3,:)), 'b-', 'LineWidth', 1.5); hold on;
plot(t_vec, rad2deg(x_test(3,:)), 'r--', 'LineWidth', 1.5);
grid on; ylabel('\psi [deg]'); title('Heading Angle');
legend('Nominal', 'Test', 'Location', 'best');

subplot(4,1,4)
plot(t_vec, rad2deg(x_nom(4,:)), 'b-', 'LineWidth', 1.5); hold on;
plot(t_vec, rad2deg(x_test(4,:)), 'r--', 'LineWidth', 1.5);
grid on; ylabel('r [deg/s]'); xlabel('Time [s]'); title('Yaw Rate');
legend('Nominal', 'Test', 'Location', 'best');

sgtitle(sprintf('Step Response: δ = %.1f deg', rad2deg(delta_step)));

% Compare final values
fprintf('   Final values after %.1f s (step input δ=%.1f deg):\n', tf, rad2deg(delta_step));
fprintf('   --------------------------------------------------------\n');
fprintf('   State      | Nominal   | Test      | Difference\n');
fprintf('   --------------------------------------------------------\n');
fprintf('   v_x [m/s]  | %8.3f  | %8.3f  | %8.3f\n', x_nom(1,end), x_test(1,end), x_test(1,end)-x_nom(1,end));
fprintf('   v_y [m/s]  | %8.3f  | %8.3f  | %8.3f\n', x_nom(2,end), x_test(2,end), x_test(2,end)-x_nom(2,end));
fprintf('   ψ [deg]    | %8.3f  | %8.3f  | %8.3f\n', rad2deg(x_nom(3,end)), rad2deg(x_test(3,end)), rad2deg(x_test(3,end)-x_nom(3,end)));
fprintf('   r [deg/s]  | %8.3f  | %8.3f  | %8.3f\n', rad2deg(x_nom(4,end)), rad2deg(x_test(4,end)), rad2deg(x_test(4,end)-x_nom(4,end)));
fprintf('   --------------------------------------------------------\n\n');

%% ========================================================================
%  5. SLIP ANGLE COMPARISON
%  ========================================================================

fprintf('5. SLIP ANGLE BEHAVIOR\n');
fprintf('   ========================================\n');

% Calculate slip angles for both trajectories
alpha_f_nom = zeros(1, N_steps);
alpha_r_nom = zeros(1, N_steps);
alpha_f_test = zeros(1, N_steps);
alpha_r_test = zeros(1, N_steps);

for k = 1:N_steps
    % Nominal model slip angles (using small angle approximation as in code)
    vx = x_nom(1,k);
    vy = x_nom(2,k);
    r = x_nom(4,k);
    alpha_f_nom(k) = delta_step - (nomModel.l_f*r + vy)/vx;
    alpha_r_nom(k) = (nomModel.l_r*r - vy)/vx;

    % Test model slip angles (using atan2 as in code)
    vx = x_test(1,k);
    vy = x_test(2,k);
    r = x_test(4,k);
    alpha_f_test(k) = atan2(vy + testModel.l_f*r, vx) - delta_step;
    alpha_r_test(k) = atan2(vy - testModel.l_r*r, vx);
end

figure('Name', 'Slip Angle Comparison', 'Position', [100 100 1400 500]);

subplot(1,2,1)
plot(t_vec, rad2deg(alpha_f_nom), 'b-', 'LineWidth', 1.5); hold on;
plot(t_vec, rad2deg(alpha_f_test), 'r--', 'LineWidth', 1.5);
grid on; ylabel('α_f [deg]'); xlabel('Time [s]');
title('Front Slip Angle');
legend('Nominal', 'Test', 'Location', 'best');

subplot(1,2,2)
plot(t_vec, rad2deg(alpha_r_nom), 'b-', 'LineWidth', 1.5); hold on;
plot(t_vec, rad2deg(alpha_r_test), 'r--', 'LineWidth', 1.5);
grid on; ylabel('α_r [deg]'); xlabel('Time [s]');
title('Rear Slip Angle');
legend('Nominal', 'Test', 'Location', 'best');

sgtitle('Slip Angle Evolution During Step Response');

fprintf('   Maximum slip angles during step response:\n');
fprintf('     Front (nominal): %.2f deg\n', rad2deg(max(abs(alpha_f_nom))));
fprintf('     Front (test):    %.2f deg\n', rad2deg(max(abs(alpha_f_test))));
fprintf('     Rear (nominal):  %.2f deg\n', rad2deg(max(abs(alpha_r_nom))));
fprintf('     Rear (test):     %.2f deg\n\n', rad2deg(max(abs(alpha_r_test))));

%% ========================================================================
%  6. INSTANTANEOUS DYNAMICS COMPARISON
%  ========================================================================

fprintf('6. INSTANTANEOUS DYNAMICS AT STEADY CORNERING\n');
fprintf('   ========================================\n');

% Test state: steady cornering at 20 m/s, 5 deg/s yaw rate
x_test_point = [20; 1.5; 0; deg2rad(5)];  % [vx; vy; psi; r]
u_test_point = [deg2rad(8); 0];           % [delta; T]
u_nom_point = deg2rad(8);                 % delta only

% Get derivatives
xdot_nom = nomModel.f(x_test_point, u_nom_point);
xdot_test = testModel.f(x_test_point, u_test_point);

fprintf('   Test point: vx=%.1f m/s, vy=%.2f m/s, r=%.2f deg/s, δ=%.1f deg\n', ...
    x_test_point(1), x_test_point(2), rad2deg(x_test_point(4)), rad2deg(u_nom_point));
fprintf('   --------------------------------------------------------\n');
fprintf('   Derivative    | Nominal   | Test      | Difference\n');
fprintf('   --------------------------------------------------------\n');
fprintf('   v_x_dot       | %8.3f  | %8.3f  | %8.3f\n', xdot_nom(1), xdot_test(1), xdot_test(1)-xdot_nom(1));
fprintf('   v_y_dot       | %8.3f  | %8.3f  | %8.3f\n', xdot_nom(2), xdot_test(2), xdot_test(2)-xdot_nom(2));
fprintf('   ψ_dot         | %8.3f  | %8.3f  | %8.3f\n', xdot_nom(3), xdot_test(3), xdot_test(3)-xdot_nom(3));
fprintf('   r_dot         | %8.3f  | %8.3f  | %8.3f\n', xdot_nom(4), xdot_test(4), xdot_test(4)-xdot_nom(4));
fprintf('   --------------------------------------------------------\n\n');

%% ========================================================================
%  7. SUMMARY AND RECOMMENDATIONS
%  ========================================================================

fprintf('\n========================================\n');
fprintf('DIAGNOSTIC SUMMARY\n');
fprintf('========================================\n\n');

fprintf('Key Findings:\n');
fprintf('  1. State dimensions: Nominal=%d, Test=%d %s\n', nomModel.n, testModel.n, ...
    ternary(nomModel.n==testModel.n, '✓', '✗ MISMATCH'));
fprintf('  2. Input dimensions: Nominal=%d, Test=%d %s\n', nomModel.m, testModel.m, ...
    ternary(nomModel.m==testModel.m, '✓', '✗ MISMATCH'));
fprintf('  3. GP dimensions: Nominal nz=%d nd=%d, Test nz=%d nd=%d\n', ...
    nomModel.nz, nomModel.nd, testModel.nz, testModel.nd);
fprintf('  4. Tire model: Nominal=Linear, Test=Pacejka\n');
fprintf('  5. Slip angle calculation: Nominal=Small-angle approx, Test=atan2\n\n');

fprintf('Potential Issues:\n');
if testModel.m ~= nomModel.m
    fprintf('  ⚠ INPUT DIMENSION MISMATCH: Test model expects 2 inputs [delta; T]\n');
    fprintf('    but tracking_true.m may be using only 1 input [delta]\n');
    fprintf('    → FIX: Update tracking_true.m to provide u=[delta; 0]\n\n');
end

if testModel.nd ~= nomModel.nd
    fprintf('  ⚠ GP OUTPUT DIMENSION MISMATCH: Test nd=%d vs Nominal nd=%d\n', testModel.nd, nomModel.nd);
    fprintf('    This affects how GP corrections are applied via Bd matrix\n');
    fprintf('    → REVIEW: Check Bd matrix compatibility in test model\n\n');
end

fprintf('  ⚠ SLIP ANGLE CALCULATION: Test uses atan2(), Nominal uses linear approx\n');
fprintf('    This causes different slip angle predictions\n');
fprintf('    → CONSIDER: Align slip angle calculations for consistency\n\n');

fprintf('  ⚠ TIRE SATURATION: Pacejka tires saturate, linear don''t\n');
fprintf('    MPC may command infeasible states in saturation region\n');
fprintf('    → CONSIDER: Add tire saturation constraints to MPC\n\n');

fprintf('========================================\n\n');

%% Helper functions
function str = check_match(a, b)
    if abs(a-b) < 1e-6
        str = '✓';
    else
        str = '✗';
    end
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
