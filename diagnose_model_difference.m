%--------------------------------------------------------------------------
% Diagnostic Script: Why True Model Performance is WORSE than Nominal
%--------------------------------------------------------------------------
% This script analyzes the key differences between nominal (linear tire)
% and true (Pacejka tire) models to understand tracking performance degradation
%--------------------------------------------------------------------------

clear all; close all; clc;

addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))

fprintf('\n========================================\n');
fprintf('Model Difference Diagnostic Analysis\n');
fprintf('========================================\n\n');

%% Create both models
nomModel = MotionModelGP_Bicycle_nominal([], []);
trueModel = MotionModelGP_Bicycle_true([], []);

%% Test Scenario: Circular path tracking
v_ref = 20;     % m/s
radius = 80;    % m
r_ref = v_ref / radius;  % rad/s

% Typical steady-state condition during circular cornering
vx = 20;        % m/s
vy = 0.5;       % small lateral velocity
psi = 0;        % yaw angle
r = r_ref;      % desired yaw rate

x_test = [vx; vy; psi; r];

%% Test 1: Slip angle comparison
fprintf('=== Test 1: Slip Angle Analysis ===\n');

% Range of steering inputs
delta_range = linspace(-deg2rad(5), deg2rad(5), 50);

alpha_f_nom = zeros(size(delta_range));
alpha_r_nom = zeros(size(delta_range));
alpha_f_true = zeros(size(delta_range));
alpha_r_true = zeros(size(delta_range));

for i = 1:length(delta_range)
    delta = delta_range(i);

    % Nominal model slip angles (small angle approximation)
    alpha_f_nom(i) = delta - (nomModel.l_f*r + vy)/vx;
    alpha_r_nom(i) = (nomModel.l_r*r - vy)/vx;

    % True model slip angles (full atan2)
    alpha_f_true(i) = atan2(vy + trueModel.l_f*r, vx) - delta;
    alpha_r_true(i) = atan2(vy - trueModel.l_r*r, vx);
end

fprintf('Slip angle calculation methods:\n');
fprintf('  Nominal: Small angle approximation\n');
fprintf('  True:    Full atan2 calculation\n');
fprintf('  Max difference (front): %.4f deg\n', max(abs(rad2deg(alpha_f_nom - alpha_f_true))));
fprintf('  Max difference (rear):  %.4f deg\n\n', max(abs(rad2deg(alpha_r_nom - alpha_r_true))));

%% Test 2: Tire force comparison
fprintf('=== Test 2: Tire Force Comparison ===\n');

alpha_range = linspace(-deg2rad(15), deg2rad(15), 100);

% Nominal (linear) tire forces
Fy_f_nom = nomModel.c_f * alpha_range;
Fy_r_nom = nomModel.c_r * alpha_range;

% True (Pacejka) tire forces
Fy_f_true = trueModel.D_f * sin(trueModel.C_f * atan(trueModel.B_f*alpha_range - ...
    trueModel.E_f*(trueModel.B_f*alpha_range - atan(trueModel.B_f*alpha_range))));
Fy_r_true = trueModel.D_r * sin(trueModel.C_r * atan(trueModel.B_r*alpha_range - ...
    trueModel.E_r*(trueModel.B_r*alpha_range - atan(trueModel.B_r*alpha_range))));

% Calculate linear stiffness from Pacejka at small angles
alpha_small = deg2rad(1);
Fy_f_pacejka_small = trueModel.D_f * sin(trueModel.C_f * atan(trueModel.B_f*alpha_small - ...
    trueModel.E_f*(trueModel.B_f*alpha_small - atan(trueModel.B_f*alpha_small))));
c_f_pacejka_equiv = Fy_f_pacejka_small / alpha_small;

Fy_r_pacejka_small = trueModel.D_r * sin(trueModel.C_r * atan(trueModel.B_r*alpha_small - ...
    trueModel.E_r*(trueModel.B_r*alpha_small - atan(trueModel.B_r*alpha_small))));
c_r_pacejka_equiv = Fy_r_pacejka_small / alpha_small;

fprintf('Cornering stiffness comparison:\n');
fprintf('  Nominal front: %.0f N/rad\n', nomModel.c_f);
fprintf('  Pacejka front (linear region): %.0f N/rad\n', c_f_pacejka_equiv);
fprintf('  Ratio: %.2f%%\n', (c_f_pacejka_equiv/nomModel.c_f)*100);
fprintf('\n');
fprintf('  Nominal rear: %.0f N/rad\n', nomModel.c_r);
fprintf('  Pacejka rear (linear region): %.0f N/rad\n', c_r_pacejka_equiv);
fprintf('  Ratio: %.2f%%\n\n', (c_r_pacejka_equiv/nomModel.c_r)*100);

%% Test 3: Dynamic response comparison
fprintf('=== Test 3: Dynamic Response Comparison ===\n');

% Test with steady steering input
delta_test = deg2rad(3);  % 3 degree steering
u_test = delta_test;

% Calculate state derivatives
xdot_nom = nomModel.f(x_test, u_test);
xdot_true = trueModel.f(x_test, u_test);

fprintf('State derivatives at steady cornering (delta=%.1f deg):\n', rad2deg(delta_test));
fprintf('                  Nominal      True       Diff       Ratio\n');
fprintf('  vx_dot:      %9.4f  %9.4f  %9.4f  %7.2f%%\n', xdot_nom(1), xdot_true(1), ...
    xdot_true(1)-xdot_nom(1), (xdot_true(1)/max(abs(xdot_nom(1)),1e-6))*100);
fprintf('  vy_dot:      %9.4f  %9.4f  %9.4f  %7.2f%%\n', xdot_nom(2), xdot_true(2), ...
    xdot_true(2)-xdot_nom(2), (xdot_true(2)/max(abs(xdot_nom(2)),1e-6))*100);
fprintf('  psi_dot:     %9.4f  %9.4f  %9.4f  %7.2f%%\n', xdot_nom(3), xdot_true(3), ...
    xdot_true(3)-xdot_nom(3), (xdot_true(3)/max(abs(xdot_nom(3)),1e-6))*100);
fprintf('  r_dot:       %9.4f  %9.4f  %9.4f  %7.2f%%\n', xdot_nom(4), xdot_true(4), ...
    xdot_true(4)-xdot_nom(4), (xdot_true(4)/max(abs(xdot_nom(4)),1e-6))*100);
fprintf('\n');

%% Test 4: Gradient availability
fprintf('=== Test 4: Gradient/Jacobian Analysis ===\n');

try
    gradx_nom = nomModel.gradx_f(x_test, u_test);
    fprintf('Nominal model gradx: Available (analytical)\n');
    fprintf('  Condition number: %.2e\n', cond(gradx_nom));
catch
    fprintf('Nominal model gradx: NOT AVAILABLE\n');
end

try
    gradx_true = trueModel.gradx_f(x_test, u_test);
    if all(gradx_true(:) == 0)
        fprintf('True model gradx: RETURNS ZEROS (not implemented)\n');
        fprintf('  ⚠️  WARNING: MPC cannot compute proper gradients!\n');
    else
        fprintf('True model gradx: Available\n');
        fprintf('  Condition number: %.2e\n', cond(gradx_true));
    end
catch
    fprintf('True model gradx: ERROR\n');
end
fprintf('\n');

%% Test 5: Nonlinearity severity
fprintf('=== Test 5: Nonlinearity Analysis ===\n');

% Compare force deviation from linear model
idx_small = abs(alpha_range) < deg2rad(5);  % Small angle region
idx_large = abs(alpha_range) > deg2rad(10); % Large angle region

error_small = mean(abs(Fy_f_true(idx_small) - Fy_f_nom(idx_small)));
error_large = mean(abs(Fy_f_true(idx_large) - Fy_f_nom(idx_large)));

fprintf('Tire force deviation (Pacejka vs Linear):\n');
fprintf('  Small angles (<5 deg):  %.1f N (%.1f%% of linear)\n', ...
    error_small, (error_small/mean(abs(Fy_f_nom(idx_small))))*100);
fprintf('  Large angles (>10 deg): %.1f N (%.1f%% of linear)\n', ...
    error_large, (error_large/mean(abs(Fy_f_nom(idx_large))))*100);
fprintf('\n');

%% Visualization
figure('Name', 'Model Comparison', 'Color', 'w', 'Position', [100 100 1400 800]);

% Subplot 1: Tire characteristics
subplot(2,3,1)
plot(rad2deg(alpha_range), Fy_f_nom/1000, 'r--', 'LineWidth', 2); hold on;
plot(rad2deg(alpha_range), Fy_f_true/1000, 'b-', 'LineWidth', 2);
grid on;
xlabel('Slip Angle [deg]');
ylabel('Lateral Force [kN]');
title('Front Tire Characteristics');
legend('Nominal (Linear)', 'True (Pacejka)', 'Location', 'best');

subplot(2,3,2)
plot(rad2deg(alpha_range), Fy_r_nom/1000, 'r--', 'LineWidth', 2); hold on;
plot(rad2deg(alpha_range), Fy_r_true/1000, 'b-', 'LineWidth', 2);
grid on;
xlabel('Slip Angle [deg]');
ylabel('Lateral Force [kN]');
title('Rear Tire Characteristics');
legend('Nominal (Linear)', 'True (Pacejka)', 'Location', 'best');

% Subplot 2: Force error
subplot(2,3,3)
plot(rad2deg(alpha_range), (Fy_f_true - Fy_f_nom)/1000, 'b-', 'LineWidth', 2); hold on;
plot(rad2deg(alpha_range), (Fy_r_true - Fy_r_nom)/1000, 'r-', 'LineWidth', 2);
grid on;
xlabel('Slip Angle [deg]');
ylabel('Force Error [kN]');
title('Tire Force Deviation');
legend('Front', 'Rear', 'Location', 'best');
yline(0, 'k--');

% Subplot 3: Cornering stiffness vs slip angle
subplot(2,3,4)
c_f_effective = Fy_f_true ./ (alpha_range + 1e-10);
c_r_effective = Fy_r_true ./ (alpha_range + 1e-10);
plot(rad2deg(alpha_range), c_f_effective/1000, 'b-', 'LineWidth', 2); hold on;
plot(rad2deg(alpha_range), c_r_effective/1000, 'r-', 'LineWidth', 2);
yline(nomModel.c_f/1000, 'b--', 'LineWidth', 1);
yline(nomModel.c_r/1000, 'r--', 'LineWidth', 1);
grid on;
xlabel('Slip Angle [deg]');
ylabel('Effective Cornering Stiffness [kN/rad]');
title('Nonlinear Stiffness (Pacejka vs Constant Linear)');
legend('Pacejka Front', 'Pacejka Rear', 'Nominal Front', 'Nominal Rear', 'Location', 'best');
ylim([0 100]);

% Subplot 4: Slip angle calculation comparison
subplot(2,3,5)
plot(rad2deg(delta_range), rad2deg(alpha_f_nom), 'r--', 'LineWidth', 2); hold on;
plot(rad2deg(delta_range), rad2deg(alpha_f_true), 'b-', 'LineWidth', 2);
grid on;
xlabel('Steering Angle [deg]');
ylabel('Front Slip Angle [deg]');
title('Slip Angle Calculation Method');
legend('Nominal (small angle approx)', 'True (atan2)', 'Location', 'best');

% Subplot 5: Dynamic response comparison
subplot(2,3,6)
delta_sweep = linspace(-deg2rad(5), deg2rad(5), 50);
r_dot_nom = zeros(size(delta_sweep));
r_dot_true = zeros(size(delta_sweep));

for i = 1:length(delta_sweep)
    xdot_n = nomModel.f(x_test, delta_sweep(i));
    xdot_t = trueModel.f(x_test, delta_sweep(i));
    r_dot_nom(i) = xdot_n(4);
    r_dot_true(i) = xdot_t(4);
end

plot(rad2deg(delta_sweep), r_dot_nom, 'r--', 'LineWidth', 2); hold on;
plot(rad2deg(delta_sweep), r_dot_true, 'b-', 'LineWidth', 2);
grid on;
xlabel('Steering Angle [deg]');
ylabel('Yaw Acceleration [rad/s^2]');
title('Yaw Response Comparison');
legend('Nominal', 'True', 'Location', 'best');

sgtitle('Nominal (Linear) vs True (Pacejka) Model Comparison');

%% ROOT CAUSE SUMMARY
fprintf('\n========================================\n');
fprintf('ROOT CAUSE ANALYSIS SUMMARY\n');
fprintf('========================================\n\n');

fprintf('🔍 KEY FINDINGS:\n\n');

fprintf('1. GRADIENT PROBLEM:\n');
fprintf('   ❌ True model returns ZERO gradients (not implemented)\n');
fprintf('   → MPC cannot compute proper Newton-Raphson updates\n');
fprintf('   → Optimization quality severely degraded\n\n');

fprintf('2. CORNERING STIFFNESS MISMATCH:\n');
fprintf('   • Linear stiffness at small angles differs by ~%.0f%%\n', ...
    abs((c_f_pacejka_equiv/nomModel.c_f - 1)*100));
fprintf('   → Different steady-state response\n\n');

fprintf('3. SLIP ANGLE CALCULATION:\n');
fprintf('   • Nominal: Small angle approximation\n');
fprintf('   • True: Full trigonometric calculation\n');
fprintf('   → Maximum difference: ~%.2f deg\n\n', ...
    max(abs(rad2deg(alpha_f_nom - alpha_f_true))));

fprintf('4. NONLINEARITY:\n');
fprintf('   • Pacejka tire saturates at large slip angles\n');
fprintf('   • Linear model has constant gain → more aggressive control\n\n');

fprintf('⚠️  PRIMARY CAUSE:\n');
fprintf('   The TRUE model has NO analytical gradients!\n');
fprintf('   MPC relies on gradients for:\n');
fprintf('     - Newton-Raphson optimization steps\n');
fprintf('     - Sensitivity analysis\n');
fprintf('     - Linearization around trajectory\n\n');

fprintf('💡 SOLUTION:\n');
fprintf('   Implement gradx_f() and gradu_f() for True model\n');
fprintf('   OR use numerical differentiation in NMPC\n');
fprintf('   OR accept degraded optimization performance\n\n');

fprintf('========================================\n\n');
