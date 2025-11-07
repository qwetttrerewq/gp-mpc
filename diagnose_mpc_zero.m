%--------------------------------------------------------------------------
% Diagnostic Script: Why is MPC output always zero?
%--------------------------------------------------------------------------

clear all; close all; clc;

addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))

fprintf('\n========================================\n');
fprintf('MPC ZERO OUTPUT DIAGNOSTIC\n');
fprintf('========================================\n\n');

%% Setup
dt = 0.1;
N = 20;
radius = 30;
v_ref = 10;
r_ref = v_ref / radius;  % 0.333 rad/s

var_w = diag([(0.3)^2 (deg2rad(2))^2]);
nominalModel = MotionModelGP_Bicycle_nominal([], var_w);

n = nominalModel.n;
m = nominalModel.m;

fprintf('Reference values:\n');
fprintf('  v_ref = %.2f m/s\n', v_ref);
fprintf('  r_ref = %.4f rad/s (%.2f deg/s)\n\n', r_ref, rad2deg(r_ref));

%% Initial State Analysis
x0 = [v_ref; 0; 0];  % [10; 0; 0]

fprintf('Initial state:\n');
fprintf('  vx = %.2f m/s (target: %.2f)\n', x0(1), v_ref);
fprintf('  vy = %.2f m/s (target: 0)\n', x0(2));
fprintf('  r  = %.4f rad/s (target: %.4f)\n\n', x0(3), r_ref);

%% Test 1: What happens with u=0?
fprintf('TEST 1: Dynamics with u=0 (zero steering)\n');
fprintf('==========================================\n');

u_zero = 0;
[x_next_zero, ~] = nominalModel.xkp1(x0, zeros(n,n), u_zero, dt);

fprintf('Next state with u=0:\n');
fprintf('  vx: %.2f -> %.2f (change: %.4f)\n', x0(1), x_next_zero(1), x_next_zero(1)-x0(1));
fprintf('  vy: %.2f -> %.4f (change: %.4f)\n', x0(2), x_next_zero(2), x_next_zero(2)-x0(2));
fprintf('  r:  %.4f -> %.4f (change: %.4f)\n\n', x0(3), x_next_zero(3), x_next_zero(3)-x0(3));

%% Test 2: What happens with small positive steering?
fprintf('TEST 2: Dynamics with u=5deg (positive steering)\n');
fprintf('=================================================\n');

u_pos = deg2rad(5);
[x_next_pos, ~] = nominalModel.xkp1(x0, zeros(n,n), u_pos, dt);

fprintf('Next state with u=5deg:\n');
fprintf('  vx: %.2f -> %.2f (change: %.4f)\n', x0(1), x_next_pos(1), x_next_pos(1)-x0(1));
fprintf('  vy: %.2f -> %.4f (change: %.4f)\n', x0(2), x_next_pos(2), x_next_pos(2)-x0(2));
fprintf('  r:  %.4f -> %.4f (change: %.4f)\n\n', x0(3), x_next_pos(3), x_next_pos(3)-x0(3));

%% Test 3: Cost function evaluation
fprintf('TEST 3: Cost Function Analysis\n');
fprintf('===============================\n');

weights.q_vx = 10;
weights.q_vy = 5;
weights.q_r = 50;
weights.q_delta = 1;

% Cost with u=0
cost_zero = weights.q_vx * (x0(1) - v_ref)^2 + ...
            weights.q_vy * x0(2)^2 + ...
            weights.q_r * (x0(3) - r_ref)^2 + ...
            weights.q_delta * u_zero^2;

% Cost with u=5deg
cost_pos = weights.q_vx * (x0(1) - v_ref)^2 + ...
           weights.q_vy * x0(2)^2 + ...
           weights.q_r * (x0(3) - r_ref)^2 + ...
           weights.q_delta * u_pos^2;

fprintf('Stage cost at initial state:\n');
fprintf('  With u=0deg:   J = %.6f\n', cost_zero);
fprintf('  With u=5deg:   J = %.6f\n', cost_pos);
fprintf('  Difference:    ΔJ = %.6f\n\n', cost_pos - cost_zero);

% Breakdown
fprintf('Cost breakdown (u=0):\n');
fprintf('  Velocity tracking: %.6f (weight: %.1f)\n', weights.q_vx * (x0(1) - v_ref)^2, weights.q_vx);
fprintf('  Lateral velocity:  %.6f (weight: %.1f)\n', weights.q_vy * x0(2)^2, weights.q_vy);
fprintf('  Yaw rate tracking: %.6f (weight: %.1f) *** LARGE ERROR ***\n', weights.q_r * (x0(3) - r_ref)^2, weights.q_r);
fprintf('  Input effort:      %.6f (weight: %.1f)\n\n', weights.q_delta * u_zero^2, weights.q_delta);

%% Test 4: Check if steering can generate yaw rate
fprintf('TEST 4: Can steering generate the required yaw rate?\n');
fprintf('====================================================\n');

% At steady-state circular motion, approximate required steering angle
% Using kinematic bicycle model: delta_ss ≈ L * r / vx
L = nominalModel.l_f + nominalModel.l_r;
delta_kinematic = L * r_ref / v_ref;

fprintf('Kinematic steady-state steering:\n');
fprintf('  Wheelbase L = %.2f m\n', L);
fprintf('  Required δ ≈ %.4f rad (%.2f deg)\n\n', delta_kinematic, rad2deg(delta_kinematic));

% Simulate with this steering for multiple steps
x_test = x0;
u_test = delta_kinematic;
fprintf('Simulating with δ = %.2f deg for 5 steps:\n', rad2deg(u_test));
for i = 1:5
    [x_test, ~] = nominalModel.xkp1(x_test, zeros(n,n), u_test, dt);
    fprintf('  Step %d: vx=%.2f, vy=%.4f, r=%.4f (target: %.4f)\n', ...
        i, x_test(1), x_test(2), x_test(3), r_ref);
end

%% Test 5: What does MPC think the cost will be?
fprintf('\nTEST 5: Predicted Cost Trajectory\n');
fprintf('==================================\n');

% Predict cost over horizon with u=0
total_cost_zero = 0;
x_pred = x0;
for i = 1:N
    cost_i = weights.q_vx * (x_pred(1) - v_ref)^2 + ...
             weights.q_vy * x_pred(2)^2 + ...
             weights.q_r * (x_pred(3) - r_ref)^2;
    total_cost_zero = total_cost_zero + cost_i;
    [x_pred, ~] = nominalModel.xkp1(x_pred, zeros(n,n), 0, dt);
end

fprintf('Total predicted cost (u=0 over %d steps): %.4f\n', N, total_cost_zero);
fprintf('  Average per step: %.4f\n', total_cost_zero/N);
fprintf('  Dominated by yaw rate error: %.4f\n\n', weights.q_r * (r_ref)^2);

%% Test 6: Direct dynamics check
fprintf('TEST 6: Dynamics Function Direct Call\n');
fprintf('======================================\n');

xdot_zero = nominalModel.f(x0, 0);
xdot_pos = nominalModel.f(x0, u_pos);

fprintf('Continuous dynamics (xdot) at x0:\n');
fprintf('  With u=0:    [%.4f, %.4f, %.4f]\n', xdot_zero);
fprintf('  With u=5deg: [%.4f, %.4f, %.4f]\n\n', xdot_pos);

%% DIAGNOSIS SUMMARY
fprintf('========================================\n');
fprintf('DIAGNOSIS SUMMARY\n');
fprintf('========================================\n\n');

fprintf('PROBLEM IDENTIFIED:\n');
fprintf('  Initial state: r = %.4f rad/s\n', x0(3));
fprintf('  Target state:  r = %.4f rad/s\n', r_ref);
fprintf('  Error: %.4f rad/s (%.2f deg/s)\n\n', r_ref - x0(3), rad2deg(r_ref - x0(3)));

fprintf('The vehicle starts with ZERO yaw rate but needs %.2f deg/s!\n\n', rad2deg(r_ref));

fprintf('MPC returns u=0 because:\n');
fprintf('  1. Initial state has r=0, but target is r=%.4f\n', r_ref);
fprintf('  2. With u=0, the vehicle STAYS at r=0 (no yaw acceleration)\n');
fprintf('  3. Cost increases over time due to r tracking error\n');
fprintf('  4. BUT: Cost gradient may be flat or optimization fails\n\n');

fprintf('POSSIBLE CAUSES:\n');
fprintf('  A. Cost function gradient is too small (bad scaling)\n');
fprintf('  B. Optimizer tolerance too loose (tol=1e-2)\n');
fprintf('  C. Initial guess is u=0 and optimizer gets stuck\n');
fprintf('  D. Dynamics Jacobian is incorrect or missing\n');
fprintf('  E. Constraint is preventing nonzero solution\n\n');

fprintf('RECOMMENDED FIXES:\n');
fprintf('  1. Increase q_r weight (currently %.1f) → try 500 or 1000\n', weights.q_r);
fprintf('  2. Decrease optimizer tolerance → try 1e-4 or 1e-6\n');
fprintf('  3. Provide better initial guess → try u_guess = %.2f deg\n', rad2deg(delta_kinematic));
fprintf('  4. Check if Jacobian functions exist (bicycle_gradx_f, bicycle_gradu_f)\n');
fprintf('  5. Simplify/remove sideslip constraint temporarily\n\n');
