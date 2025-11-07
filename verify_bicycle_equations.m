%--------------------------------------------------------------------------
% Verify Bicycle Model Equations
% Compare with standard textbook formulation
%--------------------------------------------------------------------------

clear all; close all; clc;

fprintf('\n========================================\n');
fprintf('BICYCLE MODEL EQUATION VERIFICATION\n');
fprintf('========================================\n\n');

%% Standard Bicycle Model (from Rajamani "Vehicle Dynamics and Control")
fprintf('STANDARD BICYCLE MODEL (Rajamani)\n');
fprintf('===================================\n\n');

syms vx vy r delta real
syms M I_z l_f l_r c_f c_r real positive

fprintf('State: x = [vx; vy; r]\n');
fprintf('Input: u = delta\n\n');

% Slip angles (small angle approximation)
fprintf('Slip angles (linearized):\n');
alpha_f_sym = (vy + l_f*r)/vx - delta;
alpha_r_sym = (vy - l_r*r)/vx;

fprintf('  alpha_f = (vy + l_f*r)/vx - delta\n');
fprintf('  alpha_r = (vy - l_r*r)/vx\n\n');

% Lateral forces (linear tire model)
fprintf('Lateral forces:\n');
Fy_f_sym = -c_f * alpha_f_sym;  % NOTE: negative sign!
Fy_r_sym = -c_r * alpha_r_sym;  % NOTE: negative sign!

fprintf('  Fy_f = -c_f * alpha_f  (negative for SAE convention)\n');
fprintf('  Fy_r = -c_r * alpha_r\n\n');

% Dynamics
fprintf('Dynamics (SAE vehicle coordinates, z-up, x-forward):\n');
vx_dot_sym = r * vy;  % simplified (no longitudinal force)
vy_dot_sym = (1/M) * (Fy_f + Fy_r) - r*vx;
r_dot_sym = (1/I_z) * (l_f*Fy_f - l_r*Fy_r);

fprintf('  vx_dot = r * vy\n');
fprintf('  vy_dot = (1/M) * (Fy_f + Fy_r) - r*vx\n');
fprintf('  r_dot = (1/I_z) * (l_f*Fy_f - l_r*Fy_r)\n\n');

% Expand
fprintf('Expanded form:\n');
fprintf('  vx_dot = r * vy\n');
pretty(simplify(vy_dot_sym));
fprintf('\n');
pretty(simplify(r_dot_sym));
fprintf('\n\n');

% Linearization at equilibrium (vx=V, vy=0, r=0, delta=0)
fprintf('Linearization at (vx=V, vy=0, r=0, delta=0):\n');
fprintf('==============================================\n\n');

x_sym = [vx; vy; r];
xdot_sym = [vx_dot_sym; vy_dot_sym; r_dot_sym];

A_sym = jacobian(xdot_sym, x_sym);

fprintf('Jacobian A = df/dx:\n');
A_eval = subs(A_sym, [vy, r, delta], [0, 0, 0]);
pretty(A_eval);

fprintf('\n\nEvaluating at vx=10 m/s:\n');
A_num = double(subs(A_eval, {M, I_z, l_f, l_r, c_f, c_r, vx}, ...
    {500, 600, 0.9, 1.5, 35000, 35000, 10}));

disp(A_num);

eig_A = eig(A_num);
fprintf('Eigenvalues:\n');
fprintf('  %.4f\n', eig_A);

if all(real(eig_A) < 0)
    fprintf('  ✓ System is STABLE (all eigenvalues negative)\n\n');
else
    fprintf('  ⚠ System is UNSTABLE\n\n');
end

%% What's in the current implementation?
fprintf('========================================\n');
fprintf('CURRENT IMPLEMENTATION CHECK\n');
fprintf('========================================\n\n');

% Without negative sign in Fy
Fy_f_wrong = c_f * alpha_f_sym;  % WRONG: positive
Fy_r_wrong = c_r * alpha_r_sym;  % WRONG: positive

vy_dot_wrong = (1/M) * (Fy_f_wrong + Fy_r_wrong) - r*vx;
r_dot_wrong = (1/I_z) * (l_f*Fy_f_wrong - l_r*Fy_r_wrong);

xdot_wrong = [vx_dot_sym; vy_dot_wrong; r_dot_wrong];
A_wrong = jacobian(xdot_wrong, x_sym);

fprintf('If we FORGET the negative sign in Fy:\n');
fprintf('  Fy_f = +c_f * alpha_f  (WRONG)\n');
fprintf('  Fy_r = +c_r * alpha_r  (WRONG)\n\n');

A_wrong_eval = subs(A_wrong, [vy, r, delta], [0, 0, 0]);
A_wrong_num = double(subs(A_wrong_eval, {M, I_z, l_f, l_r, c_f, c_r, vx}, ...
    {500, 600, 0.9, 1.5, 35000, 35000, 10}));

fprintf('Jacobian A (WRONG):\n');
disp(A_wrong_num);

eig_A_wrong = eig(A_wrong_num);
fprintf('Eigenvalues (WRONG):\n');
fprintf('  %.4f\n', eig_A_wrong);

if any(real(eig_A_wrong) > 0)
    fprintf('  ⚠ System is UNSTABLE (positive eigenvalues)\n');
    fprintf('  → This matches our current problem!\n\n');
end

%% Summary
fprintf('========================================\n');
fprintf('CONCLUSION\n');
fprintf('========================================\n\n');

fprintf('The tire forces MUST include a NEGATIVE sign:\n\n');
fprintf('  CORRECT:   Fy_f = -c_f * alpha_f\n');
fprintf('  INCORRECT: Fy_f = +c_f * alpha_f (current implementation)\n\n');

fprintf('This is because:\n');
fprintf('  - Positive slip angle α → tire slips outward\n');
fprintf('  - Tire force must oppose this → force inward (negative)\n');
fprintf('  - SAE vehicle coordinates: y-axis points left\n');
fprintf('  - Restoring force convention requires negative sign\n\n');

fprintf('FIX: Change line 134-135 in MotionModelGP_Bicycle_nominal.m:\n');
fprintf('  FROM: Fy_f = c_f * alpha_f;\n');
fprintf('  TO:   Fy_f = -c_f * alpha_f;\n\n');
