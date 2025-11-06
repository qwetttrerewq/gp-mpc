%==========================================================================
% Regenerate Jacobian Functions for 4-State Bicycle Model
%
% This script regenerates the analytical Jacobian functions required by
% MotionModelGP_Bicycle_nominal for the updated 4-state system:
%   x = [vx; vy; psi; r]
%
% The generated functions will be saved in the CODEGEN folder.
%==========================================================================

clear all; close all; clc;

% Add paths
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'classes'))
addpath(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', 'functions'))

fprintf('=======================================================\n');
fprintf('Regenerating Jacobian Functions for 4-State System\n');
fprintf('=======================================================\n\n');

% Create nominal model instance
nomModel = MotionModelGP_Bicycle_nominal([], []);

% Generate Jacobian functions
fprintf('Starting Jacobian generation...\n');
nomModel.generate_grad_functions();

fprintf('\n=======================================================\n');
fprintf('Jacobian generation complete!\n');
fprintf('=======================================================\n');
fprintf('\nGenerated files:\n');
fprintf('  - CODEGEN/bicycle_gradx_f.m\n');
fprintf('  - CODEGEN/bicycle_gradu_f.m\n\n');

fprintf('You can now run main_cornering.m with the updated models.\n');
