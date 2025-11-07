%--------------------------------------------------------------------------
% Scenario 1 Analysis: Compare Reference vs True vs Nominal
%
% Purpose: Analyze tracking performance for Scenario 1 (GP OFF)
%          Compare how well out.x and out.xnom follow the reference track
%
% Prerequisites: Run main_singletrack.m first with useGP=false
%--------------------------------------------------------------------------

clear all; close all; clc;

fprintf('\n========================================\n');
fprintf('Scenario 1 Analysis: Reference Tracking\n');
fprintf('========================================\n\n');

%% Load simulation results
result_file = fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', ...
    'simresults', 'scenario1_results_gp.mat');

if ~exist(result_file, 'file')
    error(['Simulation results not found!\n' ...
        'Please run main_singletrack.m first and save results:\n' ...
        '  save(''simresults/scenario1_results_gp.mat'', ''out'', ''track'')']);
end

fprintf('Loading simulation results...\n');
load(result_file, 'out', 'track');
fprintf('  Loaded successfully\n\n');

%% Find valid data range
k_valid = find(~isnan(out.x(1,:)), 1, 'last');
fprintf('Valid time steps: %d (%.2f seconds)\n', k_valid, out.t(k_valid));

% State indices (7-state model)
% [1] I_x = x position [m]
% [2] I_y = y position [m]
% [3] psi = yaw angle [rad]
% [4] V_vx = longitudinal velocity [m/s]
% [5] V_vy = lateral velocity [m/s]
% [6] psidot = yaw rate [rad/s]
% [7] track_dist = distance traveled along track centerline [m]

%% Compute tracking errors for TRUE model (out.x)
fprintf('\nComputing tracking errors for TRUE model...\n');

lag_error_true = zeros(1, k_valid);
contour_error_true = zeros(1, k_valid);
orientation_error_true = zeros(1, k_valid);

for k = 1:k_valid
    I_x = out.x(1, k);
    I_y = out.x(2, k);
    psi = out.x(3, k);
    track_dist = out.x(7, k);
    
    [lag_error_true(k), contour_error_true(k), ~, orientation_error_true(k)] = ...
        track.getVehicleDeviation([I_x; I_y], psi, track_dist);
end

%% Compute tracking errors for NOMINAL model (out.xnom)
fprintf('Computing tracking errors for NOMINAL model...\n');

lag_error_nom = zeros(1, k_valid);
contour_error_nom = zeros(1, k_valid);
orientation_error_nom = zeros(1, k_valid);

for k = 1:k_valid
    I_x = out.xnom(1, k);
    I_y = out.xnom(2, k);
    psi = out.xnom(3, k);
    track_dist = out.xnom(7, k);
    
    [lag_error_nom(k), contour_error_nom(k), ~, orientation_error_nom(k)] = ...
        track.getVehicleDeviation([I_x; I_y], psi, track_dist);
end

%% Compute Model Mismatch (TRUE vs NOMINAL prediction)
fprintf('\n=== MODEL MISMATCH ANALYSIS ===\n\n');

% State prediction error (model mismatch)
state_mismatch = out.x - out.xnom;  % [7 × k_valid+1]
mismatch_norm = vecnorm(state_mismatch(:, 1:k_valid));

fprintf('STATE PREDICTION MISMATCH (TRUE - NOMINAL):\n');
fprintf('  Position (x,y) RMS:  %.4f m\n', rms(vecnorm(state_mismatch(1:2, 1:k_valid))));
fprintf('  Velocity (vx,vy) RMS: %.4f m/s\n', rms(vecnorm(state_mismatch(4:5, 1:k_valid))));
fprintf('  Yaw angle RMS:       %.4f rad (%.2f deg)\n', rms(state_mismatch(3, 1:k_valid)), ...
    rad2deg(rms(state_mismatch(3, 1:k_valid))));
fprintf('  Overall mismatch RMS: %.4f\n', rms(mismatch_norm));
fprintf('  Max mismatch:         %.4f\n\n', max(mismatch_norm));

fprintf('INTERPRETATION:\n');
fprintf('  - NOMINAL model predictions (out.xnom) match MPC expectations perfectly\n');
fprintf('  - TRUE model (out.x) shows mismatch because MPC was designed for NOMINAL\n');
fprintf('  - The mismatch above represents the modeling error GP needs to correct\n\n');

%% Compute statistics
fprintf('\n=== TRACKING PERFORMANCE STATISTICS ===\n\n');

% Lag Error
lag_rms_true = rms(lag_error_true);
lag_rms_nom = rms(lag_error_nom);
lag_max_true = max(abs(lag_error_true));
lag_max_nom = max(abs(lag_error_nom));

fprintf('LAG ERROR (progress along track):\n');
fprintf('  TRUE model:\n');
fprintf('    RMS:  %.4f m\n', lag_rms_true);
fprintf('    Max:  %.4f m\n', lag_max_true);
fprintf('  NOMINAL model:\n');
fprintf('    RMS:  %.4f m\n', lag_rms_nom);
fprintf('    Max:  %.4f m\n', lag_max_nom);
fprintf('  Difference (Nominal better by):\n');
fprintf('    RMS:  %.4f m (%.2f%%)\n', lag_rms_true - lag_rms_nom, ...
    (lag_rms_true/lag_rms_nom - 1)*100);
fprintf('\n');

% Contour Error
contour_rms_true = rms(contour_error_true);
contour_rms_nom = rms(contour_error_nom);
contour_max_true = max(abs(contour_error_true));
contour_max_nom = max(abs(contour_error_nom));

fprintf('CONTOUR ERROR (lateral deviation from centerline):\n');
fprintf('  TRUE model:\n');
fprintf('    RMS:  %.4f m\n', contour_rms_true);
fprintf('    Max:  %.4f m\n', contour_max_true);
fprintf('  NOMINAL model:\n');
fprintf('    RMS:  %.4f m\n', contour_rms_nom);
fprintf('    Max:  %.4f m\n', contour_max_nom);
fprintf('  Difference (Nominal better by):\n');
fprintf('    RMS:  %.4f m (%.2f%%)\n', contour_rms_true - contour_rms_nom, ...
    (contour_rms_true/contour_rms_nom - 1)*100);
fprintf('\n');

% Orientation Error
orient_rms_true = rms(orientation_error_true);
orient_rms_nom = rms(orientation_error_nom);
orient_max_true = max(abs(orientation_error_true));
orient_max_nom = max(abs(orientation_error_nom));

fprintf('ORIENTATION ERROR (heading deviation):\n');
fprintf('  TRUE model:\n');
fprintf('    RMS:  %.4f rad (%.2f deg)\n', orient_rms_true, rad2deg(orient_rms_true));
fprintf('    Max:  %.4f rad (%.2f deg)\n', orient_max_true, rad2deg(orient_max_true));
fprintf('  NOMINAL model:\n');
fprintf('    RMS:  %.4f rad (%.2f deg)\n', orient_rms_nom, rad2deg(orient_rms_nom));
fprintf('    Max:  %.4f rad (%.2f deg)\n', orient_max_nom, rad2deg(orient_max_nom));
fprintf('  Difference (Nominal better by):\n');
fprintf('    RMS:  %.4f rad (%.2f deg, %.2f%%)\n', orient_rms_true - orient_rms_nom, ...
    rad2deg(orient_rms_true - orient_rms_nom), (orient_rms_true/orient_rms_nom - 1)*100);
fprintf('\n');

% Track distance traveled
dist_true = out.x(7, k_valid);
dist_nom = out.xnom(7, k_valid);

fprintf('TRACK DISTANCE TRAVELED:\n');
fprintf('  TRUE model:    %.2f m\n', dist_true);
fprintf('  NOMINAL model: %.2f m\n', dist_nom);
fprintf('  Difference:    %.2f m\n\n', dist_true - dist_nom);

%% Visualization
fprintf('Generating comparison plots...\n\n');

% -------------------------------------------------------------------------
% Figure 1: Tracking Errors Over Time
% -------------------------------------------------------------------------
fig1 = figure('Name', 'Tracking Errors', 'Color', 'w', 'Position', [100 100 1400 900]);

% Lag Error
subplot(3,1,1)
plot(out.t(1:k_valid), lag_error_true, 'b-', 'LineWidth', 2); hold on;
plot(out.t(1:k_valid), lag_error_nom, 'r--', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]');
ylabel('Lag Error [m]');
title('Lag Error (Progress Along Track)');
legend('TRUE model (Pacejka)', 'NOMINAL model (Linear)', 'Location', 'best');

% Contour Error
subplot(3,1,2)
plot(out.t(1:k_valid), contour_error_true, 'b-', 'LineWidth', 2); hold on;
plot(out.t(1:k_valid), contour_error_nom, 'r--', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]');
ylabel('Contour Error [m]');
title('Contour Error (Lateral Deviation from Centerline)');
legend('TRUE model (Pacejka)', 'NOMINAL model (Linear)', 'Location', 'best');

% Orientation Error
subplot(3,1,3)
plot(out.t(1:k_valid), rad2deg(orientation_error_true), 'b-', 'LineWidth', 2); hold on;
plot(out.t(1:k_valid), rad2deg(orientation_error_nom), 'r--', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]');
ylabel('Orientation Error [deg]');
title('Orientation Error (Heading Deviation)');
legend('TRUE model (Pacejka)', 'NOMINAL model (Linear)', 'Location', 'best');

sgtitle('Scenario 1: Reference Tracking Errors (GP OFF)', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% Figure 2: Error Distributions
% -------------------------------------------------------------------------
fig2 = figure('Name', 'Error Distributions', 'Color', 'w', 'Position', [150 150 1400 500]);

% Lag Error Distribution
subplot(1,3,1)
histogram(lag_error_true, 30, 'Normalization', 'probability', ...
    'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none'); hold on;
histogram(lag_error_nom, 30, 'Normalization', 'probability', ...
    'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
xline(0, 'k--', 'LineWidth', 1);
grid on;
xlabel('Lag Error [m]');
ylabel('Probability');
title('Lag Error Distribution');
legend('TRUE', 'NOMINAL', 'Location', 'best');

% Contour Error Distribution
subplot(1,3,2)
histogram(contour_error_true, 30, 'Normalization', 'probability', ...
    'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none'); hold on;
histogram(contour_error_nom, 30, 'Normalization', 'probability', ...
    'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
xline(0, 'k--', 'LineWidth', 1);
grid on;
xlabel('Contour Error [m]');
ylabel('Probability');
title('Contour Error Distribution');
legend('TRUE', 'NOMINAL', 'Location', 'best');

% Orientation Error Distribution
subplot(1,3,3)
histogram(rad2deg(orientation_error_true), 30, 'Normalization', 'probability', ...
    'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'none'); hold on;
histogram(rad2deg(orientation_error_nom), 30, 'Normalization', 'probability', ...
    'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'none');
xline(0, 'k--', 'LineWidth', 1);
grid on;
xlabel('Orientation Error [deg]');
ylabel('Probability');
title('Orientation Error Distribution');
legend('TRUE', 'NOMINAL', 'Location', 'best');

sgtitle('Error Distributions', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% Figure 3: XY Trajectory Comparison
% -------------------------------------------------------------------------
fig3 = figure('Name', 'Trajectory Comparison', 'Color', 'w', 'Position', [200 200 1000 800]);

% Plot trajectories
plot(out.x(1, 1:k_valid), out.x(2, 1:k_valid), 'b-', 'LineWidth', 2); hold on;
plot(out.xnom(1, 1:k_valid), out.xnom(2, 1:k_valid), 'r--', 'LineWidth', 1.5);

% Try to plot track centerline (if available)
try
    % Get track reference points
    track_dists = linspace(0, max(out.x(7, :)), 200);
    track_x = zeros(1, length(track_dists));
    track_y = zeros(1, length(track_dists));
    for i = 1:length(track_dists)
        [track_x(i), track_y(i)] = track.getTrackInfo(track_dists(i));
    end
    plot(track_x, track_y, 'k--', 'LineWidth', 1, 'DisplayName', 'Track Centerline');
catch
    % If track centerline plotting fails, skip it
end

grid on;
axis equal;
xlabel('X [m]');
ylabel('Y [m]');
title('Vehicle Trajectory in Global Frame');
legend('TRUE model (Pacejka)', 'NOMINAL model (Linear)', 'Location', 'best');

% Mark start and end
plot(out.x(1, 1), out.x(2, 1), 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
plot(out.x(1, k_valid), out.x(2, k_valid), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'DisplayName', 'End');

sgtitle('XY Trajectory Comparison', 'FontSize', 14, 'FontWeight', 'bold');

% -------------------------------------------------------------------------
% Figure 4: Track Distance Progress
% -------------------------------------------------------------------------
fig4 = figure('Name', 'Track Progress', 'Color', 'w', 'Position', [250 250 1000 400]);

plot(out.t(1:k_valid), out.x(7, 1:k_valid), 'b-', 'LineWidth', 2); hold on;
plot(out.t(1:k_valid), out.xnom(7, 1:k_valid), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('Track Distance [m]');
title('Progress Along Track Centerline');
legend('TRUE model (Pacejka)', 'NOMINAL model (Linear)', 'Location', 'best');

fprintf('========================================\n');
fprintf('Analysis complete!\n');
fprintf('========================================\n\n');

%% Save analysis results
analysis_results = struct();
analysis_results.lag_error_true = lag_error_true;
analysis_results.lag_error_nom = lag_error_nom;
analysis_results.contour_error_true = contour_error_true;
analysis_results.contour_error_nom = contour_error_nom;
analysis_results.orientation_error_true = orientation_error_true;
analysis_results.orientation_error_nom = orientation_error_nom;
analysis_results.lag_rms_true = lag_rms_true;
analysis_results.lag_rms_nom = lag_rms_nom;
analysis_results.contour_rms_true = contour_rms_true;
analysis_results.contour_rms_nom = contour_rms_nom;
analysis_results.orient_rms_true = orient_rms_true;
analysis_results.orient_rms_nom = orient_rms_nom;

save(fullfile(pwd, 'Gaussian-Process-based-Model-Predictive-Control', ...
    'simresults', 'scenario1_analysis.mat'), 'analysis_results');
fprintf('Analysis results saved to simresults/scenario1_analysis.mat\n\n');
