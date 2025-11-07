function stats = validate_GP_prediction(d_GP, trueModel, nomModel, test_data)
%--------------------------------------------------------------------------
% Validate GP Prediction Performance
%
% Evaluates how well "nominal + GP residual" approximates true tire forces
%
% Inputs:
%   d_GP       - Trained GP object
%   trueModel  - True model (Pacejka tires)
%   nomModel   - Nominal model (Linear tires)
%   test_data  - Struct with fields:
%                .x: [n x K] state trajectory
%                .u: [m x K] input trajectory
%
% Outputs:
%   stats      - Struct with validation statistics:
%                .rmse_front: RMSE for front tire force [N]
%                .rmse_rear:  RMSE for rear tire force [N]
%                .coverage_front: 2-sigma coverage rate [%]
%                .coverage_rear:  2-sigma coverage rate [%]
%                .max_error_front: Maximum error [N]
%                .max_error_rear:  Maximum error [N]
%--------------------------------------------------------------------------

fprintf('\n========================================\n');
fprintf('GP Prediction Validation\n');
fprintf('========================================\n\n');

% Extract test data
x_test = test_data.x;
u_test = test_data.u;
K = size(u_test, 2);  % Use u_test size (control inputs, not states)

% Storage
F_y_f_true = zeros(1, K);
F_y_r_true = zeros(1, K);
F_y_f_nom = zeros(1, K);
F_y_r_nom = zeros(1, K);
F_y_f_gp_augmented = zeros(1, K);
F_y_r_gp_augmented = zeros(1, K);
sigma_f = zeros(1, K);
sigma_r = zeros(1, K);

fprintf('Computing tire forces for %d test points...\n', K);

for k = 1:K
    % True tire forces (Pacejka)
    [F_y_f_true(k), F_y_r_true(k)] = trueModel.getTireForces(x_test(:,k), u_test(:,k));

    % Nominal tire forces (Linear)
    [F_y_f_nom(k), F_y_r_nom(k)] = nomModel.getTireForces(x_test(:,k), u_test(:,k));

    % GP input: z = [vy; r; delta]
    z_k = [nomModel.Bz_x * x_test(:,k); nomModel.Bz_u * u_test(:,k)];

    % GP prediction of residual
    [mu_d, var_d] = d_GP.eval(z_k, false);

    % GP-augmented prediction: Nominal + GP residual
    F_y_f_gp_augmented(k) = F_y_f_nom(k) + mu_d(1);
    F_y_r_gp_augmented(k) = F_y_r_nom(k) + mu_d(2);

    % GP uncertainty (standard deviation)
    sigma_f(k) = sqrt(var_d(1));
    sigma_r(k) = sqrt(var_d(2));
end

fprintf('  Done!\n\n');

%--------------------------------------------------------------------------
% Compute Prediction Errors
%--------------------------------------------------------------------------
% Error WITHOUT GP (nominal only)
error_nom_f = F_y_f_true - F_y_f_nom;
error_nom_r = F_y_r_true - F_y_r_nom;

% Error WITH GP (nominal + GP)
error_gp_f = F_y_f_true - F_y_f_gp_augmented;
error_gp_r = F_y_r_true - F_y_r_gp_augmented;

%--------------------------------------------------------------------------
% Statistics
%--------------------------------------------------------------------------
% RMSE
stats.rmse_nom_front = sqrt(mean(error_nom_f.^2));
stats.rmse_nom_rear = sqrt(mean(error_nom_r.^2));
stats.rmse_gp_front = sqrt(mean(error_gp_f.^2));
stats.rmse_gp_rear = sqrt(mean(error_gp_r.^2));

% Maximum absolute error
stats.max_error_nom_front = max(abs(error_nom_f));
stats.max_error_nom_rear = max(abs(error_nom_r));
stats.max_error_gp_front = max(abs(error_gp_f));
stats.max_error_gp_rear = max(abs(error_gp_r));

% 2-sigma coverage rate (95% confidence interval)
in_interval_f = abs(error_gp_f) <= 2 * sigma_f;
in_interval_r = abs(error_gp_r) <= 2 * sigma_r;
stats.coverage_front = mean(in_interval_f) * 100;
stats.coverage_rear = mean(in_interval_r) * 100;

% Improvement percentage
stats.improvement_front = (1 - stats.rmse_gp_front / stats.rmse_nom_front) * 100;
stats.improvement_rear = (1 - stats.rmse_gp_rear / stats.rmse_nom_rear) * 100;

%--------------------------------------------------------------------------
% Print Statistics
%--------------------------------------------------------------------------
fprintf('VALIDATION RESULTS:\n');
fprintf('------------------------------------------\n');
fprintf('Front Tire:\n');
fprintf('  RMSE (Nominal only):  %.2f N\n', stats.rmse_nom_front);
fprintf('  RMSE (Nominal + GP):  %.2f N\n', stats.rmse_gp_front);
fprintf('  Improvement:          %.1f%%\n', stats.improvement_front);
fprintf('  Max Error (GP):       %.2f N\n', stats.max_error_gp_front);
fprintf('  2-sigma Coverage:     %.1f%% (target: ~95%%)\n', stats.coverage_front);
fprintf('\n');
fprintf('Rear Tire:\n');
fprintf('  RMSE (Nominal only):  %.2f N\n', stats.rmse_nom_rear);
fprintf('  RMSE (Nominal + GP):  %.2f N\n', stats.rmse_gp_rear);
fprintf('  Improvement:          %.1f%%\n', stats.improvement_rear);
fprintf('  Max Error (GP):       %.2f N\n', stats.max_error_gp_rear);
fprintf('  2-sigma Coverage:     %.1f%% (target: ~95%%)\n', stats.coverage_rear);
fprintf('------------------------------------------\n\n');

%--------------------------------------------------------------------------
% Visualization
%--------------------------------------------------------------------------
fprintf('Generating validation plots...\n');

figure('Name', 'GP Prediction Validation', 'Color', 'w', 'Position', [100 100 1400 900]);

% --- Front Tire Force Comparison ---
subplot(3,2,1)
plot(F_y_f_true, 'b-', 'LineWidth', 2, 'DisplayName', 'True (Pacejka)'); hold on;
plot(F_y_f_nom, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nominal (Linear)');
plot(F_y_f_gp_augmented, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Nominal + GP');
grid on;
xlabel('Sample Index');
ylabel('F_{y,f} [N]');
title('Front Tire Lateral Force');
legend('Location', 'best');

% --- Rear Tire Force Comparison ---
subplot(3,2,2)
plot(F_y_r_true, 'b-', 'LineWidth', 2, 'DisplayName', 'True (Pacejka)'); hold on;
plot(F_y_r_nom, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nominal (Linear)');
plot(F_y_r_gp_augmented, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Nominal + GP');
grid on;
xlabel('Sample Index');
ylabel('F_{y,r} [N]');
title('Rear Tire Lateral Force');
legend('Location', 'best');

% --- Front Tire Error with Uncertainty ---
subplot(3,2,3)
plot(error_nom_f, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Error (Nominal)'); hold on;
plot(error_gp_f, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Error (Nominal + GP)');
plot(2*sigma_f, 'k--', 'LineWidth', 1, 'DisplayName', '2\sigma bounds');
plot(-2*sigma_f, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
grid on;
xlabel('Sample Index');
ylabel('Prediction Error [N]');
title(sprintf('Front Tire Error (RMSE: %.2f → %.2f N)', stats.rmse_nom_front, stats.rmse_gp_front));
legend('Location', 'best');

% --- Rear Tire Error with Uncertainty ---
subplot(3,2,4)
plot(error_nom_r, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Error (Nominal)'); hold on;
plot(error_gp_r, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Error (Nominal + GP)');
plot(2*sigma_r, 'k--', 'LineWidth', 1, 'DisplayName', '2\sigma bounds');
plot(-2*sigma_r, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
grid on;
xlabel('Sample Index');
ylabel('Prediction Error [N]');
title(sprintf('Rear Tire Error (RMSE: %.2f → %.2f N)', stats.rmse_nom_rear, stats.rmse_gp_rear));
legend('Location', 'best');

% --- Front Tire Error Histogram ---
subplot(3,2,5)
histogram(error_nom_f, 30, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Nominal'); hold on;
histogram(error_gp_f, 30, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Nominal + GP');
grid on;
xlabel('Prediction Error [N]');
ylabel('Probability Density');
title(sprintf('Front Tire Error Distribution (Improv: %.1f%%)', stats.improvement_front));
legend('Location', 'best');

% --- Rear Tire Error Histogram ---
subplot(3,2,6)
histogram(error_nom_r, 30, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'Nominal'); hold on;
histogram(error_gp_r, 30, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Nominal + GP');
grid on;
xlabel('Prediction Error [N]');
ylabel('Probability Density');
title(sprintf('Rear Tire Error Distribution (Improv: %.1f%%)', stats.improvement_rear));
legend('Location', 'best');

sgtitle('GP-Augmented Model Validation: True vs (Nominal + GP)');

fprintf('  Done!\n');
fprintf('========================================\n\n');

end
