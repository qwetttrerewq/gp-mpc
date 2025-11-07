function stats = analyze_simulation_gp_performance(out, trueModel, nomModel)
%--------------------------------------------------------------------------
% Analyze GP Performance During MPC Simulation
%
% Evaluates how well GP captured the model mismatch during closed-loop MPC
%
% Inputs:
%   out        - Simulation output struct from main_cornering_gp.m
%   trueModel  - True model (Pacejka tires)
%   nomModel   - Nominal model (Linear tires)
%
% Outputs:
%   stats      - Performance statistics struct
%--------------------------------------------------------------------------

fprintf('\n========================================\n');
fprintf('GP Performance Analysis (MPC Simulation)\n');
fprintf('========================================\n\n');

% Extract data (excluding invalid time steps)
k_valid = find(~isnan(out.u(1,:)), 1, 'last');
x_data = out.xhat(:, 1:k_valid);
u_data = out.u(:, 1:k_valid);
d_est = out.d_est(:, 1:k_valid);    % True residual (measured)
d_gp = out.d_gp(:, 1:k_valid);      % GP predicted residual

K = k_valid;

fprintf('Analyzing %d time steps...\n', K);

%--------------------------------------------------------------------------
% Compute True Tire Forces
%--------------------------------------------------------------------------
F_y_f_true = zeros(1, K);
F_y_r_true = zeros(1, K);
F_y_f_nom = zeros(1, K);
F_y_r_nom = zeros(1, K);

for k = 1:K
    [F_y_f_true(k), F_y_r_true(k)] = trueModel.getTireForces(x_data(:,k), u_data(:,k));
    [F_y_f_nom(k), F_y_r_nom(k)] = nomModel.getTireForces(x_data(:,k), u_data(:,k));
end

% GP-augmented prediction
F_y_f_gp_aug = F_y_f_nom + d_gp(1,:);
F_y_r_gp_aug = F_y_r_nom + d_gp(2,:);

%--------------------------------------------------------------------------
% Prediction Errors
%--------------------------------------------------------------------------
% Error WITHOUT GP (nominal only)
error_nom_f = F_y_f_true - F_y_f_nom;
error_nom_r = F_y_r_true - F_y_r_nom;

% Error WITH GP (nominal + GP)
error_gp_f = F_y_f_true - F_y_f_gp_aug;
error_gp_r = F_y_r_true - F_y_r_gp_aug;

% GP residual prediction error
error_res_f = d_est(1,:) - d_gp(1,:);
error_res_r = d_est(2,:) - d_gp(2,:);

%--------------------------------------------------------------------------
% Statistics
%--------------------------------------------------------------------------
stats.rmse_nom_f = sqrt(mean(error_nom_f.^2));
stats.rmse_nom_r = sqrt(mean(error_nom_r.^2));
stats.rmse_gp_f = sqrt(mean(error_gp_f.^2));
stats.rmse_gp_r = sqrt(mean(error_gp_r.^2));
stats.rmse_res_f = sqrt(mean(error_res_f.^2));
stats.rmse_res_r = sqrt(mean(error_res_r.^2));

stats.max_error_nom_f = max(abs(error_nom_f));
stats.max_error_nom_r = max(abs(error_nom_r));
stats.max_error_gp_f = max(abs(error_gp_f));
stats.max_error_gp_r = max(abs(error_gp_r));

stats.improvement_f = (1 - stats.rmse_gp_f / stats.rmse_nom_f) * 100;
stats.improvement_r = (1 - stats.rmse_gp_r / stats.rmse_nom_r) * 100;

% Mean absolute percentage error (relative to nominal force magnitude)
stats.mape_nom_f = mean(abs(error_nom_f ./ (F_y_f_nom + eps))) * 100;
stats.mape_nom_r = mean(abs(error_nom_r ./ (F_y_r_nom + eps))) * 100;
stats.mape_gp_f = mean(abs(error_gp_f ./ (F_y_f_nom + eps))) * 100;
stats.mape_gp_r = mean(abs(error_gp_r ./ (F_y_r_nom + eps))) * 100;

%--------------------------------------------------------------------------
% Print Results
%--------------------------------------------------------------------------
fprintf('\n------------------------------------------\n');
fprintf('TIRE FORCE PREDICTION ACCURACY:\n');
fprintf('------------------------------------------\n\n');

fprintf('Front Tire:\n');
fprintf('  Nominal RMSE:         %.2f N (%.1f%% MAPE)\n', stats.rmse_nom_f, stats.mape_nom_f);
fprintf('  Nominal + GP RMSE:    %.2f N (%.1f%% MAPE)\n', stats.rmse_gp_f, stats.mape_gp_f);
fprintf('  Improvement:          %.1f%%\n', stats.improvement_f);
fprintf('  Max Error (Nominal):  %.2f N\n', stats.max_error_nom_f);
fprintf('  Max Error (GP):       %.2f N\n\n', stats.max_error_gp_f);

fprintf('Rear Tire:\n');
fprintf('  Nominal RMSE:         %.2f N (%.1f%% MAPE)\n', stats.rmse_nom_r, stats.mape_nom_r);
fprintf('  Nominal + GP RMSE:    %.2f N (%.1f%% MAPE)\n', stats.rmse_gp_r, stats.mape_gp_r);
fprintf('  Improvement:          %.1f%%\n', stats.improvement_r);
fprintf('  Max Error (Nominal):  %.2f N\n', stats.max_error_nom_r);
fprintf('  Max Error (GP):       %.2f N\n\n', stats.max_error_gp_r);

fprintf('GP Residual Prediction:\n');
fprintf('  Front Residual RMSE:  %.2f N\n', stats.rmse_res_f);
fprintf('  Rear Residual RMSE:   %.2f N\n', stats.rmse_res_r);

fprintf('------------------------------------------\n\n');

%--------------------------------------------------------------------------
% Visualization
%--------------------------------------------------------------------------
fprintf('Generating performance plots...\n');

figure('Name', 'GP Performance in MPC', 'Color', 'w', 'Position', [100 100 1400 800]);

t_vec = out.t(1:K);

% --- Front Tire Force Time History ---
subplot(2,3,1)
plot(t_vec, F_y_f_true, 'b-', 'LineWidth', 2, 'DisplayName', 'True'); hold on;
plot(t_vec, F_y_f_nom, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nominal');
plot(t_vec, F_y_f_gp_aug, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Nominal+GP');
grid on;
xlabel('Time [s]');
ylabel('F_{y,f} [N]');
title('Front Tire Lateral Force');
legend('Location', 'best');

% --- Rear Tire Force Time History ---
subplot(2,3,4)
plot(t_vec, F_y_r_true, 'b-', 'LineWidth', 2, 'DisplayName', 'True'); hold on;
plot(t_vec, F_y_r_nom, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nominal');
plot(t_vec, F_y_r_gp_aug, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Nominal+GP');
grid on;
xlabel('Time [s]');
ylabel('F_{y,r} [N]');
title('Rear Tire Lateral Force');
legend('Location', 'best');

% --- Front Tire Error ---
subplot(2,3,2)
plot(t_vec, error_nom_f, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Nominal'); hold on;
plot(t_vec, error_gp_f, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Nominal+GP');
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]');
ylabel('Error [N]');
title(sprintf('Front Tire Error (RMSE: %.1f → %.1f N)', stats.rmse_nom_f, stats.rmse_gp_f));
legend('Location', 'best');

% --- Rear Tire Error ---
subplot(2,3,5)
plot(t_vec, error_nom_r, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Nominal'); hold on;
plot(t_vec, error_gp_r, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Nominal+GP');
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('Time [s]');
ylabel('Error [N]');
title(sprintf('Rear Tire Error (RMSE: %.1f → %.1f N)', stats.rmse_nom_r, stats.rmse_gp_r));
legend('Location', 'best');

% --- Error Comparison Bar Chart ---
subplot(2,3,3)
categories = {'Front', 'Rear'};
rmse_comparison = [stats.rmse_nom_f stats.rmse_nom_r;
                   stats.rmse_gp_f stats.rmse_gp_r];
b = bar(rmse_comparison');
b(1).FaceColor = [0.8 0.2 0.2];
b(2).FaceColor = [0.2 0.6 0.2];
set(gca, 'XTickLabel', categories);
ylabel('RMSE [N]');
title('Prediction Error Comparison');
legend('Nominal', 'Nominal+GP', 'Location', 'best');
grid on;

% --- Improvement Percentage ---
subplot(2,3,6)
improvement_data = [stats.improvement_f; stats.improvement_r];
bar(improvement_data);
set(gca, 'XTickLabel', categories);
ylabel('Improvement [%]');
title('GP Improvement Over Nominal');
ylim([0 100]);
grid on;
yline(50, 'r--', 'LineWidth', 1.5, 'DisplayName', '50% threshold');

sgtitle('GP-Augmented Model Performance in Closed-Loop MPC');

fprintf('  Done!\n');
fprintf('========================================\n\n');

end
