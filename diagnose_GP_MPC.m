%==========================================================================
% GP-MPC System Diagnostic Tool
%
% Step-by-step verification of:
%   1. GP residual prediction accuracy
%   2. Model dynamics comparison (true vs nominal vs GP-augmented)
%   3. MPC tracking performance
%   4. Constraint satisfaction
%   5. Uncertainty propagation
%
% Usage:
%   Run this after main_cornering.m to analyze the simulation results
%   Or run with: diagnose_GP_MPC(out, d_GP, trueModel, nomModel, estModel)
%==========================================================================

function diagnose_GP_MPC(out, d_GP, trueModel, nomModel, estModel, refPath, beta_max)

    if nargin == 0
        % Load most recent simulation results
        files = dir('simresults/cornering_*.mat');
        if isempty(files)
            error('No simulation results found. Run main_cornering.m first.');
        end
        [~, idx] = max([files.datenum]);
        load(fullfile(files(idx).folder, files(idx).name));
        fprintf('Loaded: %s\n\n', files(idx).name);
    end

    %% ====================================================================
    % STEP 1: GP RESIDUAL PREDICTION ANALYSIS
    % ====================================================================
    fprintf('=======================================================\n');
    fprintf('STEP 1: GP RESIDUAL PREDICTION ANALYSIS\n');
    fprintf('=======================================================\n\n');

    k_valid = find(~isnan(out.x(1,:)), 1, 'last') - 1;

    % Calculate residuals
    residual_true = out.d_est(:, 1:k_valid);
    residual_gp = out.d_gp(:, 1:k_valid);
    residual_error = residual_true - residual_gp;

    % Statistics
    fprintf('--- Residual Statistics ---\n');
    fprintf('Output 1 (Δvy_dot):\n');
    fprintf('  True residual mean: %.4f, std: %.4f\n', mean(residual_true(1,:)), std(residual_true(1,:)));
    fprintf('  GP prediction mean: %.4f, std: %.4f\n', mean(residual_gp(1,:)), std(residual_gp(1,:)));
    fprintf('  Error mean: %.4f, std: %.4f\n', mean(residual_error(1,:)), std(residual_error(1,:)));
    fprintf('  RMSE: %.4f\n', sqrt(mean(residual_error(1,:).^2)));
    fprintf('  Improvement: %.1f%%\n\n', (1 - norm(residual_error(1,:))/norm(residual_true(1,:)))*100);

    fprintf('Output 2 (Δr_dot):\n');
    fprintf('  True residual mean: %.4f, std: %.4f\n', mean(residual_true(2,:)), std(residual_true(2,:)));
    fprintf('  GP prediction mean: %.4f, std: %.4f\n', mean(residual_gp(2,:)), std(residual_gp(2,:)));
    fprintf('  Error mean: %.4f, std: %.4f\n', mean(residual_error(2,:)), std(residual_error(2,:)));
    fprintf('  RMSE: %.4f\n', sqrt(mean(residual_error(2,:).^2)));
    fprintf('  Improvement: %.1f%%\n\n', (1 - norm(residual_error(2,:))/norm(residual_true(2,:)))*100);

    % Plot residuals
    figure('Name', 'STEP 1: GP Residual Prediction', 'Color', 'w', 'Position', [50 50 1200 800]);

    subplot(3,2,1)
    plot(out.t(1:k_valid), residual_true(1,:), 'b-', 'LineWidth', 1.5); hold on;
    plot(out.t(1:k_valid), residual_gp(1,:), 'r--', 'LineWidth', 1.5);
    grid on; legend('True', 'GP prediction', 'Location', 'best');
    ylabel('Δvy_{dot} [m/s^2]');
    title('Lateral Acceleration Residual');

    subplot(3,2,2)
    plot(out.t(1:k_valid), residual_error(1,:), 'k-', 'LineWidth', 1.5);
    grid on; ylabel('Error [m/s^2]');
    title(sprintf('GP Prediction Error (RMSE: %.4f)', sqrt(mean(residual_error(1,:).^2))));

    subplot(3,2,3)
    plot(out.t(1:k_valid), residual_true(2,:), 'b-', 'LineWidth', 1.5); hold on;
    plot(out.t(1:k_valid), residual_gp(2,:), 'r--', 'LineWidth', 1.5);
    grid on; legend('True', 'GP prediction', 'Location', 'best');
    ylabel('Δr_{dot} [rad/s^2]');
    title('Yaw Acceleration Residual');

    subplot(3,2,4)
    plot(out.t(1:k_valid), residual_error(2,:), 'k-', 'LineWidth', 1.5);
    grid on; ylabel('Error [rad/s^2]');
    title(sprintf('GP Prediction Error (RMSE: %.4f)', sqrt(mean(residual_error(2,:).^2))));

    subplot(3,2,5)
    scatter(residual_true(1,:), residual_gp(1,:), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
    hold on; plot(xlim, xlim, 'r--', 'LineWidth', 1.5);
    grid on; xlabel('True Δvy_{dot}'); ylabel('GP prediction');
    title('Correlation Plot - Lateral Accel');
    axis equal; axis tight;

    subplot(3,2,6)
    scatter(residual_true(2,:), residual_gp(2,:), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.4);
    hold on; plot(xlim, xlim, 'r--', 'LineWidth', 1.5);
    grid on; xlabel('True Δr_{dot}'); ylabel('GP prediction');
    title('Correlation Plot - Yaw Accel');
    axis equal; axis tight;

    sgtitle('STEP 1: GP Residual Prediction Quality');


    %% ====================================================================
    % STEP 2: MODEL DYNAMICS COMPARISON
    % ====================================================================
    fprintf('\n=======================================================\n');
    fprintf('STEP 2: MODEL DYNAMICS COMPARISON\n');
    fprintf('=======================================================\n\n');

    % One-step prediction errors
    error_nom = out.xhat(:,2:k_valid+1) - out.xnom(:,2:k_valid+1);

    % GP-augmented prediction
    xhat_gp_aug = out.xnom(:,2:k_valid+1) + estModel.Bd * residual_gp;
    error_gp_aug = out.xhat(:,2:k_valid+1) - xhat_gp_aug;

    fprintf('--- One-Step Prediction Error (State-wise RMSE) ---\n');
    fprintf('State          | Nominal      | GP-augmented | Improvement\n');
    fprintf('---------------------------------------------------------------\n');
    state_names = {'vx [m/s]  ', 'vy [m/s]  ', 'psi [rad] ', 'r [rad/s] '};
    for i = 1:min(size(error_nom,1), 4)
        rmse_nom = sqrt(mean(error_nom(i,:).^2));
        rmse_gp = sqrt(mean(error_gp_aug(i,:).^2));
        improvement = (1 - rmse_gp/rmse_nom)*100;
        fprintf('%s | %.6f    | %.6f     | %+.1f%%\n', state_names{i}, rmse_nom, rmse_gp, improvement);
    end
    fprintf('\n');

    % Plot state prediction comparison (only first 3 states: vx, vy, psi - skip r as it's similar to psi)
    figure('Name', 'STEP 2: Model Dynamics Comparison', 'Color', 'w', 'Position', [100 100 1200 700]);

    states_to_plot = [1, 2, 4];  % vx, vy, r (skip psi as it's integrated from r)
    for plot_idx = 1:3
        i = states_to_plot(plot_idx);
        subplot(3,2,2*plot_idx-1)
        plot(out.t(2:k_valid+1), error_nom(i,:), 'b-', 'LineWidth', 1.5); hold on;
        plot(out.t(2:k_valid+1), error_gp_aug(i,:), 'r--', 'LineWidth', 1.5);
        grid on; ylabel(sprintf('Error %s', state_names{i}));
        legend('Nominal', 'GP-augmented', 'Location', 'best');
        if plot_idx == 1, title('One-Step Prediction Error'); end

        subplot(3,2,2*plot_idx)
        histogram(error_nom(i,:), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
        histogram(error_gp_aug(i,:), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        grid on; xlabel('Error'); ylabel('Count');
        legend('Nominal', 'GP-augmented', 'Location', 'best');
        if plot_idx == 1, title('Error Distribution'); end
    end

    sgtitle('STEP 2: Model Prediction Quality');


    %% ====================================================================
    % STEP 3: MPC TRACKING PERFORMANCE
    % ====================================================================
    fprintf('\n=======================================================\n');
    fprintf('STEP 3: MPC TRACKING PERFORMANCE\n');
    fprintf('=======================================================\n\n');

    % Tracking errors
    if exist('refPath', 'var')
        vx_error = out.x(1,1:k_valid) - refPath.v_ref;
        r_error = out.x(4,1:k_valid) - out.r_ref(1:k_valid);  % r is now 4th state

        fprintf('--- Reference Tracking Errors ---\n');
        fprintf('Longitudinal velocity (vx):\n');
        fprintf('  Mean error: %.4f m/s\n', mean(vx_error));
        fprintf('  Std error:  %.4f m/s\n', std(vx_error));
        fprintf('  RMSE:       %.4f m/s\n', sqrt(mean(vx_error.^2)));
        fprintf('  Max |error|: %.4f m/s\n\n', max(abs(vx_error)));

        fprintf('Yaw rate (r):\n');
        fprintf('  Mean error: %.4f rad/s (%.2f deg/s)\n', mean(r_error), rad2deg(mean(r_error)));
        fprintf('  Std error:  %.4f rad/s (%.2f deg/s)\n', std(r_error), rad2deg(std(r_error)));
        fprintf('  RMSE:       %.4f rad/s (%.2f deg/s)\n', sqrt(mean(r_error.^2)), rad2deg(sqrt(mean(r_error.^2))));
        fprintf('  Max |error|: %.4f rad/s (%.2f deg/s)\n\n', max(abs(r_error)), rad2deg(max(abs(r_error))));
    end

    % Plot tracking performance
    figure('Name', 'STEP 3: MPC Tracking Performance', 'Color', 'w', 'Position', [150 150 1200 800]);

    subplot(4,1,1)
    plot(out.t(1:k_valid), out.x(1,1:k_valid), 'b-', 'LineWidth', 1.5); hold on;
    if exist('refPath', 'var')
        plot(out.t(1:k_valid), refPath.v_ref*ones(1,k_valid), 'r--', 'LineWidth', 1.5);
    end
    grid on; ylabel('vx [m/s]');
    title('Longitudinal Velocity Tracking');
    legend('Actual', 'Reference', 'Location', 'best');

    subplot(4,1,2)
    plot(out.t(1:k_valid), rad2deg(out.x(3,1:k_valid)), 'b-', 'LineWidth', 1.5);
    grid on; ylabel('ψ [deg]');
    title('Yaw Angle');

    subplot(4,1,3)
    plot(out.t(1:k_valid), rad2deg(out.x(4,1:k_valid)), 'b-', 'LineWidth', 1.5); hold on;
    if exist('out', 'var') && isfield(out, 'r_ref')
        plot(out.t(1:k_valid), rad2deg(out.r_ref(1:k_valid)), 'r--', 'LineWidth', 1.5);
    end
    grid on; ylabel('r [deg/s]');
    title('Yaw Rate Tracking');
    legend('Actual', 'Reference', 'Location', 'best');

    subplot(4,1,4)
    stairs(out.t(1:k_valid), rad2deg(out.u(1,1:k_valid)), 'b-', 'LineWidth', 1.5);
    grid on; ylabel('δ [deg]'); xlabel('Time [s]');
    title('Steering Input');

    sgtitle('STEP 3: MPC Tracking Performance');


    %% ====================================================================
    % STEP 4: CONSTRAINT SATISFACTION
    % ====================================================================
    fprintf('\n=======================================================\n');
    fprintf('STEP 4: CONSTRAINT SATISFACTION\n');
    fprintf('=======================================================\n\n');

    beta = out.beta(1:k_valid);
    delta = out.u(1,1:k_valid);

    fprintf('--- Constraint Violations ---\n');
    if exist('beta_max', 'var')
        beta_violations = sum(abs(beta) > beta_max);
        fprintf('Sideslip angle (β):\n');
        fprintf('  Limit: ±%.2f deg\n', rad2deg(beta_max));
        fprintf('  Max |β|: %.2f deg\n', rad2deg(max(abs(beta))));
        fprintf('  Violations: %d / %d (%.1f%%)\n\n', beta_violations, k_valid, 100*beta_violations/k_valid);
    else
        fprintf('Sideslip angle (β):\n');
        fprintf('  Max |β|: %.2f deg\n', rad2deg(max(abs(beta))));
        fprintf('  Mean |β|: %.2f deg\n\n', rad2deg(mean(abs(beta))));
    end

    fprintf('Steering angle (δ):\n');
    fprintf('  Max |δ|: %.2f deg\n', rad2deg(max(abs(delta))));
    fprintf('  Mean |δ|: %.2f deg\n\n', rad2deg(mean(abs(delta))));

    % Steering rate
    delta_dot = diff(delta) / (out.t(2) - out.t(1));
    fprintf('Steering rate (Δδ):\n');
    fprintf('  Max |Δδ|: %.2f deg/s\n', rad2deg(max(abs(delta_dot))));
    fprintf('  Mean |Δδ|: %.2f deg/s\n\n', rad2deg(mean(abs(delta_dot))));

    % Plot constraints
    figure('Name', 'STEP 4: Constraint Satisfaction', 'Color', 'w', 'Position', [200 200 1200 600]);

    subplot(2,2,1)
    plot(out.t(1:k_valid), rad2deg(beta), 'b-', 'LineWidth', 1.5); hold on;
    if exist('beta_max', 'var')
        yline(rad2deg(beta_max), 'r--', 'LineWidth', 1.5);
        yline(-rad2deg(beta_max), 'r--', 'LineWidth', 1.5);
    end
    grid on; ylabel('β [deg]'); xlabel('Time [s]');
    title(sprintf('Sideslip Angle (max: %.2f deg)', rad2deg(max(abs(beta)))));

    subplot(2,2,2)
    histogram(rad2deg(beta), 40, 'FaceColor', 'b', 'FaceAlpha', 0.6);
    if exist('beta_max', 'var')
        hold on;
        xline(rad2deg(beta_max), 'r--', 'LineWidth', 2);
        xline(-rad2deg(beta_max), 'r--', 'LineWidth', 2);
    end
    grid on; xlabel('β [deg]'); ylabel('Count');
    title('Sideslip Distribution');

    subplot(2,2,3)
    stairs(out.t(1:k_valid), rad2deg(delta), 'b-', 'LineWidth', 1.5);
    grid on; ylabel('δ [deg]'); xlabel('Time [s]');
    title(sprintf('Steering Angle (max: %.2f deg)', rad2deg(max(abs(delta)))));

    subplot(2,2,4)
    plot(out.t(1:k_valid-1), rad2deg(delta_dot), 'b-', 'LineWidth', 1.5);
    grid on; ylabel('Δδ [deg/s]'); xlabel('Time [s]');
    title(sprintf('Steering Rate (max: %.2f deg/s)', rad2deg(max(abs(delta_dot)))));

    sgtitle('STEP 4: Constraint Satisfaction');


    %% ====================================================================
    % STEP 5: UNCERTAINTY PROPAGATION (if variance data exists)
    % ====================================================================
    fprintf('\n=======================================================\n');
    fprintf('STEP 5: UNCERTAINTY PROPAGATION\n');
    fprintf('=======================================================\n\n');

    if isfield(out, 'var_x_pred') && any(~isnan(out.var_x_pred(:)), 'all')
        fprintf('Variance data detected. Analyzing uncertainty propagation...\n\n');

        % Extract variance at first prediction step
        var_vy = squeeze(out.var_x_pred(2,2,2,1:k_valid));  % vy variance at k+1
        var_r = squeeze(out.var_x_pred(4,4,2,1:k_valid));   % r variance at k+1 (now 4th state)

        fprintf('--- Uncertainty Statistics ---\n');
        fprintf('Lateral velocity (vy) std:\n');
        fprintf('  Mean: %.4f m/s\n', mean(sqrt(var_vy)));
        fprintf('  Max:  %.4f m/s\n\n', max(sqrt(var_vy)));

        fprintf('Yaw rate (r) std:\n');
        fprintf('  Mean: %.4f rad/s (%.2f deg/s)\n', mean(sqrt(var_r)), rad2deg(mean(sqrt(var_r))));
        fprintf('  Max:  %.4f rad/s (%.2f deg/s)\n\n', max(sqrt(var_r)), rad2deg(max(sqrt(var_r))));

        % Plot uncertainty
        figure('Name', 'STEP 5: Uncertainty Propagation', 'Color', 'w', 'Position', [250 250 1200 500]);

        subplot(1,2,1)
        plot(out.t(1:k_valid), sqrt(var_vy), 'b-', 'LineWidth', 1.5);
        grid on; ylabel('σ_{vy} [m/s]'); xlabel('Time [s]');
        title('Lateral Velocity Uncertainty');

        subplot(1,2,2)
        plot(out.t(1:k_valid), rad2deg(sqrt(var_r)), 'b-', 'LineWidth', 1.5);
        grid on; ylabel('σ_r [deg/s]'); xlabel('Time [s]');
        title('Yaw Rate Uncertainty');

        sgtitle('STEP 5: GP Uncertainty Propagation');
    else
        fprintf('No variance data found in simulation results.\n');
        fprintf('Variance propagation may be disabled or not stored.\n\n');
    end


    %% ====================================================================
    % STEP 6: GP TRAINING EVOLUTION
    % ====================================================================
    fprintf('\n=======================================================\n');
    fprintf('STEP 6: GP TRAINING EVOLUTION\n');
    fprintf('=======================================================\n\n');

    % Compute rolling RMSE
    window = min(50, k_valid);
    rolling_rmse = zeros(1, k_valid - window + 1);
    for i = 1:(k_valid - window + 1)
        idx = i:(i+window-1);
        rolling_rmse(i) = sqrt(mean(sum(residual_error(:,idx).^2, 1)));
    end

    fprintf('--- GP Learning Progress ---\n');
    fprintf('Initial RMSE (first 50 steps): %.4f\n', rolling_rmse(1));
    fprintf('Final RMSE (last 50 steps):    %.4f\n', rolling_rmse(end));
    fprintf('Improvement: %.1f%%\n\n', (1 - rolling_rmse(end)/rolling_rmse(1))*100);

    if exist('d_GP', 'var')
        fprintf('GP Dictionary size: %d / %d\n', size(d_GP.X, 2), d_GP.Nmax);
        fprintf('GP Hyperparameters:\n');
        fprintf('  Signal variance: [%.4f, %.4f]\n', d_GP.var_f(1), d_GP.var_f(2));
        fprintf('  Noise variance:  [%.4f, %.4f]\n', d_GP.var_n(1), d_GP.var_n(2));
    end

    % Plot learning progress
    figure('Name', 'STEP 6: GP Learning Evolution', 'Color', 'w', 'Position', [300 300 1200 500]);

    subplot(1,2,1)
    plot(out.t(1:length(rolling_rmse)), rolling_rmse, 'b-', 'LineWidth', 1.5);
    grid on; ylabel('Rolling RMSE'); xlabel('Time [s]');
    title(sprintf('GP Learning Progress (window=%d steps)', window));

    subplot(1,2,2)
    error_norm = vecnorm(residual_error);
    plot(out.t(1:k_valid), error_norm, 'b-', 'LineWidth', 1.5);
    grid on; ylabel('Prediction Error Norm'); xlabel('Time [s]');
    title('Instantaneous GP Error');

    sgtitle('STEP 6: GP Training Evolution');


    %% ====================================================================
    % SUMMARY
    % ====================================================================
    fprintf('\n\n=======================================================\n');
    fprintf('DIAGNOSTIC SUMMARY\n');
    fprintf('=======================================================\n\n');

    fprintf('✓ STEP 1: GP residual prediction RMSE: %.4f\n', sqrt(mean(sum(residual_error.^2, 1))));
    fprintf('✓ STEP 2: State prediction improvement: %.1f%%\n', mean((1 - sqrt(mean(error_gp_aug.^2, 2))./sqrt(mean(error_nom.^2, 2)))*100));
    if exist('refPath', 'var')
        fprintf('✓ STEP 3: Velocity tracking RMSE: %.4f m/s\n', sqrt(mean(vx_error.^2)));
        fprintf('✓ STEP 3: Yaw rate tracking RMSE: %.2f deg/s\n', rad2deg(sqrt(mean(r_error.^2))));
    end
    fprintf('✓ STEP 4: Max sideslip angle: %.2f deg\n', rad2deg(max(abs(beta))));
    if exist('beta_max', 'var')
        fprintf('✓ STEP 4: Constraint violations: %d / %d (%.1f%%)\n', beta_violations, k_valid, 100*beta_violations/k_valid);
    end
    fprintf('✓ STEP 6: GP learning improvement: %.1f%%\n', (1 - rolling_rmse(end)/rolling_rmse(1))*100);

    fprintf('\n=======================================================\n\n');
end
