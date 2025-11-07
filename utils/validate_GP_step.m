function validate_GP_step(k, out, d_GP, estModel, verbose)
%==========================================================================
% Validate Single Time Step of GP-MPC
%
% Verifies at timestep k:
%   1. GP input construction (z_k)
%   2. GP prediction (d_gp) vs true residual (d_est)
%   3. State prediction with/without GP
%   4. Constraint satisfaction
%
% Usage:
%   validate_GP_step(k, out, d_GP, estModel)
%   validate_GP_step(k, out, d_GP, estModel, true)  % verbose output
%
% Example:
%   % After running main_cornering.m
%   validate_GP_step(50, out, d_GP, estModel, true);
%==========================================================================

    if nargin < 5
        verbose = false;
    end

    fprintf('\n=====================================================\n');
    fprintf('VALIDATING TIMESTEP k=%d (t=%.2f s)\n', k, out.t(k));
    fprintf('=====================================================\n\n');

    %% 1. GP Input Construction
    fprintf('--- Step 1: GP Input Construction ---\n');

    % Reconstruct z_k manually
    z_k_stored = out.z_train(:, k);
    z_k_reconstructed = [estModel.Bz_x * out.xhat(:,k); estModel.Bz_u * out.u(:,k)];

    fprintf('GP input z_k = [vx; vy; r; delta]\n');
    fprintf('  Stored:        [%.4f, %.4f, %.4f, %.4f]\n', z_k_stored);
    fprintf('  Reconstructed: [%.4f, %.4f, %.4f, %.4f]\n', z_k_reconstructed);

    if norm(z_k_stored - z_k_reconstructed) < 1e-6
        fprintf('  ✓ PASS: GP input construction is correct\n\n');
    else
        fprintf('  ✗ FAIL: Mismatch detected (error: %.6e)\n\n', norm(z_k_stored - z_k_reconstructed));
    end

    %% 2. True Residual Calculation
    fprintf('--- Step 2: True Residual Calculation ---\n');

    % True residual: d = Bd^-1 * (x_true - x_nom)
    d_est_stored = out.d_est(:, k);
    d_est_reconstructed = estModel.Bd \ (out.xhat(:,k+1) - out.xnom(:,k+1));

    fprintf('True residual d_est = [Δvy_dot; Δr_dot]\n');
    fprintf('  Stored:        [%.6f, %.6f]\n', d_est_stored);
    fprintf('  Reconstructed: [%.6f, %.6f]\n', d_est_reconstructed);

    if norm(d_est_stored - d_est_reconstructed) < 1e-6
        fprintf('  ✓ PASS: Residual calculation is correct\n\n');
    else
        fprintf('  ✗ FAIL: Mismatch detected (error: %.6e)\n\n', norm(d_est_stored - d_est_reconstructed));
    end

    %% 3. GP Prediction
    fprintf('--- Step 3: GP Prediction ---\n');

    [d_gp_mean, d_gp_var] = d_GP.eval(z_k_stored, true);
    d_gp_stored = out.d_gp(:, k);

    fprintf('GP prediction d_gp = [Δvy_dot; Δr_dot]\n');
    fprintf('  Mean (stored):       [%.6f, %.6f]\n', d_gp_stored);
    fprintf('  Mean (re-evaluated): [%.6f, %.6f]\n', d_gp_mean);
    fprintf('  Variance:            [%.6f, %.6f]\n', d_gp_var);

    % Check prediction error
    pred_error = d_est_stored - d_gp_stored;
    fprintf('\nPrediction error (d_est - d_gp):\n');
    fprintf('  [%.6f, %.6f]\n', pred_error);
    fprintf('  Norm: %.6f\n', norm(pred_error));

    % Check if error is within uncertainty bounds
    within_bounds = abs(pred_error) <= 2*sqrt(d_gp_var);
    if all(within_bounds)
        fprintf('  ✓ PASS: Prediction error within 2σ bounds\n\n');
    else
        fprintf('  ⚠ WARNING: Prediction error exceeds 2σ bounds for dimension(s): %s\n\n', mat2str(find(~within_bounds)));
    end

    %% 4. State Prediction Quality
    fprintf('--- Step 4: State Prediction Quality ---\n');

    % Nominal prediction error (without GP)
    error_nom = out.xhat(:,k+1) - out.xnom(:,k+1);

    % GP-augmented prediction
    x_pred_gp = out.xnom(:,k+1) + estModel.Bd * d_gp_stored;
    error_gp = out.xhat(:,k+1) - x_pred_gp;

    fprintf('State prediction errors:\n');
    fprintf('  State     | True      | Nominal   | GP-aug    | Improvement\n');
    fprintf('  ------------------------------------------------------------------\n');
    state_names = {'vx', 'vy', 'r'};
    for i = 1:length(state_names)
        improvement = (1 - abs(error_gp(i))/abs(error_nom(i)))*100;
        fprintf('  %-8s  | %.6f  | %.6f  | %.6f  | %+.1f%%\n', ...
            state_names{i}, out.xhat(i,k+1), error_nom(i), error_gp(i), improvement);
    end
    fprintf('\n');

    %% 5. Constraint Satisfaction
    fprintf('--- Step 5: Constraint Satisfaction ---\n');

    beta_k = out.beta(k);
    delta_k = out.u(1, k);

    fprintf('Constraints at timestep k:\n');
    fprintf('  Sideslip β:      %.4f rad (%.2f deg)\n', beta_k, rad2deg(beta_k));
    fprintf('  Steering δ:      %.4f rad (%.2f deg)\n', delta_k, rad2deg(delta_k));

    % Check if constraints are satisfied (typical values)
    beta_limit = deg2rad(15);
    delta_limit = deg2rad(30);

    beta_ok = abs(beta_k) <= beta_limit;
    delta_ok = abs(delta_k) <= delta_limit;

    if beta_ok
        fprintf('  ✓ Sideslip within limits (±%.1f deg)\n', rad2deg(beta_limit));
    else
        fprintf('  ✗ Sideslip VIOLATES limits (±%.1f deg)\n', rad2deg(beta_limit));
    end

    if delta_ok
        fprintf('  ✓ Steering within limits (±%.1f deg)\n', rad2deg(delta_limit));
    else
        fprintf('  ✗ Steering VIOLATES limits (±%.1f deg)\n', rad2deg(delta_limit));
    end
    fprintf('\n');

    %% 6. MPC Prediction Horizon (if available)
    if isfield(out, 'mu_x_pred') && verbose
        fprintf('--- Step 6: MPC Prediction Horizon ---\n');

        mu_x_pred = out.mu_x_pred(:,:,k);
        var_x_pred = out.var_x_pred(:,:,:,k);

        fprintf('Predicted state trajectory (N+1 steps):\n');
        fprintf('  Step | vx      | vy      | r       | σ_vy    | σ_r\n');
        fprintf('  --------------------------------------------------------------\n');
        for i = 1:min(5, size(mu_x_pred, 2))  % Show first 5 steps
            sigma_vy = sqrt(var_x_pred(2,2,i));
            sigma_r = sqrt(var_x_pred(3,3,i));
            fprintf('  %4d | %.4f | %.4f | %.4f | %.4f | %.4f\n', ...
                i-1, mu_x_pred(1,i), mu_x_pred(2,i), mu_x_pred(3,i), sigma_vy, sigma_r);
        end
        if size(mu_x_pred, 2) > 5
            fprintf('  ... (showing first 5 of %d steps)\n', size(mu_x_pred, 2));
        end
        fprintf('\n');
    end

    %% Summary
    fprintf('=====================================================\n');
    fprintf('VALIDATION SUMMARY\n');
    fprintf('=====================================================\n');
    fprintf('GP prediction error norm: %.6f\n', norm(pred_error));
    fprintf('State prediction improvement: %.1f%%\n', mean((1 - abs(error_gp)./abs(error_nom))*100));
    fprintf('Constraints satisfied: %s\n', string(beta_ok && delta_ok));
    fprintf('=====================================================\n\n');

end
