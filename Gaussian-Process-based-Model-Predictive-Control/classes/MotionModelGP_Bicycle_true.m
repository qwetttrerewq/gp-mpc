%--------------------------------------------------------------------------
% GP-MPC Bicycle Model - True System (Pacejka Tire)
% Represents the actual vehicle with nonlinear tire dynamics
%
% State:  x = [vx; vy; psi; r]  (longitudinal vel, lateral vel, yaw angle, yaw rate)
% Input:  u = [delta]      (steering angle)
% GP in:  z = [vy; r; delta]
% GP out: d = [ΔF_y_f; ΔF_y_r]  (tire force residuals)
%
% NOTE: Same structure as nominal model, only tire model differs
%       vx is kept constant (vx_dot = 0) for cornering simplicity
%--------------------------------------------------------------------------

classdef MotionModelGP_Bicycle_true < MotionModelGP
    %--------------------------------------------------------------------------
    %   Discrete dynamics:
    %   xk+1 = fd(xk,uk) + Bd * ( d(zk) + w ),
    %
    %       where: zk = [Bz_x*xk ; Bz_u*uk]
    %              d ~ N(mean_d(zk),var_d(zk))
    %              w ~ N(0,sigmaw)
    %
    %   State x = [vx   (longitudinal velocity [m/s])
    %              vy   (lateral velocity [m/s])
    %              psi  (yaw angle [rad])
    %              r    (yaw rate [rad/s])
    %             ]
    %
    %   Input u = [delta  (steering angle [rad])]
    %
    %   GP input z = [vy; r; delta]  (3D)
    %   GP output d = [ΔF_y_f; ΔF_y_r]  (2D tire force residuals)
    %--------------------------------------------------------------------------
    
    properties(Constant)
        % Vehicle parameters (SAME as nominal - physical reality!)
        M    = 500       % vehicle mass [kg]
        I_z  = 600       % vehicle moment of inertia [kg*m^2]
        g    = 9.81      % gravitation [m/s^2]
        l_f  = 0.9       % distance of front axle to CoG [m]
        l_r  = 1.5       % distance of rear axle to CoG [m]
        
        % Steering limits
        deltamax = deg2rad(30)   % maximum steering amplitude [rad]
        
        % Pacejka tire model parameters (Magic Formula)
        % Front tire
        % B_f = 0.4        % stiffness factor
        % C_f = 8          % shape factor
        % D_f = 4560.4     % peak value [N]
        % E_f = -0.5       % curvature factor
        
        % % Rear tire
        % B_r = 0.45       % stiffness factor
        % C_r = 8          % shape factor
        % D_r = 4000       % peak value [N]
        % E_r = -0.5       % curvature factor
        
        B_f = 8.5        % stiffness factor
        C_f = 1.3          % shape factor
        D_f = 3500     % peak value [N]
        E_f = -0.5       % curvature factor
        
        % Rear tire
        B_r = 9.0       % stiffness factor
        C_r = 1.3          % shape factor
        D_r = 3200       % peak value [N]
        E_r = -0.5       % curvature factor
    end
    
    properties(Constant)
        % GP integration matrices (SAME structure as nominal)
        Bz_x = [0 1 0 0;    % extract [vy; r] from [vx; vy; psi; r]
            0 0 0 1]
        Bz_u = 1            % extract steering angle for GP input
        
        % Bd matrix for tire force injection (SAME as nominal)
        % Assumes small angle: cos(delta) ≈ 1
        Bd = [0         0;           % vx not affected
            1/500     1/500;       % vy: (ΔF_y_f + ΔF_y_r)/M
            0         0;           % psi not affected directly
            0.9/600  -1.5/600]     % r: (l_f*ΔF_y_f - l_r*ΔF_y_r)/I_z
        
        n  = 4   % number of states x(t) = [vx; vy; psi; r]
        m  = 1   % number of inputs u(t)
        nz = 3   % dimension of z(t) = [vy; r; delta]
        nd = 2   % output dimension of d(z) = [ΔF_y_f; ΔF_y_r]
    end
    
    methods
        function obj = MotionModelGP_Bicycle_true(d, sigmaw)
            %------------------------------------------------------------------
            %   Constructor
            %   Inputs:
            %       d       - GP evaluation function (typically empty for true model)
            %       sigmaw  - Process noise covariance (2x2)
            %------------------------------------------------------------------
            % Call superclass constructor
            obj = obj@MotionModelGP(d, sigmaw);
            
            fprintf('\n=== Bicycle Model (TRUE with Pacejka tires) Created ===\n');
            fprintf('  Mass M = %.0f [kg]\n', obj.M);
            fprintf('  Yaw inertia Iz = %.0f [kg*m^2]\n', obj.I_z);
            fprintf('  Pacejka params: Df=%.0f N, Dr=%.0f N\n', obj.D_f, obj.D_r);
            fprintf('======================================================\n\n');
        end
        
        function xdot = f(obj, x, u)
            %------------------------------------------------------------------
            %   Continuous-time dynamics with PACEJKA TIRE MODEL
            %------------------------------------------------------------------
            
            % Model parameters
            M = obj.M;
            I_z = obj.I_z;
            l_f = obj.l_f;
            l_r = obj.l_r;
            deltamax = obj.deltamax;
            
            % Pacejka coefficients
            B_f = obj.B_f;
            C_f = obj.C_f;
            D_f = obj.D_f;
            E_f = obj.E_f;
            B_r = obj.B_r;
            C_r = obj.C_r;
            D_r = obj.D_r;
            E_r = obj.E_r;
            
            %--------------------------------------------------------------
            % Extract state and input
            %--------------------------------------------------------------
            vx = x(1);      % longitudinal velocity [m/s]
            vy = x(2);      % lateral velocity [m/s]
            psi = x(3);     % yaw angle [rad]
            r  = x(4);      % yaw rate [rad/s]
            
            delta = u(1);   % steering angle [rad]
            
            %--------------------------------------------------------------
            % Saturate steering input (hard saturation)
            %--------------------------------------------------------------
            delta = obj.clip(delta, -deltamax, deltamax);
            
            %--------------------------------------------------------------
            % Calculate slip angles
            %--------------------------------------------------------------
            % Front slip angle
            alpha_f = atan2(vy + l_f*r, vx) - delta;
            
            % Rear slip angle
            alpha_r = atan2(vy - l_r*r, vx);
            
            %--------------------------------------------------------------
            % Calculate tire forces in WHEEL COORDINATES
            % (Pacejka Magic Formula)
            % NOTE: Wheel coordinate has z-axis pointing DOWN
            %--------------------------------------------------------------
            W_Fy_f = D_f * sin(C_f * atan(B_f*alpha_f - E_f*(B_f*alpha_f - atan(B_f*alpha_f))));
            W_Fy_r = D_r * sin(C_r * atan(B_r*alpha_r - E_r*(B_r*alpha_r - atan(B_r*alpha_r))));
            
            %--------------------------------------------------------------
            % Transform to VEHICLE COORDINATES (z-axis pointing UP)
            % CRITICAL: Sign flip for lateral forces!
            %--------------------------------------------------------------
            V_Fy_f = -W_Fy_f;  % Sign flip: wheel→vehicle coordinate
            V_Fy_r = -W_Fy_r;  % Sign flip: wheel→vehicle coordinate
            fprintf('Debug: V_Fy_f=%.2f N, V_Fy_r=%.2f N\n', V_Fy_f, V_Fy_r);
            %--------------------------------------------------------------
            % Dynamic bicycle equations
            %--------------------------------------------------------------
            % Longitudinal dynamics (constant vx assumption)
            vx_dot = 0;  % Assume constant speed for cornering
            
            % Lateral dynamics
            vy_dot = (1/M) * (V_Fy_f*cos(delta) + V_Fy_r) - r*vx;
            
            % Yaw angle dynamics
            psi_dot = r;
            
            % Yaw rate dynamics
            r_dot = (1/I_z) * (l_f*V_Fy_f*cos(delta) - l_r*V_Fy_r);
            
            %--------------------------------------------------------------
            % Assemble state derivative
            %--------------------------------------------------------------
            xdot = [vx_dot; vy_dot; psi_dot; r_dot];
            
            %--------------------------------------------------------------
            % Check model integrity
            %--------------------------------------------------------------
            if any(isnan(xdot)) || any(isinf(xdot)) || any(imag(xdot)~=0)
                error('True Bicycle Model evaluated to Inf or NaN... CHECK MODEL!')
            end
        end
        
        
        function gradx = gradx_f(obj, ~, ~)
            %------------------------------------------------------------------
            %   Gradient w.r.t. state (not implemented for true model)
            %   True model is used for simulation only, not optimization
            %------------------------------------------------------------------
            gradx = zeros(obj.n, obj.n);
        end
        
        
        function gradu = gradu_f(obj, ~, ~)
            %------------------------------------------------------------------
            %   Gradient w.r.t. input (not implemented for true model)
            %------------------------------------------------------------------
            gradu = zeros(obj.n, obj.m);
        end
        
        
        function [F_y_f, F_y_r] = getTireForces(obj, x, u)
            %------------------------------------------------------------------
            %   Calculate tire lateral forces (Pacejka model)
            %   Returns: F_y_f (front), F_y_r (rear)
            %------------------------------------------------------------------
            vx = x(1);
            vy = x(2);
            r = x(4);
            delta = u(1);
            
            % Saturate steering
            delta = obj.clip(delta, -obj.deltamax, obj.deltamax);
            
            % Slip angles
            alpha_f = atan2(vy + obj.l_f*r, vx) - delta;
            alpha_r = atan2(vy - obj.l_r*r, vx);
            
            % Pacejka tire forces in wheel coordinates
            W_Fy_f = obj.D_f * sin(obj.C_f * atan(obj.B_f*alpha_f - ...
                obj.E_f*(obj.B_f*alpha_f - atan(obj.B_f*alpha_f))));
            W_Fy_r = obj.D_r * sin(obj.C_r * atan(obj.B_r*alpha_r - ...
                obj.E_r*(obj.B_r*alpha_r - atan(obj.B_r*alpha_r))));
            
            % Transform to vehicle coordinates
            F_y_f = -W_Fy_f;
            F_y_r = -W_Fy_r;
        end
        
        function plotTireCharacteristics(obj)
            %------------------------------------------------------------------
            %   Visualize Pacejka tire force vs slip angle
            %------------------------------------------------------------------
            alpha = linspace(-deg2rad(20), deg2rad(20), 200);
            
            % Front tire (Pacejka)
            W_Fy_f = obj.D_f * sin(obj.C_f * atan(obj.B_f*alpha - ...
                obj.E_f*(obj.B_f*alpha - atan(obj.B_f*alpha))));
            
            % Rear tire (Pacejka)
            W_Fy_r = obj.D_r * sin(obj.C_r * atan(obj.B_r*alpha - ...
                obj.E_r*(obj.B_r*alpha - atan(obj.B_r*alpha))));
            
            % Transform to vehicle coordinates
            V_Fy_f = -W_Fy_f;
            V_Fy_r = -W_Fy_r;
            
            % Linear approximation (for comparison)
            c_f_linear = 14000 * 2.5;
            c_r_linear = 14000 * 2.5;
            Fy_f_linear = c_f_linear * alpha;
            Fy_r_linear = c_r_linear * alpha;
            
            % Plot
            figure('Name', 'Tire Characteristics', 'Color', 'w');
            
            subplot(1,2,1)
            hold on; grid on;
            plot(rad2deg(alpha), V_Fy_f, 'b-', 'LineWidth', 2, 'DisplayName', 'Pacejka (True)');
            plot(rad2deg(alpha), Fy_f_linear, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Linear (Nominal)');
            xlabel('Slip Angle [deg]')
            ylabel('Lateral Force [N]')
            title('Front Tire')
            legend('Location', 'best')
            xlim([-20 20])
            
            subplot(1,2,2)
            hold on; grid on;
            plot(rad2deg(alpha), V_Fy_r, 'b-', 'LineWidth', 2, 'DisplayName', 'Pacejka (True)');
            plot(rad2deg(alpha), Fy_r_linear, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Linear (Nominal)');
            xlabel('Slip Angle [deg]')
            ylabel('Lateral Force [N]')
            title('Rear Tire')
            legend('Location', 'best')
            xlim([-20 20])
            
            sgtitle('Tire Force vs Slip Angle: True (Pacejka) vs Nominal (Linear)')
        end
    end
    
    
    methods(Static)
        function x = clip(x, lb, ub)
            % Hard saturation (non-differentiable)
            x = min(max(x, lb), ub);
        end
    end
end
