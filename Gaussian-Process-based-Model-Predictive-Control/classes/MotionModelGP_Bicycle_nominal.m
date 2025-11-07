%--------------------------------------------------------------------------
% GP-MPC Bicycle Model - Nominal (Linear Tire)
% Simplified cornering dynamics for GP-MPC research
%
% State:  x = [vx; vy; psi; r]  (longitudinal vel, lateral vel, yaw angle, yaw rate)
% Input:  u = [delta]      (steering angle)
% GP in:  z = [vy; r; delta]
% GP out: d = [ΔF_y_f; ΔF_y_r]  (tire force residuals)
%--------------------------------------------------------------------------

classdef MotionModelGP_Bicycle_nominal < MotionModelGP
    %--------------------------------------------------------------------------
    %   Discrete dynamics:
    %   xk+1 = fd(xk,uk) + Bd * ( d(zk) + w ),
    %
    %       where: zk = [Bz_x*xk ; Bz_u*uk]
    %              d ~ N(mean_d(zk),var_d(zk))
    %              w ~ N(0,sigmaw)
    %
    %   State x = [vx   (longitudinal velocity in vehicle frame [m/s])
    %              vy   (lateral velocity in vehicle frame [m/s])
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
        % Vehicle parameters (nominal model - intentionally different from true)
        M    = 500       % vehicle mass [kg]
        I_z  = 600       % vehicle moment of inertia (yaw axis) [kg*m^2]
        g    = 9.81      % gravitation [m/s^2]
        l_f  = 0.9       % distance of front axle to CoG [m]
        l_r  = 1.5       % distance of rear axle to CoG [m]
        
        % Steering limits
        deltamax = deg2rad(30)   % maximum steering amplitude [rad]
        
        % Linear tire model parameters (cornering stiffness)
        c_f = 14000 * 2.5  % front cornering stiffness [N/rad]
        c_r = 14000 * 2.5  % rear cornering stiffness [N/rad]
    end
    
    properties(Constant)
        % GP integration matrices
        % Dynamics: xk+1 = fd(xk,uk) + Bd*(d(z)+w)
        % where z = [Bz_x*x; Bz_u*u]
        
        Bz_x = [0 1 0 0;    % extract [vy; r] from [vx; vy; psi; r]
            0 0 0 1]
        Bz_u = 1            % extract steering angle for GP input
        
        % Bd matrix for tire force injection
        % Assumes small angle: cos(delta) ≈ 1
        % vy_dot = (1/M) * (F_y_f*cos(delta) + F_y_r) ≈ (1/M) * (F_y_f + F_y_r)
        % r_dot = (1/I_z) * (l_f*F_y_f*cos(delta) - l_r*F_y_r) ≈ (1/I_z) * (l_f*F_y_f - l_r*F_y_r)
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
        function obj = MotionModelGP_Bicycle_nominal(d, sigmaw)
            %------------------------------------------------------------------
            %   Constructor
            %   Inputs:
            %       d       - GP evaluation function handle: [mu,var]=d(z)
            %       sigmaw  - Process noise covariance (2x2)
            %------------------------------------------------------------------
            % Call superclass constructor
            obj = obj@MotionModelGP(d, sigmaw);
            
            % Add CODEGEN folder to path for auto-generated Jacobians
            codegen_path = fullfile(fileparts(mfilename('fullpath')), '..', 'CODEGEN');
            addpath(codegen_path)
            
            % Report vehicle characteristics
            obj.analyseVehicle();
        end
        
        function xdot = f(obj, x, u)
            %------------------------------------------------------------------
            %   Continuous-time dynamics (nominal model with linear tires)
            %   Implements dynamic bicycle model
            %------------------------------------------------------------------
            
            % Switch to symbolic variables for gradient code generation
            if isa(x, 'sym')
                syms M I_z l_f l_r deltamax c_f c_r real
            else
                M = obj.M;
                I_z = obj.I_z;
                l_f = obj.l_f;
                l_r = obj.l_r;
                deltamax = obj.deltamax;
                c_f = obj.c_f;
                c_r = obj.c_r;
            end
            
            %--------------------------------------------------------------
            % Extract state and input
            %--------------------------------------------------------------
            vx = x(1);      % longitudinal velocity [m/s]
            vy = x(2);      % lateral velocity [m/s]
            psi = x(3);      % yaw angle [rad]
            r  = x(4);      % yaw rate [rad/s]
            
            delta = u(1);   % steering angle [rad]
            
            %--------------------------------------------------------------
            % Saturate steering input
            %--------------------------------------------------------------
            if isa(x, 'sym')
                % Use smooth differentiable saturation for gradient computation
                delta = obj.sclip(delta, -deltamax, deltamax);
            else
                % Use hard saturation for simulation
                delta = obj.clip(delta, -deltamax, deltamax);
            end
            
            %--------------------------------------------------------------
            % Calculate slip angles
            %--------------------------------------------------------------
            % Front slip angle
            % alpha_f = atan2(vy + l_f*r, vx) - delta;
            alpha_f = delta - (l_f*r + vy)/vx;  % small angle approx.
            % alpha_f = -(delta - (l_f*r + vy)/vx);  % small angle approx.
            % Rear slip angle
            % alpha_r = atan2(vy - l_r*r, vx);
            alpha_r = (l_r*r - vy)/vx;
            % alpha_r = (vy - l_r*r)/vx;  % small angle approx.
            %--------------------------------------------------------------
            % Calculate tire forces (LINEAR MODEL)
            %--------------------------------------------------------------
            Fy_f = c_f * alpha_f;   % front lateral force
            Fy_r = c_r * alpha_r;   % rear lateral force
            fprintf('Debug: Fy_f: %.2f, Fy_r: %.2f\n', Fy_f, Fy_r);
            %--------------------------------------------------------------
            % Dynamic bicycle equations
            %--------------------------------------------------------------
            % Longitudinal dynamics (constant vx assumption for simplicity)
            vx_dot = 0;  % centrifugal coupling
            
            % Lateral dynamics
            vy_dot = (1/M) * (Fy_f*cos(delta) + Fy_r);
            
            % Yaw dynamics
            psi_dot = r;
            r_dot = (1/I_z) * (l_f*Fy_f - l_r*Fy_r);
            
            %--------------------------------------------------------------
            % Assemble state derivative
            %--------------------------------------------------------------
            xdot = [vx_dot; vy_dot; psi_dot; r_dot];
            
            %--------------------------------------------------------------
            % Check model integrity
            %--------------------------------------------------------------
            if ~isa(x, 'sym')
                if any(isnan(xdot)) || any(isinf(xdot)) || any(imag(xdot)~=0)
                    error('Bicycle Model evaluated to Inf or NaN... CHECK MODEL!')
                end
            end
        end
        
        
        function gradx = gradx_f(obj, x, u)
            %------------------------------------------------------------------
            %   Gradient of continuous dynamics w.r.t. state
            %   out: gradx = ∂f/∂x  (n×n matrix)
            %------------------------------------------------------------------
            params = [obj.M obj.I_z obj.l_f obj.l_r obj.deltamax ...
                obj.c_f obj.c_r]';
            
            % Try to use auto-generated function
            if exist('bicycle_gradx_f', 'file')
                gradx = bicycle_gradx_f(x, u, params);
            else
                % Fallback: numerical differentiation
                warning('bicycle_gradx_f not found. Using numerical differentiation. Run generate_grad_functions()');
                gradx = obj.numerical_gradx(x, u);
            end
        end
        
        
        function gradu = gradu_f(obj, x, u)
            %------------------------------------------------------------------
            %   Gradient of continuous dynamics w.r.t. input
            %   out: gradu = ∂f/∂u  (m×n matrix)
            %------------------------------------------------------------------
            params = [obj.M obj.I_z obj.l_f obj.l_r obj.deltamax ...
                obj.c_f obj.c_r]';
            
            % Try to use auto-generated function
            if exist('bicycle_gradu_f', 'file')
                gradu = bicycle_gradu_f(x, u, params);
            else
                % Fallback: numerical differentiation
                warning('bicycle_gradu_f not found. Using numerical differentiation. Run generate_grad_functions()');
                gradu = obj.numerical_gradu(x, u);
            end
        end
        
        
        function generate_grad_functions(obj)
            %------------------------------------------------------------------
            %   Generate MATLAB functions for analytical Jacobians
            %   Uses Symbolic Math Toolbox
            %
            %   Usage:
            %       model = MotionModelGP_Bicycle_nominal([], []);
            %       model.generate_grad_functions()
            %------------------------------------------------------------------
            fprintf('Generating Jacobian functions...\n');
            
            % Define symbolic variables
            % syms vx vy psi r real
            vx = sym('vx', 'real');
            vy = sym('vy', 'real');
            psi = sym('psi', 'real');
            r = sym('r', 'real');
            x = [vx; vy; psi; r];
            
            syms delta real
            u = delta;
            
            syms M I_z l_f l_r deltamax c_f c_r real
            params = [M I_z l_f l_r deltamax c_f c_r]';
            
            % Compute symbolic dynamics
            xdot = obj.f(x, u);
            
            % Compute Jacobians
            fprintf('  Computing gradx (∂f/∂x)...\n');
            gradx = jacobian(xdot, x)';
            
            fprintf('  Computing gradu (∂f/∂u)...\n');
            gradu = jacobian(xdot, u)';
            
            % Create CODEGEN directory if needed
            folder = fullfile(fileparts(mfilename('fullpath')), '..', 'CODEGEN');
            if ~exist(folder, 'dir')
                mkdir(folder);
            end
            addpath(folder)
            
            % Generate MATLAB function files
            fprintf('  Writing bicycle_gradx_f.m...\n');
            matlabFunction(gradx, 'Vars', {x, u, params}, ...
                'File', fullfile(folder, 'bicycle_gradx_f'), ...
                'Optimize', true);
            
            fprintf('  Writing bicycle_gradu_f.m...\n');
            matlabFunction(gradu, 'Vars', {x, u, params}, ...
                'File', fullfile(folder, 'bicycle_gradu_f'), ...
                'Optimize', true);
            
            fprintf('FINISHED! Functions bicycle_gradx_f and bicycle_gradu_f generated!\n');
        end
        
        
        function [F_y_f, F_y_r] = getTireForces(obj, x, u)
            %------------------------------------------------------------------
            %   Calculate tire lateral forces (nominal linear model)
            %   Returns: F_y_f (front), F_y_r (rear)
            %------------------------------------------------------------------
            vx = x(1);
            vy = x(2);
            r = x(4);
            delta = u(1);
            
            % Saturate steering
            delta = obj.clip(delta, -obj.deltamax, obj.deltamax);
            
            % Slip angles (small angle approximation)
            alpha_f = delta - (obj.l_f*r + vy)/vx;
            alpha_r = (obj.l_r*r - vy)/vx;
            
            % Linear tire forces
            F_y_f = obj.c_f * alpha_f;
            F_y_r = obj.c_r * alpha_r;
            
        end
        
        function analyseVehicle(obj)
            %------------------------------------------------------------------
            %   Print vehicle stability analysis (understeer/oversteer)
            %------------------------------------------------------------------
            fprintf('\n=== Bicycle Model (Nominal) Created ===\n');
            
            % Wheelbase
            L = obj.l_f + obj.l_r;
            
            % Understeer gradient (Eigenlenkgradient)
            EG = obj.M/L * (obj.l_r/obj.c_f - obj.l_f/obj.c_r);
            
            fprintf('  Wheelbase L = %.2f [m]\n', L);
            fprintf('  Mass M = %.0f [kg]\n', obj.M);
            fprintf('  Yaw inertia Iz = %.0f [kg*m^2]\n', obj.I_z);
            fprintf('  Understeer gradient EG = %.2f [deg/g]\n', rad2deg(EG)*obj.g);
            
            if EG > 0
                fprintf('  >> Vehicle is UNDERSTEER (stable)\n');
                v_ch = sqrt(L/EG);
                fprintf('  >> Characteristic velocity v_ch = %.1f [m/s]\n', v_ch);
            elseif EG < 0
                fprintf('  >> Vehicle is OVERSTEER (potentially unstable)\n');
                v_cr = sqrt(obj.c_f*obj.c_r*L^2 / (obj.M*(obj.c_f*obj.l_f - obj.c_r*obj.l_r)));
                fprintf('  >> Critical velocity v_cr = %.1f [m/s]\n', v_cr);
            else
                fprintf('  >> Vehicle is NEUTRAL STEER\n');
            end
            fprintf('======================================\n\n');
        end
    end
    
    
    %----------------------------------------------------------------------
    % Numerical differentiation fallbacks
    %----------------------------------------------------------------------
    methods(Access = private)
        function gradx = numerical_gradx(obj, x, u)
            % Numerical gradient w.r.t. state using finite differences
            eps = 1e-6;
            n = obj.n;
            gradx = zeros(n, n);
            f0 = obj.f(x, u);
            
            for i = 1:n
                x_pert = x;
                x_pert(i) = x_pert(i) + eps;
                f_pert = obj.f(x_pert, u);
                gradx(:, i) = (f_pert - f0) / eps;
            end
        end
        
        function gradu = numerical_gradu(obj, x, u)
            % Numerical gradient w.r.t. input using finite differences
            eps = 1e-6;
            m = obj.m;
            n = obj.n;
            gradu = zeros(n, m);
            f0 = obj.f(x, u);
            
            for i = 1:m
                u_pert = u;
                u_pert(i) = u_pert(i) + eps;
                f_pert = obj.f(x, u_pert);
                gradu(:, i) = (f_pert - f0) / eps;
            end
        end
    end
    
    
    %----------------------------------------------------------------------
    % Smooth alternatives for non-smooth functions (for gradient computation)
    %----------------------------------------------------------------------
    methods
        function x = sclip(obj, x, lb, ub)
            % Smooth (differentiable) saturation function
            x = x.*obj.gez(x-lb).*obj.lez(x-ub) + ub*obj.gez(x-ub) + lb*obj.lez(x-lb);
        end
    end
    
    methods(Static)
        function x = clip(x, lb, ub)
            % Hard saturation (non-differentiable)
            x = min(max(x, lb), ub);
        end
        
        function x = gez(x)
            % Smooth approximation of (x >= 0)
            alpha = 50;
            x = (1 + exp(-alpha*x)).^-1;
        end
        
        function x = lez(x)
            % Smooth approximation of (x <= 0)
            alpha = 50;
            x = 1 - (1 + exp(-alpha*x)).^-1;
        end
    end
end
