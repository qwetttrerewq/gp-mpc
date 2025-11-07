%--------------------------------------------------------------------------
% Programed by:
%   - Lucas Rath (lucasrm25@gmail.com)
%   - Modified: Simplified 3-state dynamics (V_vx, V_vy, psi_dot only)
%--------------------------------------------------------------------------

classdef MotionModelGP_test < MotionModelGP
    %--------------------------------------------------------------------------
    %   xk+1 = fd(xk,uk) + Bd * ( d(zk) + w ),
    %
    %       where: zk = [Bz_x*xk ; Bz_u*uk],
    %              d ~ N(mean_d(zk),var_d(zk))
    %              w ~ N(0,sigmaw)
    %
    %   SIMPLIFIED MODEL - Body Frame Dynamics with Yaw Angle:
    %
    %   x = [V_vx             (longitudinal velocity in vehicle coordinates)
    %        V_vy             (lateral velocity in vehicle coordinates)
    %        psi              (yaw angle)
    %        psi_dot          (yaw rate)
    %        ]
    %
    %   u = [delta          (steering angle),
    %        T              (wheel torque gain),  -1=max.braking, 1=max acc.
    %       ]
    %
    %--------------------------------------------------------------------------
    
    properties(Constant)
        M    = 500      % vehicle mass
        I_z  = 600      % vehicle moment of inertia (yaw axis)
        g    = 9.81     % gravitation
        l_f  = 0.9      % distance of the front wheel to the center of mass
        l_r  = 1.5      % distance of the rear wheel to the center of mass
        
        deltamax    = deg2rad(30)   % maximum steering amplitude
        
        maxbrakeWForce = 8000 % = 2*g*M;  % allow ~ 2g brake
        maxmotorWForce = 4000 % = 1*g*M;  % allow ~ 1g acc
        
        % Pacejka lateral dynamics parameters
        B_f = 0.4;              % stiffnes factor (Pacejka) (front wheel)
        C_f = 8;                % shape factor (Pacejka) (front wheel)
        D_f = 4560.4;           % peak value (Pacejka) (front wheel)
        E_f = -0.5;             % curvature factor (Pacejka) (front wheel)
        B_r = 0.45;             % stiffnes factor (Pacejka) (rear wheel)
        C_r = 8;                % shape factor (Pacejka) (rear wheel)
        D_r = 4000;             % peak value (Pacejka) (rear wheel)
        E_r = -0.5;             % curvature factor (Pacejka) (rear wheel)
    end
    
    properties(Constant)
        % keep in mind the dimensions:  xk+1 = fd(xk,uk) + Bd*(d(z)+w)),
        % where z = [Bz_x*x;Bz_u*u]
        Bz_x = eye(4)           % All 4 states used for GP input
        Bz_u = eye(2)           % Both inputs used for GP input
        Bd = eye(4)             % Direct disturbance on all 4 states
        n  = 4   % number of outputs x(t) - now 4 states
        m  = 2   % number of inputs u(t) - reduced to 2
        nz = 6   % dimension of z(t) = [x; u] = [4; 2]
        nd = 4   % output dimension of d(z)
    end
    
    methods(Static)
        function x = clip(x,lb,ub)
            % standard nonsmooth clip (saturation) function
            x = min(max(x,lb),ub);
        end
    end
    
    methods
        function obj = MotionModelGP_test(d,sigmaw)
            %------------------------------------------------------------------
            %   object constructor. Create model and report model stability
            %   analysis
            %------------------------------------------------------------------
            % call superclass constructor
            obj = obj@MotionModelGP(d,sigmaw);
        end
        
        function xdot = f (obj, x, u)
            %------------------------------------------------------------------
            %   Continuous time dynamics of the single track (simplified):
            %   Body frame dynamics only - no global position tracking
            %------------------------------------------------------------------
            
            %--------------------------------------------------------------
            % Model parameters
            %--------------------------------------------------------------
            g = obj.g;
            M = obj.M;
            I_z  = obj.I_z;
            l_f  = obj.l_f;
            l_r  = obj.l_r;
            deltamax = obj.deltamax;
            maxbrakeWForce = obj.maxbrakeWForce;
            maxmotorWForce = obj.maxmotorWForce;
            B_f = obj.B_f;
            C_f = obj.C_f;
            D_f = obj.D_f;
            E_f = obj.E_f;
            B_r = obj.B_r;
            C_r = obj.C_r;
            D_r = obj.D_r;
            E_r = obj.E_r;
            
            %--------------------------------------------------------------
            % State Vector (4 states now)
            %--------------------------------------------------------------
            V_vx        = x(1);     % longitudinal velocity
            V_vy        = x(2);     % lateral velocity
            psi         = x(3);     % yaw angle
            psi_dot     = x(4);     % yaw rate (r)
            
            %--------------------------------------------------------------
            % Inputs (simplified)
            %--------------------------------------------------------------
            delta     = u(1);   % steering angle
            T         = u(2);   % wheel torque gain,  -1=max.braking, 1=max acc.
            
            %--------------------------------------------------------------
            % Saturate inputs
            %--------------------------------------------------------------
            % saturate steering angle
            delta = obj.clip(delta, -deltamax, deltamax);
            
            % saturate pedal input
            T = obj.clip( T, -1, 1);
            
            %--------------------------------------------------------------
            % Wheel slip angles
            %--------------------------------------------------------------
            a_r = atan2(V_vy-l_r*psi_dot,V_vx);
            a_f = atan2(V_vy+l_f*psi_dot,V_vx) - delta;
            
            %--------------------------------------------------------------
            % Tyre forces
            %--------------------------------------------------------------
            % desired total wheel torque to be applied
            totalWForce = T * ( (T>0)*maxmotorWForce+(T<0)*maxbrakeWForce*sign(V_vx) );
            % longitudinal wheel torque distribution
            zeta = 0.5;
            
            % Wheel forces in wheel coordinates
            W_Fx_r = zeta * totalWForce;
            W_Fx_f = (1-zeta) * totalWForce;
            % Pacejka tyre lateral dynamics
            W_Fy_r = D_r*sin(C_r*atan(B_r*a_r-E_r*(B_r*a_r -atan(B_r*a_r))));
            W_Fy_f = D_f*sin(C_f*atan(B_f*a_f-E_f*(B_f*a_f -atan(B_f*a_f))));
            
            % Wheel forces in vehicle coordinates
            V_Fx_r = W_Fx_r;
            V_Fx_f = W_Fx_f;
            V_Fy_r = - W_Fy_r;
            V_Fy_f = - W_Fy_f;
            
            % fprintf('Fy_f: %.2f, Fy_r: %.2f\n', V_Fy_f, V_Fy_r);
            %--------------------------------------------------------------
            % Calculate state space time derivatives (4 states)
            %--------------------------------------------------------------
            V_vx_dot = 1/M * (V_Fx_r + V_Fx_f*cos(delta) - V_Fy_f*sin(delta) + M*V_vy*psi_dot);
            V_vy_dot = 1/M * (V_Fy_r + V_Fx_f*sin(delta) + V_Fy_f*cos(delta) - M*V_vx*psi_dot);
            % psi_dot (yaw angle derivative = yaw rate r, already extracted from state)
            r_dot = 1/I_z * (V_Fy_f*l_f*cos(delta) + V_Fx_f*l_f*sin(delta) - V_Fy_r*l_r);
            
            %--------------------------------------------------------------
            % write outputs
            %--------------------------------------------------------------
            xdot  = [V_vx_dot; V_vy_dot; psi_dot; r_dot];
            
            %--------------------------------------------------------------
            % check model integrity
            %--------------------------------------------------------------
            if any(isnan(xdot)) || any(isinf(xdot)) || any(imag(xdot)~=0)
                error('Single Track Model evaluated to Inf of NaN... CHECK MODEL!!!')
            end
        end
        
        function gradx = gradx_f(obj, ~, ~)
            %------------------------------------------------------------------
            %   Continuous time dynamics.
            %   out:
            %       gradx: <n,n> gradient of xdot w.r.t. x
            %------------------------------------------------------------------
            gradx = zeros(obj.n);
        end
        
        function gradu = gradu_f(obj, ~, ~)
            %------------------------------------------------------------------
            %   Continuous time dynamics.
            %   out:
            %       gradu: <m,n> gradient of xdot w.r.t. u
            %------------------------------------------------------------------
            gradu = zeros(obj.m,obj.n);
        end
        
    end
end
