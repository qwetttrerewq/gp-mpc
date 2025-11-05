%--------------------------------------------------------------------------
% CircularPath - Simple reference trajectory generator
%
% Generates circular and figure-8 reference paths for cornering tests
% Lightweight alternative to RaceTrack class
%--------------------------------------------------------------------------

classdef CircularPath
    properties
        type        % 'circular' or 'figure8'
        radius      % [m]
        center      % [x, y] center position
        v_ref       % reference velocity [m/s]
    end

    methods
        function obj = CircularPath(type, radius, center, v_ref)
        %------------------------------------------------------------------
        %   Constructor
        %   Inputs:
        %       type    - 'circular' or 'figure8'
        %       radius  - radius of circle [m]
        %       center  - [x, y] center position [m]
        %       v_ref   - reference velocity [m/s]
        %------------------------------------------------------------------
            if nargin < 4
                v_ref = 15;  % default 15 m/s
            end
            if nargin < 3
                center = [0, 0];
            end
            if nargin < 2
                radius = 50;  % default 50 m radius
            end
            if nargin < 1
                type = 'circular';
            end

            obj.type = type;
            obj.radius = radius;
            obj.center = center;
            obj.v_ref = v_ref;
        end


        function [psi_ref, r_ref, v_ref] = getReference(obj, t)
        %------------------------------------------------------------------
        %   Get reference trajectory at time t
        %   Outputs:
        %       psi_ref - reference yaw angle [rad]
        %       r_ref   - reference yaw rate [rad/s]
        %       v_ref   - reference velocity [m/s]
        %------------------------------------------------------------------
            v_ref = obj.v_ref;

            switch lower(obj.type)
                case 'circular'
                    % Constant radius circular path
                    omega = v_ref / obj.radius;  % angular velocity
                    psi_ref = omega * t;
                    r_ref = omega;

                case 'figure8'
                    % Figure-8 pattern (lemniscate)
                    % Parametric: x = R*sin(ωt), y = R*sin(ωt)*cos(ωt)
                    omega = v_ref / (2*obj.radius);  % slower for figure-8
                    theta = omega * t;

                    % Yaw angle from tangent to curve
                    dx = obj.radius * omega * cos(theta);
                    dy = obj.radius * omega * (cos(2*theta));
                    psi_ref = atan2(dy, dx);

                    % Yaw rate (derivative of psi)
                    % Numerical approximation
                    dt = 0.01;
                    theta_next = omega * (t + dt);
                    dx_next = obj.radius * omega * cos(theta_next);
                    dy_next = obj.radius * omega * (cos(2*theta_next));
                    psi_next = atan2(dy_next, dx_next);
                    r_ref = (psi_next - psi_ref) / dt;

                case 'constant'
                    % Constant heading (straight line)
                    psi_ref = 0;
                    r_ref = 0;

                otherwise
                    error('Unknown path type: %s', obj.type);
            end
        end


        function [x_path, y_path] = getPathPoints(obj, N)
        %------------------------------------------------------------------
        %   Get discretized path points for visualization
        %   Inputs:
        %       N - number of points
        %   Outputs:
        %       x_path, y_path - path coordinates [m]
        %------------------------------------------------------------------
            if nargin < 2
                N = 100;
            end

            theta = linspace(0, 2*pi, N);

            switch lower(obj.type)
                case 'circular'
                    x_path = obj.center(1) + obj.radius * cos(theta);
                    y_path = obj.center(2) + obj.radius * sin(theta);

                case 'figure8'
                    % Lemniscate parametric equations
                    x_path = obj.center(1) + obj.radius * sin(theta);
                    y_path = obj.center(2) + obj.radius * sin(theta) .* cos(theta);

                case 'constant'
                    % Straight line
                    x_path = linspace(0, 100, N);
                    y_path = zeros(1, N);

                otherwise
                    error('Unknown path type: %s', obj.type);
            end
        end


        function plotPath(obj, figHandle)
        %------------------------------------------------------------------
        %   Plot the reference path
        %------------------------------------------------------------------
            if nargin < 2
                figure('Color', 'w');
            else
                figure(figHandle);
            end

            [x_path, y_path] = obj.getPathPoints(200);

            plot(x_path, y_path, 'k--', 'LineWidth', 2);
            hold on; grid on; axis equal;
            xlabel('X [m]');
            ylabel('Y [m]');
            title(sprintf('%s Path (R=%.1f m, v_{ref}=%.1f m/s)', ...
                  obj.type, obj.radius, obj.v_ref));
        end


        function sideslip_ref = getSideslipRef(obj)
        %------------------------------------------------------------------
        %   Calculate reference sideslip angle for circular motion
        %   β_ref = atan(lr * r / vx) for rear-wheel reference point
        %------------------------------------------------------------------
            if strcmp(lower(obj.type), 'circular')
                % For circular motion at constant speed
                r_ref = obj.v_ref / obj.radius;
                l_r = 1.5;  % rear axle distance (from vehicle params)

                % Sideslip angle at CoG
                sideslip_ref = atan(l_r * r_ref / obj.v_ref);
            else
                sideslip_ref = 0;  % assume zero for other patterns
            end
        end
    end


    methods(Static)
        function obj = createCircular(radius, v_ref)
        %------------------------------------------------------------------
        %   Quick constructor for circular path
        %------------------------------------------------------------------
            if nargin < 2, v_ref = 15; end
            if nargin < 1, radius = 50; end
            obj = CircularPath('circular', radius, [0, 0], v_ref);
        end


        function obj = createFigure8(radius, v_ref)
        %------------------------------------------------------------------
        %   Quick constructor for figure-8 path
        %------------------------------------------------------------------
            if nargin < 2, v_ref = 12; end
            if nargin < 1, radius = 40; end
            obj = CircularPath('figure8', radius, [0, 0], v_ref);
        end


        function obj = createConstant(v_ref)
        %------------------------------------------------------------------
        %   Quick constructor for constant heading (straight)
        %------------------------------------------------------------------
            if nargin < 1, v_ref = 20; end
            obj = CircularPath('constant', 0, [0, 0], v_ref);
        end
    end
end
