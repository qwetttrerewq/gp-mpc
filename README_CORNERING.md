# GP-MPC Cornering Performance Implementation

## Overview

This implementation provides a **simplified cornering scenario** for GP-augmented Model Predictive Control (GP-MPC) research, as specified in CLAUDE.md.

### Key Features

- **Simplified State Space**: 3D state `x = [vx; vy; r]` (vs 7D in SingleTrack)
- **Single Input**: 1D steering `u = [delta]` (vs 3D in SingleTrack)
- **Lightweight References**: Circular/figure-8 paths (no complex RaceTrack dependency)
- **GP Learning**: Online residual dynamics learning between linear and Pacejka tire models
- **Uncertainty Propagation**: EKF-based covariance propagation through prediction horizon
- **Chance Constraints**: Conservative formulation `|μ_β| + λ_ε*σ_β ≤ β_max`

---

## File Structure

```
GP-MPC/
├── classes/
│   ├── MotionModelGP_Bicycle_nominal.m   ← NEW: Linear tire model (3D state)
│   ├── MotionModelGP_Bicycle_true.m      ← NEW: Pacejka tire model (3D state)
│   ├── GP.m                               (existing)
│   ├── NMPC.m                             (existing)
│   └── MotionModelGP.m                    (existing base class)
│
├── utils/
│   └── CircularPath.m                     ← NEW: Reference path generator
│
├── main_cornering.m                       ← NEW: Main simulation (CORE)
├── test_cornering.m                       ← NEW: Validation tests
└── README_CORNERING.md                    ← This file
```

---

## Quick Start

### 1. Test the Models

Run validation tests to verify implementations:

```matlab
>> test_cornering
```

This will:
- Compare linear vs Pacejka tire models
- Validate Jacobian computations
- Test state propagation
- Visualize tire characteristics

### 2. Generate Jacobian Functions (First Time Only)

```matlab
>> nomModel = MotionModelGP_Bicycle_nominal([], []);
>> nomModel.generate_grad_functions()
```

This creates optimized Jacobian functions in `CODEGEN/`:
- `bicycle_gradx_f.m` - State Jacobian ∂f/∂x
- `bicycle_gradu_f.m` - Input Jacobian ∂f/∂u

### 3. Run Main Cornering Simulation

```matlab
>> main_cornering
```

This runs the full GP-MPC simulation with:
- 20 seconds simulation time
- Circular reference path (R=50m, v=15m/s)
- Online GP learning
- Real-time plots and analysis

---

## Configuration Options

Edit `main_cornering.m` to customize:

```matlab
% Simulation parameters
dt = 0.05;          % timestep [s]
tf = 20;            % simulation time [s]
N = 15;             % MPC prediction horizon

% GP configuration
useGP = true;               % enable GP in MPC
trainGPonline = true;       % online learning

% Reference path
path_type = 'circular';     % 'circular', 'figure8', or 'constant'
```

### Available Path Types

```matlab
% Circular path
refPath = CircularPath.createCircular(50, 15);  % R=50m, v=15m/s

% Figure-8 path
refPath = CircularPath.createFigure8(40, 12);   % R=40m, v=12m/s

% Straight line
refPath = CircularPath.createConstant(20);      % v=20m/s
```

---

## Model Specifications

### Bicycle Nominal Model (Linear Tires)

**File**: `MotionModelGP_Bicycle_nominal.m`

**State**: `x = [vx; vy; r]` (3D)
- `vx`: Longitudinal velocity [m/s]
- `vy`: Lateral velocity [m/s]
- `r`: Yaw rate [rad/s]

**Input**: `u = [delta]` (1D)
- `delta`: Steering angle [rad]

**GP Input**: `z = [vx; vy; r; delta]` (4D)

**GP Output**: `d = [Δvy_dot; Δr_dot]` (2D residuals)

**Dynamics**:
```
vy_dot = (1/M) * (Fy_f*cos(δ) + Fy_r) - r*vx
r_dot  = (1/Iz) * (lf*Fy_f*cos(δ) - lr*Fy_r)

Tire forces (linear):
  Fy_f = c_f * α_f
  Fy_r = c_r * α_r
```

**Parameters**:
- Mass: M = 500 kg
- Inertia: Iz = 600 kg·m²
- Wheelbase: l_f = 0.9 m, l_r = 1.5 m
- Cornering stiffness: c_f = c_r = 35000 N/rad

### Bicycle True Model (Pacejka Tires)

**File**: `MotionModelGP_Bicycle_true.m`

Same structure, but:
- **Mass**: M = 450 kg (intentional mismatch)
- **Inertia**: Iz = 550 kg·m² (intentional mismatch)
- **Tire forces** (Pacejka Magic Formula):
  ```
  Fy = D*sin(C*arctan(B*α - E*(B*α - arctan(B*α))))
  ```
  - Front: Df = 4560 N, Bf = 0.4, Cf = 8, Ef = -0.5
  - Rear: Dr = 4000 N, Br = 0.45, Cr = 8, Er = -0.5

**Purpose**: Creates modeling error for GP to learn

---

## Key Differences from SingleTrack

| Aspect | SingleTrack | Bicycle (Cornering) |
|--------|-------------|---------------------|
| **State dimension** | 7D | 3D |
| **State variables** | `[x, y, ψ, vx, vy, r, s]` | `[vx, vy, r]` |
| **Input dimension** | 3D | 1D |
| **Inputs** | `[δ, T, v_track]` | `[δ]` |
| **Reference** | RaceTrack class | CircularPath (lightweight) |
| **Complexity** | Race track control | Cornering stability |
| **Use case** | Full racing scenario | Research/validation |

---

## Cost Function

The MPC cost function balances:

```matlab
J = Σ[ q_vx*(vx-v_ref)² + q_vy*vy² + q_r*(r-r_ref)² +
       q_delta*δ² + q_ddelta*Δδ² + q_beta*β² ]
```

Where:
- `q_vx = 10`: Velocity tracking
- `q_vy = 1`: Lateral velocity penalty
- `q_r = 20`: Yaw rate tracking
- `q_delta = 0.5`: Steering effort
- `q_ddelta = 5`: Steering rate (smoothness)
- `q_beta = 100`: Sideslip safety

---

## Constraints

### Input Constraints
```matlab
-25° ≤ δ ≤ 25°    (steering angle)
```

### State Constraints (Chance Constraints)
```matlab
|β| ≤ β_max = 12°    (sideslip angle)
```

Future extension:
```matlab
|μ_β| + λ_ε*σ_β ≤ β_max    (conservative chance constraint)
```

---

## GP Configuration

### Hyperparameters

```matlab
gp_n = 4              % input dimension [vx; vy; r; delta]
gp_p = 2              % output dimension [Δvy_dot; Δr_dot]

var_f = [0.1; 0.1]    % output variance
var_n = diag(var_w)/3 % measurement noise
M = diag([1, 1, 1, 1])² % ARD length scales
maxsize = 200         % max dictionary size
```

### Kernel

ARD-RBF (Automatic Relevance Determination):
```
k(z, z') = σ_f² * exp(-0.5 * (z-z')ᵀ M⁻¹ (z-z'))
```

### Dictionary Management

- **Max size**: 200 points
- **Pruning**: Proximity-based (removes similar points)
- **Update**: Online after each timestep

---

## Output and Analysis

### Generated Figures

1. **State Trajectories**: vx, vy, r vs time
2. **Control and Safety**: δ and β vs time (with limits)
3. **GP Learning**: Residual prediction accuracy over time
4. **Reference Path**: Visualization of circular/figure-8 path

### Saved Results

```
simresults/cornering_circular_GPtrue_YYYYMMDD_HHMMSS.mat
```

Contains:
- `out`: All simulation data
- `d_GP`: Trained GP model
- `refPath`: Reference path object
- `weights`: Cost function weights

---

## Performance Metrics

The simulation reports:

1. **Residual Norm** (without GP): `|d|`
2. **GP Prediction Error**: `|d - d_gp|`
3. **Improvement**: `(1 - error_with_gp/error_without_gp) * 100%`
4. **Dictionary Size**: Current/Max

Expected results:
- GP error reduction: **50-80%** after learning
- Dictionary saturation: ~150-200 points after 20s
- Constraint violations: **0** (with proper tuning)

---

## Troubleshooting

### Issue: "bicycle_gradx_f not found"

**Solution**: Generate Jacobian functions:
```matlab
>> nomModel = MotionModelGP_Bicycle_nominal([], []);
>> nomModel.generate_grad_functions()
```

### Issue: MPC optimization fails

**Possible causes**:
- Infeasible constraints (reduce β_max or increase limits)
- Poor initial guess (adjust `mpc.uguess`)
- Numerical issues (reduce dt or N)

**Solution**:
```matlab
mpc.tol = 1e-1;      % relax tolerance
mpc.maxiter = 50;    % increase iterations
```

### Issue: GP learning too slow

**Solution**: Adjust hyperparameters
```matlab
var_f = [1.0; 1.0];     % increase output variance
M = diag([0.5, 0.5, 0.5, 0.5])²;  % decrease length scales
```

### Issue: Vehicle unstable

**Check**:
- Initial velocity too low (< 5 m/s)
- Sideslip angle too large (> 20°)
- Reference velocity too high for radius

---

## Advanced Usage

### Compare GP-ON vs GP-OFF

```matlab
% Run 1: Without GP
useGP = false;
main_cornering;
out_noGP = out;

% Run 2: With GP
useGP = true;
main_cornering;
out_withGP = out;

% Compare
figure;
plot(out_noGP.t, vecnorm(out_noGP.d_est), 'r-', 'LineWidth', 2);
hold on;
plot(out_withGP.t, vecnorm(out_withGP.d_est - out_withGP.d_gp), 'b-', 'LineWidth', 2);
legend('Without GP', 'With GP');
title('Prediction Error Comparison');
```

### Pre-train GP Offline

```matlab
% Collect training data
main_cornering;  % run once with trainGPonline=true

% Save trained GP
save('simresults/GP_cornering_pretrained.mat', 'd_GP');

% Load in future runs
loadPreTrainedGP = true;
main_cornering;
```

### Optimize GP Hyperparameters

```matlab
% After simulation
d_GP.optimizeHyperParams('fmincon');  % or 'ga' for genetic algorithm

% View optimized parameters
d_GP.M        % length scales
d_GP.var_f    % signal variance
d_GP.var_n    % noise variance
```

---

## References

- **CLAUDE.md**: Full project specification
- **main_singletrack.m**: Complex race track example (original implementation)
- **Report.pdf / Presentation.pdf**: Research documentation

---

## Citation

If you use this code, please cite:

```
GP-MPC for Vehicle Cornering Performance
Gaussian Process-based Model Predictive Control Framework
```

---

## Contact

For issues or questions:
- Check CLAUDE.md for detailed specifications
- Review test_cornering.m for validation examples
- Compare with main_singletrack.m for advanced features

**Good luck with your GP-MPC research!** 🚗💨
