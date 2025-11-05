# CLAUDE.md

This document defines the context and coding rules that any **code assistant (e.g., Claude Code)** must follow when working in this repository.

---

## Project Overview

This project applies a **Gaussian Process (GP)–augmented Model Predictive Control (GP-MPC)** framework to **vehicle cornering performance**.
The **nominal model** is a **dynamic bicycle model** with **linear tire stiffness**, while the **true model** uses the **Pacejka tire model**.
The GP learns the **residual dynamics** between the true and nominal models, capturing nonlinear tire effects (saturation, combined slip, load transfer), and propagates its **uncertainty** during prediction.
In the extended formulation, **chance-constrained MPC** ensures probabilistic safety margins.

**Core research objectives**

1. **Residual Dynamics Learning:** Learn ([\Delta\dot v_y,\ \Delta\dot r] = f*{\text{true}} - f*{\text{nom}}) using a **sparse GP**
2. **Uncertainty Propagation:** Propagate GP predictive variance through the state covariance via **first-order EKF approximation**
3. **Chance Constraints:** Apply conservative transformation (|\mu|+\lambda\_\epsilon\sigma) to enforce probabilistic safety bounds

**Supported MATLAB Version:** R2019a or later

---

## Execution

### Main Simulation

- **Cornering Performance Demo (Default Example)**

  ```matlab
  main_cornering.m
  ```

  Runs a **cornering stability/path-tracking MPC simulation** using the nominal (linear tire) and GP-augmented models.
  The track is optional — simple references such as **constant heading**, **circular**, or **figure-eight paths** are sufficient.

- **Auxiliary Tests**

  - `test-files/test_GP.m` — GP regression and hyperparameter verification
  - `test-files/tyres_diff.m` — Linear vs. Pacejka tire comparison
  - `test-files/genFigs.m` — Plot generation

---

## Code Architecture

### 1) Dynamic Models (`classes/MotionModelGP*.m`)

All models inherit from the abstract class **`MotionModelGP`**.

[
x_{k+1} = f_d(x_k,u_k) + B_d (d(z_k) + w_k), \qquad
z_k = \begin{bmatrix} B_{z_x}x_k \ B_{z_u}u_k \end{bmatrix}, \quad
w_k \sim \mathcal{N}(0,\Sigma_w)
]

- `fd(xk, uk)`: discrete **nominal dynamics**
- `d(zk)`: **GP-predicted residual dynamics** (mean & variance)
- `z = [v_x, v_y, r, \delta]^\top` by default (optionally replace with (\alpha_f,\alpha_r))
- `Bd`: injection channel for residuals, typically ([\Delta\dot v_y,\Delta\dot r])

**Implemented Classes**

- `MotionModelGP_Bicycle_nominal.m`

  - Nominal **linear tire model**: (F*{y,f}=C*{\alpha f}\alpha*f,; F*{y,r}=C\_{\alpha r}\alpha_r)
  - Slip angles: (\alpha_f=\delta-\arctan\frac{v_y+l_f r}{v_x},\ \alpha_r=-\arctan\frac{v_y-l_r r}{v_x})
  - Uses standard dynamic bicycle formulation (mass, inertia, geometry)

- `MotionModelGP_Bicycle_true.m`

  - True **Pacejka tire model:** (F_y = D\sin(C\arctan(B\alpha - E(B\alpha-\arctan(B\alpha)))))
  - Same vehicle structure, nonlinear tire substitution only

**Required Methods (for subclassing)**

- `f(x,u)` — Continuous-time dynamics
- `gradx_f(x,u)` / `gradu_f(x,u)` — Jacobians
- Class constants: `Bd`, `Bz_x`, `Bz_u`, `n`, `m`, `nd`, `nz`

> Recommended state/input:
> (x=[v_x,v_y,r]^\top) (+ optional position/yaw), (u=[\delta]^\top)

---

### 2) Gaussian Process (`classes/GP.m`)

Supports multi-output regression (independent or coregionalized), using **ARD-RBF + White** kernel, with **sparse dictionary** management.

- Online learning: `add(X,Y)`, `updateModel()`
- Prediction: `[mu_y, var_y] = eval(x)`
- Hyperparameter optimization: `optimizeHyperParams('fmincon' | 'ga')`
- Dictionary pruning by proximity/information gain (max size `Nmax`)

**I/O Convention**

- **Input (z)**: ([v_x,v_y,r,\delta]^\top)
- **Output (d(z))**:
  [
  [\Delta\dot v_y,\ \Delta\dot r] =
  [\dot v_y^{\text{true}}-\dot v_y^{\text{nom}},
  \dot r^{\text{true}}-\dot r^{\text{nom}}]
  ]

---

### 3) Nonlinear MPC (`classes/NMPC.m`)

Formulation:

```
MIN  Σ_{i=0:N-1} fo(ti, xi, ui, r) + fend(tN, xN, r)
s.t. xi+1 = E[f(xi, ui)]             (GP mean + variance propagation)
     h(xi, ui) = 0
     g(xi, ui) ≤ 0                   (Chance constraints via conservative form)
```

- Warm-start: `uguess`, `eguess`
- EKF-based covariance propagation
- Chance constraint: (|\mu|+\lambda\_\epsilon\sigma \le \text{bound})

---

## Cornering Scenario Formulation

### Dynamics

[
\begin{aligned}
\dot v_x &= r v_y + \tfrac{1}{m}(F_{x,f}\cos\delta - F_{y,f}\sin\delta + F_{x,r}) \
\dot v_y &= -r v_x + \tfrac{1}{m}(F_{y,f}\cos\delta + F_{x,f}\sin\delta + F_{y,r}) \
\dot r &= \tfrac{1}{I_z}(l_f(F_{y,f}\cos\delta + F_{x,f}\sin\delta) - l_r F_{y,r})
\end{aligned}
]

- Nominal: (F*{y,\cdot}=C*{\alpha\cdot}\alpha\_\cdot)
- True: (F*{y,\cdot}=\text{Pacejka}(\alpha*\cdot))

### GP Augmentation

[
\dot x = f_{\text{nom}}(x,u) + B_d,\mu_d(z),\qquad
\Sigma_{x+}\approx J,\mathrm{blkdiag}(\Sigma_x,\Sigma_d(z),\Sigma_w),J^\top
]

### Cost Function

- Simple heading or yaw control:
  [
  J=\sum*{i=0}^{N-1}
  (q*\psi e*{\psi,i}^2 + q_r r_i^2 + q*\delta \delta*i^2 + q*{\Delta\delta}(\Delta\delta_i)^2

* q*v(v*{x,i}-v\_{x,\text{ref}})^2)
  ]

- For track-based control: replace by contour/lag error formulation

### Constraints

[
|\delta|\le\delta_{\max},\quad |\Delta\delta|\le\Delta\delta_{\max},\quad
|\beta|\le\beta_{\max},\quad
|\mu_\beta|+\lambda_\epsilon\sigma_\beta \le \beta_{\max}
]

---

## Simulation Loop Overview

1. **Model Setup**

   ```matlab
   trueModel = MotionModelGP_Bicycle_true([], Σ_w);
   nomModel  = MotionModelGP_Bicycle_nominal([], []);
   estModel  = MotionModelGP_Bicycle_nominal(@d_GP.eval, Σ_w);
   ```

2. **MPC Initialization**
   Define `f = @(x,Σ,u) estModel.xkp1(x, Σ, u, dt)`; configure cost, constraints, RTI/SQP options.

3. **Main Loop (k=0..K)**

   - (u_k = \text{MPC.optimize}(xhat_k))
   - True propagation: (x\_{k+1}=\text{trueModel.xkp1}(...))
   - Nominal propagation: (x\_{k+1}^{\text{nom}}=\text{nomModel.xkp1}(...))
   - Residual: (d*{\text{est}} = B_d^{\dagger}(x*{k+1}-x\_{k+1}^{\text{nom}}))
   - Update GP: `d_GP.add(z_k, d_est); d_GP.updateModel()`
   - Log results and plots

---

## Directory Structure

```
GP-MPC-Cornering/
├── classes/
│   ├── MotionModelGP.m
│   ├── MotionModelGP_Bicycle_nominal.m
│   ├── MotionModelGP_Bicycle_true.m
│   ├── GP.m
│   └── NMPC.m
├── functions/
│   ├── sigmaEllipse2D.m
│   └── logdet.m
├── CODEGEN/
├── simresults/
├── test-files/
│   ├── test_GP.m
│   ├── tyres_diff.m
│   └── genFigs.m
├── main_cornering.m
└── Report.pdf / Presentation.pdf
```

---

## Key Parameters (Example)

```matlab
dt = 0.05;
N  = 15;
maxiter = 30;

loadPreTrainedGP = false;
useGP = true;
trainGPonline = true;

% x = [vx; vy; r];   u = [delta];
% z = [vx; vy; r; delta];   d = [Δvydot; Δrdot];
```

---

## Mathematical Core

### Residual Learning

[
d(z)=
\begin{bmatrix}
\dot v_y^{\text{true}}-\dot v_y^{\text{nom}}\
\dot r^{\text{true}}-\dot r^{\text{nom}}
\end{bmatrix},\quad
z=[v_x,v_y,r,\delta]^\top
]

### Uncertainty Propagation (EKF 1st order)

[
\mu_{x+}=f_{\text{nom}}(\mu_x,u)+B_d,\mu_d(z),\quad
\Sigma_{x+}\approx J,\mathrm{blkdiag}(\Sigma_x,\Sigma_d(z),\Sigma_w),J^\top
]

### Chance Constraint (Conservative Form)

[
|\mu_\beta|+\lambda_\epsilon\sigma_\beta \le \beta_{\max},\quad
\lambda_\epsilon=\Phi^{-1}(1-\epsilon)
]

---

## Development Workflow

1. **Add new model:** subclass `MotionModelGP` and implement `f`, Jacobians, and matrices
2. **Train GP:** offline pretraining → online update with dictionary control
3. **Integrate into MPC:** use `xkp1()` in prediction model
4. **Evaluate:** compare GP-OFF vs GP-ON (RMSE, constraint violations, computation time)

---

## Hyperparameter Tuning

```matlab
d_GP.optimizeHyperParams('fmincon');
% or
d_GP.optimizeHyperParams('ga');
```

- Inspect relative magnitudes of length-scales, signal/noise variance
- Calibrate via coverage rate (e.g., 95% interval accuracy)

---

## Analysis Example

```matlab
predErrorNOgp   = Bd \ (xhat(:,1:k-1) - xnom(:,1:k-1));
dgp             = d_GP.eval( zhat );
predErrorWITHgp = Bd \ ( xhat(:,2:k) - (xnom(:,2:k) + Bd*dgp) );
```

- Visualize: β, r, δ responses; RMS error; constraint violations; computation statistics

---

## Common Pitfalls

1. **Linearization instability:** scaling, Jacobian clipping, smaller dt/N
2. **GP over/under-confidence:** dictionary trimming, noise variance tuning, normalization
3. **Computation cost:** sparse GP, RTI mode, warm-start reuse
4. **Insufficient training coverage:** extend dataset (speed/steering sweep, balanced α range)

---

## Research Alignment (Key Differentiators)

- Same dynamic bicycle structure for both models → residual represents **pure tire nonlinearity**
- GP input (z=[v_x,v_y,r,\delta]) covers nonlinear regions without explicit slip-angle computation
- Track-free evaluation: circular or figure-eight cornering sufficient for validation; path-based tracking optional

---
