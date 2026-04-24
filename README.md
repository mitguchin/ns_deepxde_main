# 2D Navier-Stokes Solver using PINNs(ns_deepxde_main)
: This project implements a Physics Informed Neural Network(PINN) to solve the 2D steady-state incompressible Navier-Stokes equations
without the need for traditonal mesh generation. By leveraging the DeepXDE framework, the physical governing equations are directly integrated into the neural nerwork's loss function.

---

1. Problem Statement: Navier-Stokes Equation
The object is to simulate fluid flow within a rectangular domain by minimizing the residuals of the governing physical laws through automatic differentiation.

* Governing Equations
The steady-state incompressible Navier-Stokes and Continuity equations are defined as:


Momentum ($x$-direction):

$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial x} - \frac{\mu}{\rho} \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$


Momentum ($y$-direction):

$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{1}{\rho} \frac{\partial p}{\partial y} - \frac{\mu}{\rho} \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$$


Continuity:

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$


* Simulation Setup Domain:

A rectangle ($L=2, D=1$) centered at the origin.

Physical Parameters: Density ($\rho$) = 1, Dynamic Viscosity ($\mu$) = 1



* Boundary Conditions (BC): 


Inlet ($x = -L/2$): Constant velocity $u = 1, v = 0$.

Outlet ($x = L/2$): Reference pressure $p = 0$ and $v = 0$.

Walls ($y = \pm D/2$): No-slip condition $u = 0, v = 0$.


---

2. Technical Highlights


* Neural Network Architecture

Structure: Fully Connected Neural Network(FNN).

I/O: 2 Inputs ($x,y$) $\rightarrow4 3 Outputs ($u,v,p$)

Depth: 5 hidden layers with 64 neurons each.

Activation: Tanh(Hyperbolic Tangent) is utilized to ensure smooth higher-order derivatives required for the Laplacian
($\nabla^2$) terms in the PDE loss.

* Physics-Informed Training (Autograd)

Using the DeepXDE framework, the physical residuals are optimized:


PDE Loss: Evaluated at 2,000 domain points to enforce fluid conservation laws.

BC Loss: Evaluated at 200 boundary points to satisfy Dirichlet conditions.

Automatic Differentiation: Employs Jacobians & Hessians to calculate exact spatial derivatives without the trucation errors of traditional mesh-based solvers.


---

3. Two stage Optimization Strategy

A hybrid approach was implemented to achieve both robust exploration and high-precision convergence:


* Adam Optimizer:

Epochs: 10,000

Used for the initial training phase to navigate the loss landscape and establish the general flow field.


* L-BFGS Optimizer:

Max Iterations: 3,000

A second-order optimizer that utilizes Hessian approximations to achieve high-precision refinement in the final training stage.


---

4. Result & Visualization

The trained PINN act as a continuous surrogate model, allowing for near-instantaneous inference at any coordinate within the domain.


* Flow Field Prediction:

Predictions for velocity components ($u,v$) and pressure ($p$) were generated across 500,000 sample points.


* Visualization:

High-fidelity contour plots using the $jet$ colormap visualize the velocity gradients near the walls and the pressure drop across the channel.

---


5. Implementation Details


* Framework: DeepXDE with a TensorFlow backend.

* Key Libraries: NumPy for data processing, Matplotlib for scientific visualization.

* Hardware: Optimized for CUDA-enabled GPU acceleration.
