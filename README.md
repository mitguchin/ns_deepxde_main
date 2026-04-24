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


Boundary Conditions (BC): 

Inlet ($x = -L/2$): Constant velocity $u = 1, v = 0$.
Outlet ($x = L/2$): Reference pressure $p = 0$ and $v = 0$.
Walls ($y = \pm D/2$): No-slip condition $u = 0, v = 0$.

