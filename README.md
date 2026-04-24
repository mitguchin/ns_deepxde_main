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

