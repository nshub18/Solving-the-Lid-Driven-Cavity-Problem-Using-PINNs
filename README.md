
# Solving the 2D Lid-Driven Cavity Problem using PINNs

This repository contains a **Physics-Informed Neural Network (PINN)** implementation to solve the classic 2D Lid-Driven Cavity flow problem. By embedding the governing physical laws directly into the neural network's loss function, this model approximates the steady-state, incompressible Navier-Stokes equations without requiring traditional mesh discretization (like FDM, FVM, or FEM).

---

## 📌 Problem Description

The lid-driven cavity flow is a standard computational fluid dynamics (CFD) benchmark used to study incompressible viscous flow behavior inside a square cavity $[0, 1] \times [0, 1]$. 

### Boundary Conditions:
*   **Top Lid ($y = 1$):** Moves horizontally to the right with a constant velocity ($u = 1.0, v = 0$).
*   **Stationary Walls ($x=0, x=1, y=0$):** No-slip boundary condition ($u = 0, v = 0$).
*   **Fluid Properties:** Kinematic viscosity $\nu = 0.01$, corresponding to a Reynolds number of $Re = 100$.

---

## 🧮 Mathematical Formulation

The flow is governed by the 2D steady-state, incompressible Navier-Stokes equations:

1. **Continuity Equation (Mass Conservation):**
   $$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

2. **X-Momentum Equation:**
   $$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

3. **Y-Momentum Equation:**
   $$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)$$

The PINN is trained to minimize the residuals of these partial differential equations (PDEs) at interior collocation points, alongside the boundary condition errors.

---

## ⚙️ Network Architecture & Setup

*   **Inputs:** Spatial coordinates $(x, y)$
*   **Outputs:** Velocity components $(u, v)$ and pressure $p$
*   **Architecture:** 8 fully connected hidden layers with 20 neurons each
*   **Activation Function:** Hyperbolic tangent (`tanh`)
*   **Optimizer:** Adam ($\text{learning rate} = 10^{-3}$)
*   **Training Epochs:** 30,000

---

## 🚀 How to Run

### Prerequisites
Make sure you have Python 3.10–3.12 installed along with the required libraries:
```bash
pip install tensorflow matplotlib numpy
