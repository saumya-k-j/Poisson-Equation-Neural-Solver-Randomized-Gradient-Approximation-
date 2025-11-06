# 🧮 Neural Approximation of the Poisson Equation Using Randomized Gradient Approximation (RGA)

This repository contains an end-to-end implementation of **Algorithm 2: Randomized Gradient Approximation (RGA)** from Section 7.3 of the referenced paper.  
The goal is to approximate the analytical solution \( u^*(x) \) of the **Poisson equation** by training a neural network \( u_\theta(x) \) with an unbiased stochastic gradient estimator.

> **Author:** *Ssaumya Jaiswal*  
> **Institution:** Penn State University — Department of Computer Science and Mathematics  
> **Keywords:** Poisson Equation · Neural Solvers · Stochastic Approximation · Randomized Gradient Algorithms

---

## 📘 1. Problem Formulation

We consider a 1-D stochastic kernel defined on \([0,1]\):

\[
X_{n+1} = \frac{X_n + Z}{2}, \quad Z \sim \text{Bernoulli}\!\left(\tfrac{1}{2}\right),
\]

with reward functions

\[
r(x) =
\begin{cases}
x, & \text{linear case}, \\
x^2, & \text{quadratic case}.
\end{cases}
\]

The analytical Poisson solutions are

\[
u^*(x) =
\begin{cases}
2x, & r(x)=x, \\
\frac{4}{3}x^2 + \frac{2}{3}x, & r(x)=x^2.
\end{cases}
\]

The **Poisson equation** is approximated by learning a parametric function \(u_\theta(x)\) that minimizes the expected residual under the kernel.

---

## 🔬 2. Randomized Gradient Approximation (RGA)

The RGA algorithm provides an **unbiased estimator** of the gradient of the Poisson residual by using *two independent Markov chains*.

For samples \( (X_0,X_1,X_2) \) and \( (X_0,X'_1,X'_2) \) drawn independently from the kernel:

\[
\begin{aligned}
g &= \big(u_\theta(X_0) - 2u_\theta(X_1) + u_\theta(X_2)\big)
    - \big(r(X_0) - r(X_1)\big), \\
h &= \big(u_\theta(X_0) - 2u_\theta(X'_1) + u_\theta(X'_2)\big), \\
\mathcal{L}(\theta) &= 2\,\mathbb{E}[\,g_{\text{detach}}\,h\,].
\end{aligned}
\]

`g` is detached from the gradient graph to ensure unbiasedness, and the update rule becomes a form of **Monte-Carlo Poisson residual minimization**.

---

## ⚙️ 3. Implementation Overview

| Component | Description |
|------------|-------------|
| **Framework** | PyTorch (manual gradient descent) |
| **Network** | Single hidden-layer perceptron with ReLU activation |
| **Sampling** | Two-step Markov kernel \( X_{n+1}=(X_n+Z)/2 \) |
| **Training** | Direct parameter update: \( \theta \leftarrow \theta - \alpha\nabla_\theta \mathcal{L} \) |
| **Evaluation** | MSE, MAE, and integral of squared error with mean alignment |

---

## 🧩 4. Repository Structure
Neural-Poisson-RGA-1D-Interactive/
├── src/
│ ├── poisson_rga.py # core training algorithm + metrics + plots
│ └── metrics_experiment.py # hyperparameter sweeps → CSV
├── app/
│ └── streamlit_app.py # interactive visualization dashboard
├── data/
│ └── metrics_table.csv # logged experimental metrics
├── results/
│ ├── plots_T/ # iteration comparison
│ ├── plots_alpha/ # learning-rate comparison
│ └── plots_batch/ # batch-size comparison
├── report/ # export your PDF analysis here
├── requirements.txt
└── README.md

📚 11. References

Deep Learning for Markov Chains: Lyapunov Functions, Poisson’s Equation, and Stationary Distributions
Yanlin Qu1*, Jose Blanchet2 and Peter Glynn2 1*Columbia Business School.

Sutton & Barto (2018). Reinforcement Learning: An Introduction.

Zhang et al. (2020). Deep PDE Solvers via Monte Carlo Methods.

✍️ 12. Author

Ssaumya Jaiswal
