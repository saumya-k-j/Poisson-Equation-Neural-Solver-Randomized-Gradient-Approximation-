# 🧠 Neural Poisson RGA (Algorithm 2 Implementation)
A deep learning–based solver for the **Poisson equation** using **Randomized Gradient Approximation (RGA)** — derived from  
*“Deep Learning for Markov Chains: Lyapunov Functions, Poisson’s Equation, and Stationary Distributions”*  
by **Yanlin Qu**, **Jose Blanchet**, and **Peter Glynn** *(Columbia Business School & Stanford University, 2023)*.

---

## 🧩 Overview

**Neural Poisson RGA** is a lightweight research framework that demonstrates how neural networks can approximate solutions to **Poisson’s Equation** via **Monte Carlo–based stochastic gradient estimation**.  
The implementation is built entirely in **PyTorch**, features reproducible experiments, and includes an **interactive Streamlit dashboard** for live visualization.

The project re-creates **Algorithm 2 (RGA)** from the Qu–Blanchet–Glynn paper, showing how *unbiased gradient estimation* enables convergence to the true solution \(u^*\) without directly computing PDE gradients.

---

## ✨ Features

🎓 **Accurate Algorithm 2 reproduction** — faithful to the Qu–Blanchet–Glynn paper.  
⚙️ **Configurable Parameters** — Tune iterations (T), learning rate (α), batch size, and width.  
📊 **Automatic Metrics** — Computes MSE, MAE, and ∫(error²) dx.  
🎛️ **Interactive Dashboard** — Streamlit GUI for live visual convergence.  
📈 **Metrics Logging** — Saves results to CSV for analysis.  
🧮 **Analytical Validation** — Compares learned \(u_\theta(x)\) vs. analytical \(u^*(x)\).

---

## 🧰 Tech Stack

| Component | Purpose |
|------------|----------|
| **Python 3.10+** | Core language |
| **PyTorch** | Neural training and autograd |
| **Matplotlib** | Visualization |
| **Streamlit + Plotly** | Interactive GUI |
| **Pandas / NumPy** | Metrics and data handling |

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Neural-Poisson-RGA-1D-Interactive.git
cd Neural-Poisson-RGA-1D-Interactive
```

### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧮 How to Run

### ▶️ Train the Solver
```bash
python src/poisson_rga.py
```
Plots the learned \(u_\theta(x)\) and analytical \(u^*(x)+c\) (mean-aligned), and prints **MSE**, **MAE**, and **Integrated Error**.

### 🧪 Run Hyperparameter Experiments
```bash
python src/metrics_experiment.py
```
Runs multiple configurations of *T*, α, and batch size, saving results to `data/metrics_table.csv`.

### 🌐 Launch the Interactive App
```bash
streamlit run app/streamlit_app.py
```
Opens a browser dashboard where you can adjust hyperparameters and visualize convergence live.

---

## 📊 Evaluation Metrics

All metrics are computed on a uniform 401-point grid after mean alignment.

| Metric | Formula | Meaning |
|---------|----------|---------|
| **MSE** | \( \frac{1}{N}\sum (u_\theta - u^*)^2 \) | Average squared deviation |
| **MAE** | \( \frac{1}{N}\sum |u_\theta - u^*| \) | Mean absolute deviation |
| **∫(error²)dx** | \( \int_0^1 (u_\theta - u^*)^2 dx \) | Continuous energy of error |

---

## 🧠 The Mathematics Behind It

The model solves a **Poisson equation** associated with a Markov kernel:

$$
X_{n+1} = \frac{X_n + Z}{2}, \qquad Z \sim \mathrm{Bernoulli}\!\left(\tfrac12\right)
$$

with reward functions:

$$
r(x) =
\begin{cases}
x, & \text{(linear case)}\\
x^2, & \text{(quadratic case)}
\end{cases}
$$

and analytical solutions:

$$
u^*(x) =
\begin{cases}
2x, & r(x)=x,\\
\frac{4}{3}x^2 + \frac{2}{3}x, & r(x)=x^2.
\end{cases}
$$

Algorithm 2 uses two independent Markov chains to build an unbiased stochastic gradient estimator:

$$
\mathcal{L}(\theta) = 2\,\mathbb{E}\!\left[(g_\text{detach})\,h\right],
$$

where  
\(g = (u(X_0)-2u(X_1)+u(X_2))-(r(X_0)-r(X_1))\),  
and  
\(h = (u(X_0)-2u(X'_1)+u(X'_2))\).

---

## 📈 Results Summary

| Parameter | Observation |
|------------|--------------|
| **Iterations (T)** | Error decreases rapidly up to ~5 000 iterations then plateaus. |
| **Learning Rate (α)** | 0.05 achieves fast and stable convergence. |
| **Batch Size** | Larger batches (≥ 1024) yield smoother gradients. |

| T | α | MSE | MAE | ∫error² dx |
|:--|:--|:--|:--|:--|
| 500  | 0.01 | 0.0385 | 0.161 | 0.0386 |
| 2 000 | 0.01 | 0.0054 | 0.054 | 0.0054 |
| 5 000 | 0.01 | 0.0026 | 0.037 | 0.0026 |
| 10 000 | 0.01 | 0.0012 | 0.026 | 0.0012 |

---

## 🎛️ Streamlit Dashboard Highlights

| Feature | Description |
|----------|--------------|
| **Reward Type** | Choose between r(x)=x and r(x)=x² |
| **Iterations (T)** | Number of SGD updates |
| **Learning Rate (α)** | Gradient update step |
| **Batch Size / Width** | Controls training stability |
| **Seed Control** | Ensures reproducibility |
| **Live Updates** | Observe uθ(x) vs u*(x)+c in real time |
| **CSV Export** | Download metrics post-training |

---

## 🧭 Future Directions

🚀 Extend to 2-D/3-D Poisson and Laplace Equations  
⚙️ Integrate with PINNs (Physics-Informed Neural Networks)  
📉 Study RGA vs Deterministic Gradient Variance  
🧩 Adaptive Learning Rate Schedulers for Long Horizon Training  

---

## 📚 References

1️⃣ **Qu, Y.**, **Blanchet, J.**, & **Glynn, P. W.** (2023).  
*Deep Learning for Markov Chains: Lyapunov Functions, Poisson’s Equation, and Stationary Distributions.*  
arXiv: [2508.16737](https://arxiv.org/abs/2508.16737)

2️⃣ **Sutton, R. S.** & **Barto, A. G.** (2018). *Reinforcement Learning: An Introduction.* MIT Press.

3️⃣ **Zhang, K.**, **Liu, Q.**, & **Zhu, J.** (2020). *Deep PDE Solvers via Monte Carlo Methods.* NeurIPS 2020.

---

## ✍️ Author

**Ssaumya Jaiswal**  
Department of Computer Science & Mathematics, Penn State University  
📧 ssaumya.jaiswal@psu.edu

---

⭐ **If you found this project helpful, please consider starring it!**
