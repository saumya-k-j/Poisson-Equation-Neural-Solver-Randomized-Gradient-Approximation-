# Neural Poisson RGA (1D) — Interactive

Neural approximation of the 1‑D **Poisson equation** using **Randomized Gradient Approximation (RGA, Algorithm 2 §7.3)**. Includes a clean training script, an experiment runner that logs metrics to CSV, and a Streamlit app for interactive visualization.

## Quickstart
```bash
pip install -r requirements.txt
python src/poisson_rga.py
# or interactive:
streamlit run app/streamlit_app.py
```
Edit hyperparameters (`T`, `ALPHA`, `BATCH`, `WIDTH`, `REWARD`) at the top of `src/poisson_rga.py`.

## Experiments → CSV
```bash
python src/metrics_experiment.py
```
This writes `data/metrics_table.csv` with columns: `run,T,alpha,batch,mse,mae,integral`.

## Repository Layout
```
Neural-Poisson-RGA-1D-Interactive/
├── app/streamlit_app.py        # interactive viz
├── src/poisson_rga.py          # training + metrics + plot
├── src/metrics_experiment.py   # sweep runner
├── results/plots_*             # put your figures here
├── data/metrics_table.csv      # metrics log (created)
├── report/                     # export your PDF analysis here
├── requirements.txt
└── README.md
```
— Generated 2025-11-06.
