# 🌌 Quantum‑Enhanced ML Applications

An open‑source project exploring **hybrid quantum–classical machine learning** with a modern, interactive UI.
Built on **Qiskit ≥ 1.x (Sampler API)**, **SciPy optimizers**, and a **Streamlit** app for demos.

> ⚠️ Work in progress — contributions & feedback are welcome!

---

## 🚀 Overview

This repository demonstrates how quantum computing can augment traditional ML workflows on small datasets:

* **Hybrid models**: parameterized quantum circuits trained with classical optimizers.
* **Modern Qiskit**: uses the **Sampler** primitive (no deprecated OpFlow/QuantumInstance).
* **Interactive UI**: Streamlit web app to train, evaluate, and run live inference.
* **Visuals**: confusion matrix, accuracy, circuit preview, and downloadable predictions.

---

## ✨ Features

* 🧩 Parameterized quantum classifier (feature‑encoded rotations + trainable gates).
* ⚛️ **Qiskit Sampler** back‑end and **Aer** simulators.
* 📊 Training & evaluation on the **Iris** dataset.
* 🔮 Inference playground (feature sliders → instant prediction).
* 🧱 Circuit rendering (matplotlib drawer).
* 🌐 One‑click Streamlit UI for demos.

---

## 🗂 Repository Structure

```
Quantum-Enhanced-ML-Applications/
│
├─ notebooks/                 # Optional exploratory notebooks
├─ results/                   # Lightweight logs & predictions (no large files)
├─ src/
│  ├─ quantum_models.py       # Sampler-based classifier (Qiskit ≥ 1.x)
│  ├─ utils.py                # Data loading & preprocessing (Iris)
│  └─ visualization_utils.py  # Plotting helpers
│
├─ run_experiment.py          # CLI runner: train + evaluate
├─ streamlit_app.py           # 🌐 Streamlit UI
├─ requirements.txt           # Dependencies
├─ LICENSE
└─ README.md
```

---

## ⚙️ Getting Started

### Prerequisites

* Python **3.10–3.12** recommended
* (Windows) Microsoft C++ Build Tools may help for scientific wheels

### 1) Clone & create a virtual environment

```bash
git clone https://github.com/iamvisheshsrivastava/Quantum-Enhanced-ML-Applications.git
cd Quantum-Enhanced-ML-Applications

# Windows (PowerShell)
python -m venv venv
./venv/Scripts/Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you don’t use the requirements file:

```bash
pip install qiskit qiskit-aer qiskit-machine-learning streamlit scikit-learn numpy matplotlib pandas python-dotenv
```

---

## ▶️ Run the Streamlit App (recommended)

```bash
streamlit run streamlit_app.py
```

What you get:

* **Train / Re‑train** button (COBYLA via SciPy, configurable iterations)
* Metrics (accuracy, loss) + **confusion matrix**
* **Inference playground** (feature sliders → prediction)
* **Circuit preview** via Qiskit drawer
* **Download CSV** of predictions

> Tip: The app can "match qubits to features" so Iris (4 features) uses 4 qubits automatically.

---

## 🧪 Run From CLI

```bash
python run_experiment.py
```

This script loads the Iris dataset, trains the classifier, predicts on the test split, and renders a confusion matrix.

---

## 🔐 (Optional) IBM Quantum Access

You can keep everything on simulators, or run on IBM Quantum later.

1. Create an IBM Quantum account and API token.
2. Add a local `.env` (already git‑ignored):

```env
IBM_QUANTUM_TOKEN=your_token_here
```

3. In code, read the token with `python-dotenv` and select an IBM backend (future toggle in the app).

> Never commit your token. `.env` is already ignored by `.gitignore`.

---

## 📦 requirements.txt (reference)

```txt
# Core scientific stack
numpy
scipy
pandas
matplotlib
scikit-learn

# Quantum computing
qiskit
qiskit-aer
qiskit-machine-learning  # optional but handy

# App & utils
streamlit
python-dotenv
```

---

## 🛠️ Troubleshooting (Qiskit ≥ 1.x)

* `ModuleNotFoundError: qiskit.opflow` → OpFlow was removed. Use **Sampler** and standard measurements.
* `QuantumInstance` missing → removed. Use **Sampler/Estimator** primitives instead.
* `bind_parameters` missing → renamed to **`assign_parameters`**.
* `qiskit.algorithms.optimizers` missing → use **SciPy**: `from scipy.optimize import minimize` with `method="COBYLA"`.
* Simulators not found → ensure **`qiskit-aer`** is installed.

---

## 🧭 Roadmap

* IBM hardware toggle in the UI (simulator vs real backend)
* Data re‑uploading feature maps & quantum kernels
* Benchmarks vs classical baselines
* Parameter save/load, experiment tracking
* More datasets & visual dashboards

---

## 🤝 Contributing

1. Fork the repo & create a feature branch: `git checkout -b feat/name`
2. Commit changes: `git commit -m "feat: ..."`
3. Push & open a Pull Request.

Please keep results small; link large assets externally.

---

## 📄 License

MIT — see [`LICENSE`](./LICENSE).

---

## 📫 Contact

* Email: **[srivastava.vishesh9@gmail.com](mailto:srivastava.vishesh9@gmail.com)**
* LinkedIn: **Vishesh Srivastava**

If you use this project in your work, a star ⭐ on GitHub is appreciated!
