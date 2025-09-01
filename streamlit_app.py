# streamlit_app.py
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# your modules
from src.utils import load_iris_data
from src.quantum_models import SimpleQuantumClassifier

# --------- Page setup & styles ----------
st.set_page_config(
    page_title="Quantum-Enhanced ML • Demo",
    page_icon="🌌",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* Global */
:root {
  --card-bg: rgba(255,255,255,0.75);
  --glass: backdrop-filter: blur(10px);
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}
/* Hero */
.hero {
  background: linear-gradient(135deg,#0b1020 0%, #142850 60%, #27496d 100%);
  color: #f6fbff;
  border-radius: 22px;
  padding: 28px 28px;
  margin-bottom: 18px;
  box-shadow: 0 20px 40px rgba(0,0,0,.25);
}
.hero h1 {
  margin: 0 0 6px 0;
  font-weight: 800;
}
.hero p {
  opacity: .92;
  margin: 0;
}

/* “Glass” cards */
.card {
  background: var(--card-bg);
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 20px rgba(0,0,0,.08);
}

/* Metric pills */
.metric {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(0,0,0,.06);
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 10px 16px;
  font-weight: 700;
  border: 0;
  background: linear-gradient(135deg,#7bdcff,#4f9cff);
  color: #07152b;
  transition: transform .03s ease-in-out;
}
.stButton>button:active { transform: translateY(1px); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------- Helpers ----------
@st.cache_data
def get_data():
    x_train, x_test, y_train, y_test = load_iris_data()
    return x_train, x_test, y_train, y_test

def compute_confusion_figure(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig

def make_download_csv(preds, y_true):
    buf = io.StringIO()
    buf.write("index,true_label,pred_label\n")
    for i,(t,p) in enumerate(zip(y_true, preds)):
        buf.write(f"{i},{t},{p}\n")
    return buf.getvalue().encode()

# --------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Tune the experiment and train the quantum model.")
    iters = st.slider("Optimizer iterations", 5, 200, 30, step=5)
    match_qubits = st.checkbox("Match #qubits to #features", value=True,
                               help="Recommended: one qubit per feature")
    manual_qubits = st.number_input("If unchecked, choose #qubits",
                                    min_value=1, max_value=8, value=2, step=1, disabled=match_qubits)
    run_button = st.button("🚀 Train / Re-train")

# --------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>🌌 Quantum-Enhanced ML — Interactive Demo</h1>
  <p>Train a tiny hybrid quantum-classical classifier on the Iris dataset, visualize results,
  try live predictions, and preview the circuit — all in your browser.</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------- Data ----------
x_train, x_test, y_train, y_test = get_data()
n_features = x_train.shape[1]
n_classes = len(set(y_train))

# --------- Session state for model/results ----------
if "clf" not in st.session_state:
    st.session_state.clf = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "preds" not in st.session_state:
    st.session_state.preds = None
if "loss" not in st.session_state:
    st.session_state.loss = None

# --------- Top metrics row ----------
mcol1, mcol2, mcol3, mcol4 = st.columns([1.2,1,1,1])
with mcol1:
    st.markdown('<div class="card"><div class="metric">📚 Dataset: <b>Iris</b></div></div>', unsafe_allow_html=True)
with mcol2:
    st.markdown(f'<div class="card"><div class="metric">🔢 Features: <b>{n_features}</b></div></div>', unsafe_allow_html=True)
with mcol3:
    st.markdown(f'<div class="card"><div class="metric">🏷️ Classes: <b>{n_classes}</b></div></div>', unsafe_allow_html=True)
with mcol4:
    current_qubits = n_features if match_qubits else int(manual_qubits)
    st.markdown(f'<div class="card"><div class="metric">⚛️ Qubits: <b>{current_qubits}</b></div></div>', unsafe_allow_html=True)

st.markdown("")

# --------- Train model ----------
if run_button:
    with st.status("Training quantum classifier…", expanded=True) as status:
        st.write("• Initializing model")
        clf = SimpleQuantumClassifier(num_features=n_features, num_qubits=current_qubits)

        st.write(f"• Optimizing (COBYLA) for **{iters}** iterations")
        loss = clf.fit(x_train, y_train, maxiter=int(iters))

        st.write("• Predicting on test split")
        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)

        st.session_state.clf = clf
        st.session_state.preds = preds
        st.session_state.loss = float(loss)
        st.session_state.trained = True

        st.write(f"• Done. Accuracy: **{acc:.3f}**, Loss: **{loss:.4f}**")
        status.update(label="✅ Training complete", state="complete")

# --------- Results & Visuals ----------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("📈 Results")
    if st.session_state.trained:
        preds = st.session_state.preds
        loss = st.session_state.loss
        acc = accuracy_score(y_test, preds)

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Final Loss", f"{loss:.4f}")

        fig_cm = compute_confusion_figure(
            y_true=y_test,
            y_pred=preds,
            class_names=["Setosa","Versicolor","Virginica"],
            title="Confusion Matrix (Test)"
        )
        st.pyplot(fig_cm, use_container_width=True)

        # Download predictions
        st.download_button(
            "⬇️ Download predictions CSV",
            data=make_download_csv(preds, y_test),
            file_name="iris_quantum_predictions.csv",
            mime="text/csv",
            help="Index, true label, predicted label"
        )
    else:
        st.info("Click **Train / Re-train** in the sidebar to run the experiment.")

with right:
    st.subheader("🧪 Inference Playground")
    st.caption("Tweak features and get an instant prediction from the trained model.")

    if st.session_state.trained and st.session_state.clf is not None:
        # Build sliders from data ranges
        mins = np.min(np.vstack([x_train, x_test]), axis=0)
        maxs = np.max(np.vstack([x_train, x_test]), axis=0)

        # Create sliders for each feature
        user_x = []
        for i in range(n_features):
            user_x.append(st.slider(f"Feature x{i}", float(mins[i]), float(maxs[i]),
                                    float(np.mean([mins[i], maxs[i]])), step=float((maxs[i]-mins[i])/100 or 0.01)))
        user_x = np.array(user_x, dtype=float).reshape(1, -1)

        if st.button("🔮 Predict sample"):
            pred = st.session_state.clf.predict(user_x)[0]
            label = ["Setosa","Versicolor","Virginica"][int(pred)] if pred in [0,1,2] else str(pred)
            st.success(f"Predicted class: **{label}**")
    else:
        st.info("Train the model first to enable live predictions.")

# --------- Circuit preview ----------
st.subheader("🧩 Circuit Preview")
if st.session_state.trained and st.session_state.clf is not None:
    try:
        # Draw the circuit with matplotlib
        from qiskit.visualization import circuit_drawer
        fig_circ = circuit_drawer(st.session_state.clf.qc, output="mpl")
        st.pyplot(fig_circ, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render circuit: {e}")
else:
    st.caption("Train the model to preview the current parameterized circuit.")

# --------- Footer ----------
with st.expander("About this demo"):
    st.markdown(
        """
**What is this?** An interactive demo of a tiny hybrid quantum-classical model using Qiskit’s
modern *Sampler* API. It encodes features as rotations, optimizes trainable gates (COBYLA via SciPy),
and predicts by measuring Z-expectation.

**Next ideas:**
- IBM Quantum backend toggle (real device vs simulator)  
- More datasets & feature maps (data re-uploading, ZZFeatureMap, kernels)  
- Benchmarks vs classical baselines  
- Save & load trained parameters  
"""
    )
