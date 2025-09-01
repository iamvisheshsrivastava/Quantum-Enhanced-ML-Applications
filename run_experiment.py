"""
run_experiment.py

End-to-end pipeline for testing SimpleQuantumClassifier on the Iris dataset.
"""

from src.quantum_models import SimpleQuantumClassifier
from src.utils import load_iris_data
from src.visualization_utils import plot_confusion_matrix

print("🔹 Loading data...")
x_train, x_test, y_train, y_test = load_iris_data()

print("🔹 Initializing classifier...")
clf = SimpleQuantumClassifier(num_features=4, num_qubits=4)

print("🔹 Training classifier (this may take a minute)...")
loss = clf.fit(x_train, y_train, maxiter=30)  # reduce maxiter for quick tests
print(f"✅ Final training loss: {loss:.4f}")

print("🔹 Making predictions...")
predictions = clf.predict(x_test)

print("🔹 Plotting confusion matrix...")
plot_confusion_matrix(y_test, predictions, class_names=["Setosa", "Versicolor", "Virginica"])

print("✅ Experiment complete.")
