from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from qiskit.primitives import Sampler
import numpy as np
from scipy.optimize import minimize


class SimpleQuantumClassifier:
    def __init__(self, num_features=4, num_qubits=2):
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.params = None
        self._build_circuit()
        self.sampler = Sampler()

    def _build_circuit(self):
        """Parameterized quantum circuit."""
        self.data_params = [Parameter(f"x_{i}") for i in range(self.num_features)]
        self.theta_params = [Parameter(f"theta_{i}") for i in range(self.num_qubits)]

        self.qc = QuantumCircuit(self.num_qubits)
        # Encode features
        for i in range(self.num_qubits):
            self.qc.ry(self.data_params[i % self.num_features], i)
        # Trainable params
        for i in range(self.num_qubits):
            self.qc.rx(self.theta_params[i], i)
        # Add measurement
        self.qc.measure_all()

    def _evaluate(self, x, params):
        """Run circuit with given data + params and return expectation of Z on first qubit."""
        bind_dict = {self.data_params[i]: float(x[i]) for i in range(self.num_features)}
        bind_dict.update({self.theta_params[i]: float(params[i]) for i in range(self.num_qubits)})

        qc_bound = self.qc.assign_parameters(bind_dict)
        job = self.sampler.run(qc_bound)
        result = job.result()
        counts = result.quasi_dists[0]

        p0, p1 = 0.0, 0.0
        for bit, prob in counts.items():
            if isinstance(bit, str):  # old style "01"
                if bit[-1] == "0":
                    p0 += prob
                else:
                    p1 += prob
            elif isinstance(bit, int):  # new style integer
                if bit % 2 == 0:  # even integers -> last qubit = 0
                    p0 += prob
                else:
                    p1 += prob

        return p0 - p1

    def loss_function(self, params, x_data, y_data):
        """Mean squared error between expectation and target."""
        total_loss = 0.0
        for x, y in zip(x_data, y_data):
            target = 1.0 if y == 0 else -1.0
            value = self._evaluate(x, params)
            total_loss += (value - target) ** 2
        return total_loss / len(x_data)

    def fit(self, x_train, y_train, maxiter=50):
        y_train_binary = [0 if y == 0 else 1 for y in y_train]
        init_params = np.random.randn(self.num_qubits)

        def objective(params):
            return self.loss_function(params, x_train, y_train_binary)

        # Use SciPy minimize with COBYLA method
        result = minimize(objective, init_params, method="COBYLA", options={"maxiter": maxiter})
        self.params = result.x
        return objective(self.params)

    def predict(self, x_data):
        predictions = []
        for x in x_data:
            value = self._evaluate(x, self.params)
            predictions.append(0 if value >= 0 else 1)
        return np.array(predictions)
