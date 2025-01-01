"""
quantum_models.py

This file demonstrates a minimal quantum-classical hybrid model using Qiskit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit import Aer, transpile
from qiskit.utils import QuantumInstance
from qiskit.opflow import StateFn, CircuitStateFn, ListOp, PauliSumOp
from qiskit.opflow import Gradient, ExpectationFactory, OperatorBase
from qiskit.algorithms.optimizers import COBYLA

class SimpleQuantumClassifier:
    """
    A minimalistic quantum classifier using parameterized gates and Qiskit's
    opflow for gradient-based optimization (or in this case, COBYLA as an example).
    """

    def __init__(self, num_features=4, num_qubits=2):
        """
        Args:
            num_features (int): Dimensionality of the input features.
            num_qubits (int): Number of qubits in the quantum circuit.
        """
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.params = None
        self._build_circuit()

    def _build_circuit(self):
        """
        Creates a parameterized quantum circuit for classification.
        """
        # Number of parameters to encode data + trainable parameters
        # For a basic circuit, let's have num_features + (some trainable rotations)
        self.data_params = [Parameter(f"x_{i}") for i in range(self.num_features)]
        self.theta_params = [Parameter(f"theta_{i}") for i in range(self.num_qubits)]

        self.qc = QuantumCircuit(self.num_qubits)

        # Encode data into qubit rotations
        for i in range(self.num_qubits):
            self.qc.ry(self.data_params[i % self.num_features], i)

        # Trainable parameters as additional rotations
        for i in range(self.num_qubits):
            self.qc.rx(self.theta_params[i], i)

        # Measurement is implicit in the Expectation formalism (opflow)

    def _construct_op(self, x, y):
        """
        Constructs the operator for the loss function.
        
        We want to measure something that correlates with the predicted label.
        For simplicity, let's use a simple 'Z' measurement on the first qubit 
        to separate classes.

        Args:
            x (array): feature vector of shape (num_features,)
            y (int): label (assuming 0 or 1 for a binary classification in this minimal example)
                     For multi-class, you'd need a different approach.

        Returns:
            OperatorBase: The operator representing the loss for sample x with label y.
        """
        # Bind feature values into the circuit parameters
        param_dict = {self.data_params[i]: float(x[i]) for i in range(self.num_features)}
        
        # We'll keep trainable parameters as symbolic for gradient-based updates.
        # We do NOT bind them here yet, so they remain as Parameter objects.

        # Create a circuit with data parameters bound
        qc_bound = self.qc.bind_parameters(param_dict)

        # We measure Z on the first qubit
        observable = PauliSumOp.from_list([("Z" + "I"*(self.num_qubits-1), 1.0)])
        
        # If y == 0, we want measurement close to +1
        # If y == 1, we want measurement close to -1
        # So let's define a "target" = +1 for y=0, -1 for y=1
        target = 1.0 if y == 0 else -1.0
        # For multi-class or actual data with more labels, you'd expand this logic.

        # Expectation of Z is ~ +1 if state is aligned with |0>, -1 if aligned with |1>
        # We'll define a "loss" = (Expectation(Z) - target)^2
        # We'll do that as (Z - target I)^2 expanded = Z^2 - 2*Z*target + target^2
        # but let's build it in a straightforward way:

        op = (CircuitStateFn(qc_bound, is_measurement=False) ^
              StateFn(observable, is_measurement=True))

        # We'll store the target in this operator as a classical offset later
        # For simplicity, let's just measure the raw expectation first,
        # and handle the (pred - target)^2 in the cost function evaluation.

        return op, target

    def loss_function(self, params, x_data, y_data):
        """
        Computes the mean squared error between measured expectation and target.
        """
        # Bind the trainable circuit parameters
        param_dict = {self.theta_params[i]: float(params[i]) for i in range(self.num_qubits)}

        # Evaluate each data sample and accumulate
        total_loss = 0.0

        for x, y in zip(x_data, y_data):
            op, target = self._construct_op(x, y)
            
            # Convert op to a parameterized expression
            # Then bind the trainable parameters
            bound_op = op.bind_parameters(param_dict)

            # Evaluate using statevector or shot-based simulation
            sampler = self.quantum_instance
            value = sampler.execute(bound_op).real

            # MSE cost: (value - target)^2
            total_loss += (value - target)**2

        return total_loss / len(x_data)

    def fit(self, x_train, y_train, maxiter=100):
        """
        Train the quantum classifier using COBYLA for simplicity.
        """
        # For demonstration, let's do a simple binary classification
        # Convert any label > 0 to 1, label = 0 remains 0
        # (Iris dataset has 3 classes, so you'll need a more advanced approach
        # or subset the data for a binary scenario.)
        y_train_binary = [0 if y == 0 else 1 for y in y_train]

        # Setup Qiskit quantum instance (using statevector simulator for simplicity)
        backend = Aer.get_backend('statevector_simulator')
        self.quantum_instance = QuantumInstance(backend=backend)

        # Initialize parameter guess
        init_params = np.random.randn(self.num_qubits)

        def objective(params):
            return self.loss_function(params, x_train, y_train_binary)

        optimizer = COBYLA(maxiter=maxiter)
        result = optimizer.minimize(fun=objective, x0=init_params)
        self.params = result.x

        # Return final loss, for logging
        final_loss = objective(self.params)
        return final_loss

    def predict(self, x_data):
        """
        Predict labels for given x_data, using the optimized parameters.
        We do a simple threshold on the expectation of Z.
        """
        param_dict = {self.theta_params[i]: float(self.params[i]) for i in range(self.num_qubits)}
        predictions = []

        for x in x_data:
            op, _ = self._construct_op(x, 0)  # label doesn't matter for measurement
            bound_op = op.bind_parameters(param_dict)
            sampler = self.quantum_instance
            value = sampler.execute(bound_op).real  # Expectation of Z

            # If value > 0 => predicted label = 0, else 1
            pred_label = 0 if value >= 0 else 1
            predictions.append(pred_label)

        return np.array(predictions)
