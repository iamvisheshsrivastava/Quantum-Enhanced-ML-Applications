"""
backend_utils.py

Utility functions to manage Qiskit backends (local simulators and IBM Quantum).
"""

import os
from dotenv import load_dotenv
from qiskit import Aer, IBMQ

def get_backend(use_ibm=False, backend_name="qasm_simulator"):
    """
    Returns a Qiskit backend for running quantum circuits.

    Args:
        use_ibm (bool): If True, load backend from IBM Quantum.
        backend_name (str): Backend name (e.g., 'qasm_simulator', 'ibmq_qasm_simulator', 'ibmq_quito').

    Returns:
        backend (Backend): Qiskit backend instance.
    """
    if not use_ibm:
        return Aer.get_backend(backend_name)

    # Load IBM Quantum credentials
    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")

    if not token:
        raise ValueError("IBM_QUANTUM_TOKEN not found in .env file")

    try:
        IBMQ.load_account()
    except Exception:
        IBMQ.save_account(token, overwrite=True)
        IBMQ.load_account()

    provider = IBMQ.get_provider(hub="ibm-q")
    backend = provider.get_backend(backend_name)
    return backend
