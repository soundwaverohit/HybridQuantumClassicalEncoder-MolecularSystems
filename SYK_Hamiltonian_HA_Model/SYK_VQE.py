import argparse
from itertools import combinations

import numpy as np
from scipy.linalg import eigh

import cirq
from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

import torch 

import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, ADAM
from qiskit.opflow import MatrixOp
import warnings
from qiskit.circuit.library import RZGate, RXGate, RXXGate
from math import pi

warnings.filterwarnings('ignore')

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n-1)

def get_couplings(N, var, L_inds, R_inds, seed, q):
    """Returns dictionaries of hamiltonian terms and their coefficients"""
    np.random.seed(seed)
    couplings = np.random.normal(scale=np.sqrt(var), size=len(L_inds))
    phase = (-1)**(q/2)
    J_L = {i: c for i, c in zip(L_inds, couplings)}
    J_R = {i: phase * c for i, c in zip(R_inds, couplings)}
    return J_L, J_R

def convert_H_majorana_to_qubit(inds, J_dict, N):
    """Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms"""
    ham_terms = [MajoranaOperator(ind, J_dict[ind]) for ind in inds]
    ham_sum = sum_ops(ham_terms)
    return jordan_wigner(ham_sum)

def q_helper(idx):
    """Returns qubit object based on index"""
    return cirq.LineQubit(idx)

def construct_pauli_string(ham, key):
    """Converts Pauli terms in the Hamiltonian to a string representation"""
    gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

    def list_of_terms(key):
        return [gate_dict[label](q_helper(idx)) for (idx, label) in key]

    return cirq.PauliString(ham.terms[key], list_of_terms(key))

def sum_ops(operators):
    """Wrapper for summing a list of majorana operators"""
    return sum(operators, MajoranaOperator((), 0))

def gs_energy(hamiltonian):
    """Use scipy to get the ground state energy"""
    from scipy.linalg import eigvalsh
    return eigvalsh(hamiltonian, eigvals=(0,0))

def main(N, seed, mu):
    q = 4 # setting q = N is all to all connectivity
    J = 1 # overall coupling strength

    J_var = 2**(q-1) * J**2 * factorial(q-1) / (q * N**(q-1))

    L_indices = range(0, N)
    R_indices = range(N, 2 * N)
    SYK_L_indices = list(combinations(L_indices, q))
    SYK_R_indices = list(combinations(R_indices, q))
    interaction_indices = [(l, r) for l, r in zip(L_indices, R_indices)]

    J_L, J_R = get_couplings(N, J_var, SYK_L_indices, SYK_R_indices, seed, q)
    interaction_strength = {ind: 1j * mu for ind in interaction_indices}

    H_L = convert_H_majorana_to_qubit(SYK_L_indices, J_L, N)
    H_R = convert_H_majorana_to_qubit(SYK_R_indices, J_R, N)
    H_int = convert_H_majorana_to_qubit(interaction_indices, interaction_strength, N)

    total_ham = H_L + H_R + H_int

    matrix_ham = get_sparse_operator(total_ham)
    return matrix_ham.todense()

N=8
seed= 0
mu= 0.01
hamiltonian_matrix = main(N,seed, mu)
#hamiltonian_matrix= torch.tensor(hamiltonian_matrix)

print(len(hamiltonian_matrix))



# Convert the dense Hamiltonian matrix to a suitable operator for VQE
hamiltonian_operator = MatrixOp(hamiltonian_matrix)


# Define a simple parameterized circuit as the ansatz
def create_ansatz(num_qubits):
    num_params_per_layer = 3 * num_qubits + num_qubits // 2
    total_params = 4 * num_params_per_layer
    params = [Parameter(f'θ{i}') for i in range(total_params)]#ParameterVector('θ', length=num_qubits)
    qc = QuantumCircuit(num_qubits)

    d = 4  # replace with desired depth
    n = 8

    param_idx = 0

    for layer in range(d):
        for qubit in range(num_qubits):
            # Apply RZ, RX, RZ with unique parameters
            qc.rz(params[param_idx], qubit)
            param_idx += 1
            qc.rx(params[param_idx], qubit)
            param_idx += 1
            qc.rz(params[param_idx], qubit)
            param_idx += 1
        
        # Apply RXX gates with unique parameters
        for qubit in range(0, num_qubits, 2):
            qc.rxx(params[param_idx], qubit, qubit + 1)
            param_idx += 1

    return qc, params


"""

def create_ansatz(num_qubits):
    depth=4
    # Each layer has 3 rotations per qubit and half the number of qubits RXX gates
    num_params_per_rotation_layer = 3 * num_qubits
    num_params_per_rxx_layer = num_qubits // 2
    num_params_per_layer = num_params_per_rotation_layer + num_params_per_rxx_layer
    total_params = depth * num_params_per_layer

    # Create a list of parameters
    params = [Parameter(f'θ{i}') for i in range(total_params)]
    qc = QuantumCircuit(num_qubits)

    # Assign parameters layer by layer
    for layer in range(depth):
        # Assign parameters for rotation gates
        for qubit in range(num_qubits):
            param_idx = layer * num_params_per_layer + qubit * 3
            qc.rz(params[param_idx], qubit)
            qc.rx(params[param_idx + 1], qubit)
            qc.rz(params[param_idx + 2], qubit)

        # Assign parameters for RXX gates in the layer
        for qubit in range(0, num_qubits, 2):
            param_idx = layer * num_params_per_layer + num_params_per_rotation_layer + qubit // 2
            qc.rxx(params[param_idx], qubit, qubit + 1)

    return qc, params
"""

num_qubits = int(np.log2(hamiltonian_matrix.shape[0]))
ansatz, parameters = create_ansatz(num_qubits)

# Use SPSA optimizer, it's suitable for noisy optimization like on a real quantum device
optimizer = SPSA(maxiter=1000)

# Setup quantum instance to use the statevector simulator
quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator_statevector'))

# Initialize VQE with the ansatz, optimizer, and the quantum instance
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance)

# Run VQE to find the lowest eigenvalue of the Hamiltonian
vqe_result = vqe.compute_minimum_eigenvalue(operator=hamiltonian_operator)

# Extract the lowest eigenvalue
lowest_eigenvalue = np.real(vqe_result.eigenvalue)
print("The VQE given lowest eigenvalue is: ", lowest_eigenvalue)

# Compare to exact solver
exact_solver = NumPyMinimumEigensolver()
exact_result = exact_solver.compute_minimum_eigenvalue(operator=hamiltonian_operator)

print('Exact Solver Result:', exact_result.eigenvalue.real)
