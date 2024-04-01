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

N = 16  # Number of spins

import qutip as qt

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
si = qt.qeye(2)

def find_neighbors(site, L=2):
    row, col = divmod(site, L)
    nearest_neighbors = []
    next_nearest_neighbors = []

    # Nearest neighbors
    if row > 0: nearest_neighbors.append((row - 1) * L + col)  # Up
    if row < L - 1: nearest_neighbors.append((row + 1) * L + col)  # Down
    if col > 0: nearest_neighbors.append(row * L + (col - 1))  # Left
    if col < L - 1: nearest_neighbors.append(row * L + (col + 1))  # Right

    # Next-nearest neighbors
    if row > 0 and col > 0: next_nearest_neighbors.append((row - 1) * L + (col - 1))  # Up-Left
    if row > 0 and col < L - 1: next_nearest_neighbors.append((row - 1) * L + (col + 1))  # Up-Right
    if row < L - 1 and col > 0: next_nearest_neighbors.append((row + 1) * L + (col - 1))  # Down-Left
    if row < L - 1 and col < L - 1: next_nearest_neighbors.append((row + 1) * L + (col + 1))  # Down-Right

    return nearest_neighbors, next_nearest_neighbors

# Define the kronecker product (tensor product) function for QuTiP objects
def tensor_product_qutip(ops):
    return qt.tensor(ops)

# Function to create site operators in the many-body Hilbert space
def site_operator(N, op, site):
    op_list = [si] * N
    op_list[site] = op
    return tensor_product_qutip(op_list)

# Hamiltonian construction function for the 2x2 lattice
# Hamiltonian construction function for the 2x2 lattice
def construct_hamiltonian_qutip(L=2, J2=0.5):
    N_sites = L * L  # Total number of sites
    H = 0  # Start with an empty Hamiltonian
    
    for site in range(N_sites):
        # Add nearest-neighbor interactions
        nearest_neighbors, _ = find_neighbors(site, L=L)
        for neighbor in nearest_neighbors:
            if site < neighbor:  # To avoid double counting
                H += (1/4) * (site_operator(N_sites, sx, site) * site_operator(N_sites, sx, neighbor) +
                               site_operator(N_sites, sy, site) * site_operator(N_sites, sy, neighbor) +
                               site_operator(N_sites, sz, site) * site_operator(N_sites, sz, neighbor))
        # Add next-nearest-neighbor interactions
        _, next_nearest_neighbors = find_neighbors(site, L=L)
        for neighbor in next_nearest_neighbors:
            if site < neighbor:  # To avoid double counting
                H += J2 * (site_operator(N_sites, sx, site) * site_operator(N_sites, sx, neighbor) +
                            site_operator(N_sites, sy, site) * site_operator(N_sites, sy, neighbor) +
                            site_operator(N_sites, sz, site) * site_operator(N_sites, sz, neighbor))
    
    return H


# Construct the Hamiltonian for a 2x2 lattice
H_qutip = construct_hamiltonian_qutip(L=4, J2=0.5)

H_qutip

import numpy as np
from qiskit import Aer, execute, QuantumCircuit
from qiskit.opflow import I, X, Y, Z, PauliExpectation, CircuitSampler, StateFn
from qiskit.utils import QuantumInstance
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector
H_np = H_qutip.full()

# Assuming heisenberg_hamiltonian(N, J) is defined elsewhere

# Function to calculate the energy of a state
def calculate_energy(state, hamiltonian_matrix):
    return np.dot(state.T, np.dot(hamiltonian_matrix, state))

# Convert a bitstring to a state vector
def bitstring_to_state_vector(bitstring):
    state = np.zeros(2**N)  # Adjusted for N qubits
    index = int(bitstring, 2)
    state[index] = 1
    return state


def vqe_ansatz(N, parameters):
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.rx(parameters[i], i)
        qc.rz(parameters[i + N], i)
    for i in range(N - 1):
        qc.cz(i, (i + 1) % N)  # Optional: Adjust connectivity as needed for your lattice
    return qc


# VQE Ansatz definition, VQE objective function here

# Objective function for VQE
parameters = ParameterVector('θ', 2 * N)  # Adjusted for N qubits

def vqe_objective(parameters, hamiltonian_matrix, ansatz, backend):
    simulator = Aer.get_backend('statevector_simulator')
    # Bind parameters to the circuit
    parameter_dict = dict(zip(ansatz.parameters, parameters))
    bound_ansatz = ansatz.bind_parameters(parameter_dict)
    
    # Execute the circuit
    job = execute(bound_ansatz, simulator)
    result = job.result()
    statevector = result.get_statevector(bound_ansatz)
    
    # Use MatrixOp for Hamiltonian if it's a NumPy array
    hamiltonian = MatrixOp(hamiltonian_matrix)
    
    # Calculate the expectation value
    op = ~StateFn(hamiltonian) @ StateFn(statevector)
    expectation = PauliExpectation().convert(op)
    value = np.real(expectation.eval())
    
    return value

# Quantum-Enhanced VMC function, including VQE to find the optimal parameters
def qe_vmc(N, hamiltonian_matrix, num_samples=1000):
    # VQE part to find optimal parameters (setup and optimization)
    # Placeholder for optimal_params
    parameters = ParameterVector('θ', 2 * N)
    ansatz = vqe_ansatz(N, parameters)
    backend = QuantumInstance(Aer.get_backend('qasm_simulator'))
    
    # Find optimal parameters
    initial_params = np.random.rand(2 * N)
    result = minimize(vqe_objective, initial_params, args=(hamiltonian_matrix, ansatz, backend), method='COBYLA')
    optimal_params = result.x
    #optimal_params = # Result from VQE optimization
    
    # Prepare state with optimal parameters and measure
    backend = Aer.get_backend('qasm_simulator')
    qc = QuantumCircuit(N)
    # Assuming qc is the circuit with the optimal state
    
    # Sample from the circuit
    qc.measure_all()
    samples = execute(qc, backend=backend, shots=num_samples).result().get_counts()

    # Estimate the ground state energy
    energies = []
    for bitstring, count in samples.items():
        state_vector = bitstring_to_state_vector(bitstring)
        energy = calculate_energy(state_vector, hamiltonian_matrix)
        energies.append(energy * count)

    average_energy = sum(energies) / num_samples
    return average_energy

# Example parameters for the Heisenberg Hamiltonian
J = 1.0  # Coupling constant
hamiltonian_matrix = H_np  # Convert to matrix if not already

# Running QEVMC to estimate ground state energy
ground_state_energy = qe_vmc(N, hamiltonian_matrix)
print("Estimated ground state energy:", ground_state_energy)
