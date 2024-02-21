import torch
import torch.nn as nn
import torch.optim as optim

from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.circuit import QuantumCircuit, Parameter

from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from itertools import combinations
import cirq

from scipy.sparse.linalg import eigsh,eigs
import scipy.linalg as la
import scipy.sparse as sp


import numpy as np
import math
import matplotlib.pyplot as plt


import os
import csv
import pandas as pd
import argparse

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='model1', help='Name of the model to save')
parser.add_argument("--N", type=int, default=8, help="Number of fermions")
parser.add_argument("--qubits", type=int, default=4, help="enter number of qubits")
parser.add_argument("--depth", type=int, default=2, help="depth of quantum circuit")

args = parser.parse_args()
N=args.N
model_name = args.model_name
num_qubits = args.qubits
depth = args.depth

print("Model Name: ", model_name)
print("Number of qubits: ", num_qubits)
print("Depth of quantum circuit: ", depth)

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

seed= 0
mu= 0.01
hamiltonian_matrix = main(N,seed, mu)
hamiltonian_matrix= torch.tensor(hamiltonian_matrix)

print("The Size of the hamiltonian matrix is given by: ", len(hamiltonian_matrix))


print()
print("Initializing Hybrid Quantum Neural network")
print()

# Define the classical encoder neural network
class ClassicalEncoder(nn.Module):
    def __init__(self):
        super(ClassicalEncoder, self).__init__()
        self.fc = nn.Sequential(
            #nn.Linear(4096, 32),
            #nn.ReLU(),
            #nn.Linear(2048, 1024),
            #nn.ReLU(),
            #nn.Linear(1024, 512),
            #nn.ReLU(),
            #nn.Linear(512, 256),
            #nn.ReLU(),
            nn.Linear(len(hamiltonian_matrix), 128).to(torch.complex128),  # First layer with 7 inputs and 14 outputs
            #nn.ReLU(),         # Activation function
            nn.Linear(128, 64).to(torch.complex128), # Second layer with 14 inputs and 28 outputs
            #nn.ReLU(),         # Activation function
            nn.Linear(64, 32).to(torch.complex128), # Third layer with 28 inputs and 56 outputs
            #nn.ReLU(),         # Activation function
            nn.Linear(32,16).to(torch.complex128),
            #nn.ReLU(),
            #nn.Linear(16, 8).to(torch.complex128), # Fourth layer reducing from 56 to 28 outputs
            #nn.ReLU(),         # Activation function
            #nn.Linear(8, 4).to(torch.complex128), # Fifth layer reducing from 28 to 14 outputs
            #nn.ReLU(),
            #nn.Linear(4,1)
        )
    
    def forward(self, x):
        return self.fc(x)

encoder = ClassicalEncoder()
#print("The encoder is: ", encoder)


class QuantumCircuitModule:
    def __init__(self, num_qubits, depth):
        super(QuantumCircuitModule, self).__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.params = torch.nn.Parameter(torch.rand((3 * num_qubits * depth + depth * (num_qubits - 1)), dtype=torch.float32) * 0.01)  # Small random initialization
        
        # Create a list of parameters for the quantum circuit, with correlations for avoiding barren plateaus
        self.theta = [Parameter(f'θ{i}') for i in range(len(self.params))]

    def forward(self, x):
        # Initialize the quantum circuit
        backend=Aer.get_backend('qasm_simulator')
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        param_counter = 0
        
        y= x.detach().numpy()

        norm = np.sqrt(np.sum(np.abs(y)**2))
        if norm == 0:
            raise ValueError("Cannot normalize the zero vector")
        y = y / norm
        #y= math.ceil(y)

        qc.initialize(y,[i for i in range(self.num_qubits)])
        # Apply initial rotations with a careful initialization

        qc
        

        for layer in range(self.depth):
            # Add rotation layers
            for qubit in range(self.num_qubits):
                qc.rz(self.theta[param_counter], qubit)
                param_counter += 1
                qc.rx(self.theta[param_counter], qubit)
                param_counter += 1
                qc.rz(self.theta[param_counter], qubit)
                param_counter += 1
            
            # Add the entangling RXX gates in a pattern to increase entanglement gradually
            for qubit in range(self.num_qubits - 1):
                qc.rxx(self.theta[param_counter], qubit, (qubit + 1) % self.num_qubits)
                param_counter += 1

            # Add barrier to define layers clearly
        qc.barrier()

        # Add measurements
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        #print(qc.draw('text'))

        # Bind the parameters to the values from the input
        param_values = [p.item() for p in self.params]
        qc_bound = qc.bind_parameters({self.theta[i]: param_values[i] for i in range(len(self.params))})

        # Execute the quantum circuit
        transpiled_circuit = transpile(qc_bound, backend)
        job = execute(transpiled_circuit, backend, shots=2048)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)

        expectation_values = []
        for i in range(self.num_qubits):
            # Calculate expectation value for each qubit
            p0 = counts.get('0' * i + '0' + '0' * (3 - i), 0) / 1024
            p1 = counts.get('0' * i + '1' + '0' * (3 - i), 0) / 1024
            expectation_value = p0 - p1
            expectation_values.append(expectation_value)

        # Convert the most common bitstring to numpy array and then to PyTorch tensor
        output_bitstring = max(counts, key=counts.get)
        output_data = np.array([int(bit) for bit in output_bitstring[::-1]])  # Reverse to match qubit ordering
        output_tensor = torch.tensor(output_data, dtype=torch.complex128)
        #print(torch.tensor(expectation_values, dtype=torch.float32))
        #print(output_tensor)

        return output_tensor#torch.tensor(expectation_values, dtype=torch.float32)




# Define the classical decoder neural network
class ClassicalDecoder(nn.Module):
    def __init__(self):
        super(ClassicalDecoder, self).__init__()
        self.fc = nn.Sequential(
            #nn.Linear(4, 8),    # First layer with 4 inputs and 8 outputs
            #nn.ReLU(),          # Activation function
            #nn.Linear(8, 4),   # Second layer with 8 inputs and 16 outputs
            #nn.ReLU(),          # Activation function
            nn.Linear(4, 8).to(torch.complex128),  # Third layer with 16 inputs and 32 outputs
            #nn.ReLU(),          # Activation function
            nn.Linear(8, 16).to(torch.complex128),
            #nn.ReLU(),
            nn.Linear(16, 32).to(torch.complex128),  # Fourth layer reducing from 32 to 16 outputs
            #nn.ReLU(),          # Activation function
            nn.Linear(32, 64).to(torch.complex128),
            #nn.ReLU(),
            nn.Linear(64, 128).to(torch.complex128),
            #nn.ReLU(),
            nn.Linear(128, len(hamiltonian_matrix)).to(torch.complex128),
            #nn.ReLU(),
            #nn.Linear(1024, 2048),
            #nn.ReLU(),
            #nn.Linear(2048, 4096)
            
        )
    
    def forward(self, x):
        return self.fc(x)

decoder = ClassicalDecoder()
#print("The decoder is: ", decoder)


class HybridModel(nn.Module):
    def __init__(self, num_qubits, depth):
        super(HybridModel, self).__init__()
        self.encoder = ClassicalEncoder()
        self.decoder = ClassicalDecoder()
        self.qcircuit_module = QuantumCircuitModule(num_qubits, depth)  # Initialize the quantum circuit module

    def forward(self, x):
        encoded= self.encoder(x)
        quantum_result = self.qcircuit_module.forward(encoded)  # Use the result from the quantum circuit module
        decoded = self.decoder(quantum_result)
        return decoded

# Initialize the model
model = HybridModel(num_qubits, depth)

def find_lowest_eigenvalue(matrix):
    # Compute all eigenvalues, but only the first eigenvectors
    eigenvalues, _ = la.eigh(matrix, eigvals=(0, 0))
    return eigenvalues[0]

lowest_eigenvalue = find_lowest_eigenvalue(hamiltonian_matrix)


def energy_expectation(output, hamiltonian):

    # Convert hamiltonian to double
    hamiltonian = hamiltonian.type(torch.complex128)

    # Convert output to double
    wavefunction = output.type(torch.complex128)

    # Normalize the wavefunction
    norm_wavefunction = wavefunction / torch.sqrt(torch.sum(torch.abs(wavefunction)**2))

    # Calculate the energy expectation value
    energy = torch.vdot(norm_wavefunction, torch.mv(hamiltonian, norm_wavefunction)).real

    #eigenvalues, _ = la.eigh(hamiltonian, eigvals=(0, 0))

    #val = torch.tensor(eigenvalues, requires_grad= True)

    return energy










# Sample input
input_data = torch.rand(len(hamiltonian_matrix), requires_grad=True).type(torch.complex128) # Example input
#input_data= torch.tensor([ 0.3679, -0.0602,  0.6200,  0.1083, -0.0054,  0.0107,  0.1241, 0.3679, -0.0602,  0.6200,  0.1083, -0.0054,  0.0107,  0.1241])

# Optimization setup
#print("The model parameters are: ", model.parameters)
# Add weight decay to the optimizer (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=0.1)
num_epochs = 1000
loss_values = []

loss_function = nn.MSELoss()

outputs=[]

for epoch in range(num_epochs):
    optimizer.zero_grad()            # Clear existing gradients
    output = model(input_data)       # Forward pass

    # Ensure output requires grad
    if not output.requires_grad:
        raise RuntimeError("Output does not require gradients. Check model implementation.")

    # Calculate the loss
    #initial_hamiltonian = hamiltonian_initial_module.mf.get_hcore()
    #final_hamiltonian = hamiltonian_final_module.mf.get_hcore()
    loss = energy_expectation(output,hamiltonian_matrix)
    # Check if loss requires grad
    #if not loss.requires_grad:
     #   raise RuntimeError("Loss does not require gradients. Check energy_expectation implementation.")

    loss.backward()                  # Backward pass
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f'NaN gradient in {name}')
            break
    optimizer.step()                 # Update parameters
    loss_values.append(loss.item())  # Store loss for plotting
    outputs.append(output)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


final_val= float(energy_expectation(outputs[999], hamiltonian_matrix))
print("Lowest Eigenvalue:", lowest_eigenvalue)
print("Final output of the model: ", final_val) 
print("energy difference is: ", abs(abs(lowest_eigenvalue)- abs(final_val)))


def diff_calculator(true,network_energy):
    value = abs(true- network_energy)/network_energy
    return abs(value *100)


print("Energy difference in percentage : ", diff_calculator(lowest_eigenvalue, final_val))



# Plotting the loss values
print("PLOTTING")
plt.plot(loss_values)
plt.axhline(y=lowest_eigenvalue, color='r', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time for N=8')
plt.savefig("SYK_Hamiltonian_HA_Model/SYK_plot")


df = pd.read_csv("SYK_Hamiltonian_HA_Model/SYK_modified_results.csv")


def name_generator(time):
    string= "experiment_number_"
    string= string+ str(time)
    return string



true_energy= lowest_eigenvalue
Experiment_run= loss_values[len(loss_values)-1]
# CSV Logging
log_fields = ['Experiment_run', 'true_energy', 'hybrid_model_energy', "difference", "SYK_N_Value","num_qubits", "depth"]
log_data = [name_generator(len(df)), true_energy, final_val,diff_calculator(lowest_eigenvalue, final_val), N, num_qubits, depth]

# Check if file exists
file_exists = os.path.isfile('SYK_Hamiltonian_HA_Model/SYK_modified_results.csv')

# Write to CSV
with open('SYK_Hamiltonian_HA_Model/SYK_modified_results.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=log_fields)
    if not file_exists:
        writer.writeheader()
    writer.writerow({field: data for field, data in zip(log_fields, log_data)})