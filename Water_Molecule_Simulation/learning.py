import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Water_Molecule_Simulation.classical_encoder import ClassicalEncoder
from Water_Molecule_Simulation.classical_decoder import ClassicalDecoder
from Water_Molecule_Simulation.QuantumCircuit import run_quantum_circuit, run_quantum_circuit_and_calculate_expectation_values
from Water_Molecule_Simulation.hamiltonian_matrix import large_H_spin_sparse_csr, smallest_eigenvalue
#from pyscf import gto, scf
import pandas as pd 
import os 
import csv
import argparse


parser = argparse.ArgumentParser(description='Train a QuantumModel with custom parameters.')
parser.add_argument('--model_name', type=str, default='model1', help='Name of the model to save')
args = parser.parse_args()


model_name = args.model_name
print("Model Name: ", model_name)


class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.encoder = ClassicalEncoder()
        self.decoder = ClassicalDecoder()
        self.qcircuit = run_quantum_circuit #run_quantum_circuit_and_calculate_expectation_values

    def forward(self, x):
        encoded = self.encoder(x)
        quantum_result = self.qcircuit(encoded)
        decoded = self.decoder(quantum_result)
        return decoded

# Initialize the model
model = HybridModel()


# Define the loss function (energy expectation)
def energy_expectation(output, hamiltonian):
    # Your implementation here using PyTorch operations
    # Define the molecule

    # Convert Hamiltonian matrix to complex tensor

    # Extract the row indices, column indices, and values from the CSR matrix
    rows, cols = hamiltonian.nonzero()
    values = hamiltonian.data

    # Convert to torch tensors
    rows = torch.tensor(rows, dtype=torch.int64)
    cols = torch.tensor(cols, dtype=torch.int64)
    values = torch.tensor(values, dtype=torch.cfloat)

    # Create indices tensor for the sparse tensor
    indices = torch.vstack((rows, cols))

    # Create the PyTorch sparse tensor
    H_complex = torch.sparse_coo_tensor(indices, values, size=large_H_spin_sparse_csr.shape)

    # Convert to dense tensor and then take the real part
    H_complex_dense = H_complex.to_dense()
    H_complex_real_dense = H_complex_dense.real

    # Assuming values is the tensor of complex values
    real_values = values.real  # Extract the real part before creating the sparse tensor

    # Create the sparse tensor with real values
    H_complex_real = torch.sparse_coo_tensor(indices, real_values, size=large_H_spin_sparse_csr.shape)



    # Convert to real part only if needed
    #H_complex = H_complex.real()
    #H_complex = torch.tensor(hamiltonian, dtype=torch.cfloat).real

    wavefunction = output  # Assuming this is a complex tensor
    #wavefunction_np = wavefunction.detach().numpy()
    #expectation_value = np.dot(wavefunction_np, np.dot(hamiltonian, wavefunction_np))

    # Normalize the wavefunction
    norm_wavefunction = wavefunction / torch.sqrt(torch.sum(torch.abs(wavefunction)**2))

    # Check if the size of the Hamiltonian matches the size of the wavefunction
    # This is crucial, and you need to address this if there's a mismatch
    assert H_complex.shape[0] == norm_wavefunction.shape[0], "Size mismatch between Hamiltonian and wavefunction"

    # Calculate the energy expectation value
    energy = torch.vdot(norm_wavefunction, torch.mv(H_complex_real, norm_wavefunction)).real


    return energy #torch.tensor([0.0], requires_grad=True)  # Example placeholder


# Sample input
input_data = torch.rand(16380, requires_grad=True)  # Example input
#input_data= torch.tensor([ 0.3679, -0.0602,  0.6200,  0.1083, -0.0054,  0.0107,  0.1241, 0.3679, -0.0602,  0.6200,  0.1083, -0.0054,  0.0107,  0.1241])

# Optimization setup
#print("The model parameters are: ", model.parameters)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
loss_values = []

for epoch in range(num_epochs):
    optimizer.zero_grad()            # Clear existing gradients
    output = model(input_data)       # Forward pass

    # Ensure output requires grad
    if not output.requires_grad:
        raise RuntimeError("Output does not require gradients. Check model implementation.")

    # Calculate the loss
    #initial_hamiltonian = hamiltonian_initial_module.mf.get_hcore()
    #final_hamiltonian = hamiltonian_final_module.mf.get_hcore()
    loss = energy_expectation(output,large_H_spin_sparse_csr)
    # Check if loss requires grad
    if not loss.requires_grad:
        raise RuntimeError("Loss does not require gradients. Check energy_expectation implementation.")

    loss.backward()                  # Backward pass
    optimizer.step()                 # Update parameters
    loss_values.append(loss.item())  # Store loss for plotting
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


def diff_calculator(true,network_energy):
    value = abs(true- network_energy)
    return value


print("Energy difference in percentage : ", diff_calculator(smallest_eigenvalue, loss_values[99]))




df = pd.read_csv("experiment_results.csv")


def name_generator(time):
    string= "experiment_number_"
    string= string+ str(time)
    return string



true_energy= smallest_eigenvalue
Experiment_run= loss_values[99]
# CSV Logging
log_fields = ['Experiment_run', 'true_energy', 'hybrid_model_energy', "difference"]
log_data = [name_generator(len(df)), true_energy, loss_values[99],diff_calculator(smallest_eigenvalue, loss_values[99])]

# Check if file exists
file_exists = os.path.isfile('experiment_results.csv')

# Write to CSV
with open('experiment_results.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=log_fields)
    if not file_exists:
        writer.writeheader()
    writer.writerow({field: data for field, data in zip(log_fields, log_data)})