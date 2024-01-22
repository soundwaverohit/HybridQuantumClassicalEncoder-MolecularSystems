import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import types
import sys

# Importing the contents of the files as modules
# This involves dynamically creating modules from the file contents

# Helper function to create a module from file content
def create_module(file_name, content):
    module = types.ModuleType(file_name)
    exec(content, module.__dict__)
    sys.modules[file_name] = module
    return module

# Define file paths
file_paths = [
    "classical_encoder.py",
    "classical_decoder.py",
    "QuantumCircuit.py",
]

# Read file contents
file_contents = {}
for file_path in file_paths:
    with open(file_path, 'r') as file:
        file_contents[file_path] = file.read()

# Creating modules from the provided files
classical_encoder_module = create_module("classical_encoder", file_contents["classical_encoder.py"])
classical_decoder_module = create_module("classical_decoder", file_contents["classical_decoder.py"])
quantum_circuit_module = create_module("QuantumCircuit", file_contents["QuantumCircuit.py"])

# Define the hybrid model
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.encoder = classical_encoder_module.ClassicalEncoder()
        self.decoder = classical_decoder_module.ClassicalDecoder()
        self.qcircuit = quantum_circuit_module.run_quantum_circuit

    def forward(self, x):
        encoded = self.encoder(x)
        quantum_result = self.qcircuit(encoded)
        decoded = self.decoder(quantum_result)
        return decoded

# Initialize the model
model = HybridModel()

from pyscf import gto, scf

# Define the loss function (energy expectation)
def energy_expectation(output, hamiltonian):
    # Your implementation here using PyTorch operations
    # Define the molecule

    # Convert Hamiltonian matrix to complex tensor
    H_complex = torch.tensor(hamiltonian, dtype=torch.cfloat).real

    wavefunction = output  # Assuming this is a complex tensor
    wavefunction_np = wavefunction.detach().numpy()
    expectation_value = np.dot(wavefunction_np, np.dot(hamiltonian, wavefunction_np))

    # Normalize the wavefunction
    norm_wavefunction = wavefunction / torch.sqrt(torch.sum(torch.abs(wavefunction)**2))

    # Check if the size of the Hamiltonian matches the size of the wavefunction
    # This is crucial, and you need to address this if there's a mismatch
    assert H_complex.shape[0] == norm_wavefunction.shape[0], "Size mismatch between Hamiltonian and wavefunction"

    # Calculate the energy expectation value
    energy = torch.vdot(norm_wavefunction, torch.mv(H_complex, norm_wavefunction)).real


    return energy #torch.tensor([0.0], requires_grad=True)  # Example placeholder

# Sample input
input_data = torch.rand(7, requires_grad=True)  # Example input

# Optimization setup
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 200
loss_values = []
mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')
# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

for epoch in range(num_epochs):
    optimizer.zero_grad()            # Clear existing gradients
    output = model(input_data)       # Forward pass

    # Ensure output requires grad
    if not output.requires_grad:
        raise RuntimeError("Output does not require gradients. Check model implementation.")

    # Calculate the loss
    #initial_hamiltonian = hamiltonian_initial_module.mf.get_hcore()
    #final_hamiltonian = hamiltonian_final_module.mf.get_hcore()
    loss = abs(abs(energy_expectation(output, mf.get_hcore())) - abs(-74.9630631297277)) / abs(-74.9630631297277)

    # Check if loss requires grad
    if not loss.requires_grad:
        raise RuntimeError("Loss does not require gradients. Check energy_expectation implementation.")

    loss.backward()                  # Backward pass
    optimizer.step()                 # Update parameters
    loss_values.append(loss.item())  # Store loss for plotting
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
