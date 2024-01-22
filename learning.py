import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from classical_encoder import ClassicalEncoder
from classical_decoder import ClassicalDecoder
from QuantumCircuit import run_quantum_circuit

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.encoder = ClassicalEncoder()
        self.decoder = ClassicalDecoder()
        self.qcircuit = run_quantum_circuit

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
    #wavefunction_np = wavefunction.detach().numpy()
    #expectation_value = np.dot(wavefunction_np, np.dot(hamiltonian, wavefunction_np))

    # Normalize the wavefunction
    norm_wavefunction = wavefunction / torch.sqrt(torch.sum(torch.abs(wavefunction)**2))

    # Check if the size of the Hamiltonian matches the size of the wavefunction
    # This is crucial, and you need to address this if there's a mismatch
    assert H_complex.shape[0] == norm_wavefunction.shape[0], "Size mismatch between Hamiltonian and wavefunction"

    # Calculate the energy expectation value
    energy = torch.vdot(norm_wavefunction, torch.mv(H_complex, norm_wavefunction)).real


    return energy #torch.tensor([0.0], requires_grad=True)  # Example placeholder

# Sample input
#input_data = torch.rand(7, requires_grad=True)  # Example input
input_data= torch.tensor([ 0.3679, -0.0602,  0.6200,  0.1083, -0.0054,  0.0107,  0.1241])

# Optimization setup
#print("The model parameters are: ", model.parameters)
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
    loss = energy_expectation(output, mf.get_hcore())
    # Check if loss requires grad
    if not loss.requires_grad:
        raise RuntimeError("Loss does not require gradients. Check energy_expectation implementation.")

    loss.backward()                  # Backward pass
    optimizer.step()                 # Update parameters
    loss_values.append(loss.item())  # Store loss for plotting
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
