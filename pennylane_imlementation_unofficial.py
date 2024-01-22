import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

import numpy as np
import matplotlib.pyplot as plt

# ===================================================================================
#                       Create Data
# ===================================================================================

num_train = 7
num_test = 20
len_vector = 7

np.random.seed(1)
x_train = torch.tensor(np.random.rand(num_train, len_vector), dtype=torch.float32)
x_test = torch.tensor(np.random.rand(num_test, len_vector), dtype=torch.float32)

# ===================================================================================
#                       Classical Layers using PyTorch Sequential
# ===================================================================================

# Define first layer, hidden layers, and output layer with the output of two neurons

model = nn.Sequential(
            nn.Linear(7, 14),  # First layer with 7 inputs and 14 outputs
            nn.ReLU(),         # Activation function       # Activation function
            nn.Linear(14, 7)           
        )

# ===================================================================================
#                                Quantum Layer
# ===================================================================================

num_modes = 7
num_basis = 7

dev = qml.device("strawberryfields.fock", wires=num_modes, cutoff_dim=num_basis)

class QuantumLayer(nn.Module):
    def __init__(self, num_layers, num_modes):
        super().__init__()
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.weights = nn.Parameter(torch.randn(num_layers, 5 * num_modes))

    def forward(self, x):
        def q_layer(v):
            qml.Rotation(v[0], wires=0)
            qml.Squeezing(v[1], 0.0, wires=0)
            qml.Rotation(v[2], wires=0)
            qml.Displacement(v[3], 0.0, wires=0)
            qml.Kerr(v[4], wires=0)
        
        @qml.qnode(dev, interface='torch')
        def quantum_nn(inputs, weights):
            # Assuming inputs is a tensor with shape [batch_size, 2]
            for i in range(inputs.shape[0]):
                qml.Displacement(inputs[i, 0].item(), inputs[i, 1].item(), wires=0)
            for v in weights:
                q_layer(v)
            return qml.probs(wires=0)
        
        # Clone and detach to ensure compatibility with PyTorch's computational graph
        x = x.clone().detach().requires_grad_(True)
        return quantum_nn(x, self.weights).float()
    

# ===================================================================================
#                             Hybrid Model
# ===================================================================================

num_layers = 25
num_modes = 1

qlayer = QuantumLayer(num_layers, num_modes)
#model.add_module("QuantumLayer", qlayer)


# ===================================================================================
#                                    Training
# ===================================================================================

opt = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# Initialize lists to store the history
loss_history = []
accuracy_history = []

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

mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')
# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()



# Define a function to calculate accuracy (replace this with your accuracy calculation method)
def calculate_accuracy(y_pred, y_true):
    # Dummy accuracy calculation: replace with your actual accuracy calculation
    # For example, you could define accuracy as the percentage of predictions within a certain range of the actual values
    return torch.mean((y_pred - y_true).abs() / y_true.abs()).item()

for epoch in range(10):
    print(epoch)
    opt.zero_grad()
    print("optimzation zero gradient is created")
    output = model(x_train)
    print("output and shape is generated")
    print(output.shape)
    loss = abs(abs(energy_expectation(output, mf.get_hcore())) - abs(-74.9630631297277)) / abs(-74.9630631297277)
    loss.backward()
    opt.step()
    accuracy = calculate_accuracy(output, x_train)
    
    # Append loss and accuracy to their respective lists
    loss_history.append(loss.item())
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}")

