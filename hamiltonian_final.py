from executor import decoded_output
from qiskit_nature.second_q.drivers import PySCFDriver
import torch
from pyscf import gto, scf

# Define the molecule
mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')
# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Get the Hamiltonian matrix (core Hamiltonian in the AO basis)
H_core = mf.get_hcore()
print('Hamiltonian matrix')
print(H_core)

# Convert Hamiltonian matrix to complex tensor
H_complex = torch.tensor(H_core, dtype=torch.cfloat).real

wavefunction = decoded_output  # Assuming this is a complex tensor

# Normalize the wavefunction
norm_wavefunction = wavefunction / torch.sqrt(torch.sum(torch.abs(wavefunction)**2))

# Check if the size of the Hamiltonian matches the size of the wavefunction
# This is crucial, and you need to address this if there's a mismatch
assert H_complex.shape[0] == norm_wavefunction.shape[0], "Size mismatch between Hamiltonian and wavefunction"

# Calculate the energy expectation value
energy = torch.vdot(norm_wavefunction, torch.mv(H_complex, norm_wavefunction)).real
print("Energy ", energy)


print("True ground state of water: ", )

# Define the molecule
mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')

# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Print the ground state energy
print("Ground state energy (Hartree-Fock approximation):", mf.e_tot, "Hartree")