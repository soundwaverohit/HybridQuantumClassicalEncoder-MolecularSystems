from pyscf import gto, scf
import torch
import numpy as np

# Define the molecule
# Example: Hydrogen molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.735', basis='sto-3g')

# Perform Hartree-Fock calculation to get the molecular orbitals
mf = scf.RHF(mol)
mf.kernel()

# Obtain the one-electron (kinetic + nuclear attraction) integral matrix
h_core = mf.get_hcore()

# The molecular orbital coefficients
mo_coeff = mf.mo_coeff
print("Molecular orbitial coefficients: ", mo_coeff)

# The molecular orbital occupation numbers
mo_occ = mf.mo_occ
print("Molecular orbital occupation numbers: ", mo_occ)


# Number of electrons
nelec = mol.nelectron
print("Number of electrons: ", nelec)

# For the H2 molecule in its ground state, we consider the first two orbitals (1s orbitals for each hydrogen atom)
def slater_determinant(r):
    # Assuming the first two orbitals are occupied
    occupied_orbitals = mo_coeff[:, mo_occ > 0]

    # Evaluate the orbitals at position r
    ao_values = mol.eval_gto('GTOval_sph', r)
    mo_values = np.dot(ao_values, occupied_orbitals)

    # Construct the Slater determinant
    slater_det = np.linalg.det(mo_values[:, :nelec//2])
    return slater_det


# Example: Evaluate the wavefunction at a specific point
r = np.array([[0.0, 0.0, 0.0]])  # Position to evaluate the wavefunction
wavefunction_value = slater_determinant(r)
print("Wavefunction value at r =", r, ":", wavefunction_value)



# Convert the Hamiltonian matrix to a 1D tensor
# Flatten the matrix and ensure it's a real number tensor (if needed)
hamiltonian_coefficients = torch.tensor(h_core.flatten(), dtype=torch.float32)

# Now, hamiltonian_coefficients contains the Hamiltonian matrix coefficients
inputs= hamiltonian_coefficients
print("Hamiltonian coefficients are: ", inputs)