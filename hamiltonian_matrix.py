from qiskit_nature.second_q.drivers import PySCFDriver
import torch
from pyscf import gto, scf
import numpy as np

mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')

# Perform Hartree-Fock calculation
mf = scf.RHF(mol)
mf.kernel()

# Get MO coefficients and integrals
mo_coeff = mf.mo_coeff
h1 = mf.get_hcore(mol)
g2 = mol.intor('int2e', aosym='s1')

# Transform g2 to MO basis
g2_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', g2, mo_coeff, mo_coeff, mo_coeff, mo_coeff)

# Construct the Hamiltonian matrix in MO basis
num_orbitals = mo_coeff.shape[1]
H = np.zeros((num_orbitals, num_orbitals))

# Add one-electron integrals
for i in range(num_orbitals):
    for j in range(num_orbitals):
        H[i, j] += h1[i, j]

# Add two-electron integrals
for i in range(num_orbitals):
    for j in range(num_orbitals):
        for k in range(num_orbitals):
            for l in range(num_orbitals):
                H[i, j] += 0.5 * g2_mo[i, j, k, l] * ((k == l) - 0.5 * (i == l) * (j == k))

num_spin_orbitals = 2 * num_orbitals  # Double the number of orbitals for spin
H_spin = np.zeros((num_spin_orbitals, num_spin_orbitals))

# Fill in the Hamiltonian matrix for spin orbitals
for i in range(num_orbitals):
    for j in range(num_orbitals):
        # Alpha spin block
        H_spin[i, j] = H[i, j]
        # Beta spin block
        H_spin[i + num_orbitals, j + num_orbitals] = H[i, j]
# The Hamiltonian matrix 'H' is now constructed
                
#print(H_spin)
print(len(H_spin))

import numpy as np

# Assume H_spin is already defined as your 14x14 Hamiltonian matrix
# Diagonalize the Hamiltonian matrix
eigenvalues, eigenvectors = np.linalg.eigh(H_spin)

# eigenvalues contains the energy levels
# eigenvectors contains the corresponding quantum states
print(min(eigenvalues))