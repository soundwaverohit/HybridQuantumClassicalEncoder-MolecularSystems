from pyscf import gto, scf
import numpy as np
import scipy.sparse as sp


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
print("for the small 14x14 matrix: ", min(eigenvalues))



# Convert H_spin to a sparse matrix
H_spin_sparse = sp.csr_matrix(H_spin)

size = 2**14

# Create an empty sparse matrix of the desired size
large_H_spin_sparse = sp.lil_matrix((size, size))

# Calculate the number of repetitions needed
repetitions = size // H_spin_sparse.shape[0]

# Populate the large sparse matrix with the pattern from H_spin
for i in range(repetitions):
    for j in range(repetitions):
        start_row = i * H_spin_sparse.shape[0]
        start_col = j * H_spin_sparse.shape[1]
        end_row = start_row + H_spin_sparse.shape[0]
        end_col = start_col + H_spin_sparse.shape[1]
        large_H_spin_sparse[start_row:end_row, start_col:end_col] = H_spin_sparse

# Convert to CSR format for efficient arithmetic and matrix-vector operations
large_H_spin_sparse_csr = large_H_spin_sparse.tocsr()

from scipy.sparse.linalg import eigsh

# Assuming large_H_spin_sparse_csr is your large sparse Hamiltonian matrix
# Find the smallest eigenvalue
# 'which='SA'' means to find the smallest algebraic eigenvalue
eigenvalues, eigenvectors = eigsh(large_H_spin_sparse_csr, k=1, which='SA')

# Extract the smallest eigenvalue
smallest_eigenvalue = eigenvalues[0]
print("Smallest eigenvalue: for the large 2^14x2^14 matrix: ", smallest_eigenvalue)