from pyscf import gto, scf
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse as sp

# Step 1: Define the Molecular System
# A water molecule is defined using PySCF's `gto.M` function.
mol = gto.M(atom=[
    ['O', (0.0, 0.0, 0.0)],          # Oxygen at origin
    ['H', (0.0, -0.757, 0.587)],     # Hydrogen 1
    ['H', (0.0, 0.757, 0.587)]       # Hydrogen 2
], basis='sto-3g')

# Step 2: Perform Hartree-Fock Calculation
# A restricted Hartree-Fock (RHF) calculation is performed on the molecule.
mf = scf.RHF(mol)
mf.kernel()

# Step 3: Obtain Molecular Orbitals Coefficients and Integrals
# `mo_coeff` stores the molecular orbital coefficients.
# `h1` is the core Hamiltonian matrix, containing kinetic energy and nuclear attraction integrals.
# `g2` contains the two-electron repulsion integrals over atomic orbitals.
mo_coeff = mf.mo_coeff
h1 = mf.get_hcore(mol)
g2 = mol.intor('int2e', aosym='s1')

# Step 4: Transform Two-Electron Integrals to MO Basis
# The two-electron integrals `g2` are transformed to the molecular orbital (MO) basis.
g2_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', g2, mo_coeff, mo_coeff, mo_coeff, mo_coeff)

# Step 5: Construct Hamiltonian in MO Basis
# A Hamiltonian matrix `H` is initialized with zeros.
num_orbitals = mo_coeff.shape[1]
H = np.zeros((num_orbitals, num_orbitals))

# Add the one-electron integrals (`h1`) to `H`.
for i in range(num_orbitals):
    for j in range(num_orbitals):
        H[i, j] += h1[i, j]

# Step 6: Add Two-Electron Integrals
# This nested loop adds the two-electron integrals to the Hamiltonian matrix.
for i in range(num_orbitals):
    for j in range(num_orbitals):
        for k in range(num_orbitals):
            for l in range(num_orbitals):
                H[i, j] += 0.5 * g2_mo[i, j, k, l] * ((k == l) - 0.5 * (i == l) * (j == k))

# Step 7: Account for Spin Orbitals
# The Hamiltonian matrix is expanded to include spin orbitals.
num_spin_orbitals = 2 * num_orbitals
H_spin = np.zeros((num_spin_orbitals, num_spin_orbitals))

# Each element of the original Hamiltonian is replicated in both the alpha and beta spin blocks.
for i in range(num_orbitals):
    for j in range(num_orbitals):
        H_spin[i, j] = H[i, j]
        H_spin[i + num_orbitals, j + num_orbitals] = H[i, j]

# Step 8: Scale Up to a Larger Sparse Matrix
# The Hamiltonian matrix is converted into a sparse matrix format.
H_spin_sparse = sp.csr_matrix(H_spin)

# A much larger sparse matrix (`large_H_spin_sparse`) is created.
size = 2**14
large_H_spin_sparse = sp.lil_matrix((size, size))
# Convert H_spin to a sparse matrix
H_spin_sparse = sp.csr_matrix(H_spin)

size = 2**14

num_repetitions = size // H_spin.shape[0]

# Create a list of H_spin matrices
H_spin_blocks = [H_spin] * num_repetitions

# Create the block diagonal matrix
large_H_spin_block = la.block_diag(*H_spin_blocks)

# Convert to a sparse matrix
large_H_spin_sparse_csr = sp.csr_matrix(large_H_spin_block)

# Summary
# This code provides a detailed demonstration of constructing a quantum Hamiltonian matrix
# for a small molecule, then scaling it up to a large sparse matrix.
# The large matrix is intended for demonstration and may not accurately represent
# a real quantum system of that size.
# The code combines quantum chemical calculations with matrix operations
# to model the electronic structure of a molecule.


from scipy.sparse.linalg import eigsh

# Assuming large_H_spin_sparse_csr is your large sparse Hamiltonian matrix
# Find the smallest eigenvalue
# 'which='SA'' means to find the smallest algebraic eigenvalue
eigenvalues, eigenvectors = eigsh(large_H_spin_sparse_csr, k=1, which='SA')

# Extract the smallest eigenvalue
smallest_eigenvalue = eigenvalues[0]
print("Smallest eigenvalue: for the large 2^14x2^14 matrix: ", smallest_eigenvalue)